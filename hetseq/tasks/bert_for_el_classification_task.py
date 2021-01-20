import torch
import datasets
from datasets import ClassLabel
from transformers import BertTokenizerFast

from hetseq.tasks import Task
from hetseq.data_collator import YD_DataCollatorForELClassification
from hetseq.bert_modeling import BertConfig
from hetseq.model import BertForELClassification
from hetseq.data import BertELDataset


from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID


_EL_COLUMNS = ['input_ids', 'labels', 'token_type_ids', 'attention_mask', 'entity_labels']

_UNK_ENTITY_ID = 1
_UNK_ENTITY_NAME = 'UNK_ENT'
_EMPTY_ENTITY_ID = 0
_EMPTY_ENTITY_NAME = 'EMPTY_ENT'
_OUT_DICT_ENTITY_ID = -1
_IGNORE_CLASSIFICATION_LABEL = -100
NER_LABEL_DICT = {'B': 0, 'I': 1, 'O': 2}


class BertForELClassificationTask(Task):
    def __init__(self, args):
        super(BertForELClassificationTask, self).__init__(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        # **YD** add BertTokenizerFast to be suitable for CONLL2003 NER task, pipeline is similar to
        # https://github.com/huggingface/transformers/tree/master/examples/token-classification
        # 1.obtain tokenizer and data_collator

        tokenizer = BertTokenizerFast(args.dict)
        data_collator = YD_DataCollatorForELClassification(tokenizer, max_length=args.max_pred_length, padding=True)

        # 2. process datasets, (tokenization of NER data)
        # **YD**, add args in option.py for fine-tuning task
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.extension_file
        dataset = datasets.load_dataset(extension, data_files=data_files)

        # 3. setup num_labels
        if 'train' in dataset:
            column_names = dataset["train"].column_names
            features = dataset["train"].features
        elif 'validation' in dataset:
            column_names = dataset["validation"].column_names
            features = dataset["validation"].features
        elif 'test' in dataset:
            column_names = dataset["test"].column_names
            features = dataset["test"].features
        else:
            raise ValueError('dataset must contain "train"/"validation"/"test"')

        text_column_name = 'tokens'
        label_column_name = 'ner_tags'
        entity_column_name = 'entity_names'

        assert text_column_name in column_names
        assert label_column_name in column_names
        assert entity_column_name in column_names

        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            if 'train' in label_column_name:
                label_list = get_label_list(datasets["train"][label_column_name])
            elif 'validation' in label_column_name:
                label_list = get_label_list(datasets["validation"][label_column_name])
            elif 'test' in label_column_name:
                label_list = get_label_list(datasets["test"][label_column_name])
            else:
                raise ValueError('dataset must contain "train"/"validation"/"test"')

            label_to_id = {l: i for i, l in enumerate(label_list)}

        num_labels = len(label_list)

        # **YD** preparing ent_name_id from deep_ed to
        # transform (entity name or entity wikiid) to thid (entity embedding lookup index)
        ent_name_id = EntNameID(args)

        # 4. tokenization
        # Tokenize all texts and align the labels with them.
        # Tokenize all texts and align the **NER_labels and Entity_labels** with them.
        # **YD** only mention (GT label= 'B' or 'I') is considered to do entity disambiguation task.
        # in training time, if certain entity in dictionary, it is labeled with correct entity id.
        # if certain entity is not in dictionary, or certain mention has no corresponding entity,
        # it is labelled with incorrect entity.

        # in inference time, NER label together with ED label to do evaluation.
        # if certain token labels with 'B' and has not unknown predicted entity, it is predicted with entity. The mention
        # part is decided with the following 'I' label.
        # otherwise, if it has unknown predicted entity, all 'B' and following 'I' becomes 'O' label.
        def tokenize_and_align_labels(examples, label_all_tokens=False):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=False,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
                return_offsets_mapping=True,
            )
            # print('tokenized_inputs', tokenized_inputs)

            offset_mappings = tokenized_inputs.pop("offset_mapping")
            labels = []
            entity_labels = []
            for label, offset_mapping, entity_label in zip(examples[label_column_name], offset_mappings,
                                                           examples[entity_column_name]):

                label_index = 0
                current_label = -100
                label_ids = []

                current_entity_label = -100
                entity_label_ids = []
                for offset in offset_mapping:
                    # We set the label for the first token of each word.
                    # Special characters will have an offset of (0, 0)
                    # so the test ignores them.
                    if offset[0] == 0 and offset[1] != 0:
                        current_label = label_to_id[label[label_index]]
                        label_index += 1
                        label_ids.append(current_label)

                        current_entity_label = entity_label[label_index - 1]

                        if label[label_index - 1] == NER_LABEL_DICT['O']:
                            current_entity_label = -100
                        else:
                            # print(label[label_index-1])
                            # print(label_to_id)
                            assert label[label_index - 1] == NER_LABEL_DICT['B'] or label[label_index - 1] == \
                                   NER_LABEL_DICT['I']

                            if current_entity_label == _EMPTY_ENTITY_NAME or label[label_index - 1] == NER_LABEL_DICT['I']:
                                current_entity_label = -100
                            else:
                                assert label[label_index - 1] == NER_LABEL_DICT['B']
                                tmp_label = ent_name_id.get_thid(
                                    ent_name_id.get_ent_wikiid_from_name(current_entity_label, True)
                                )
                                if tmp_label != ent_name_id.unk_ent_thid:
                                    current_entity_label = tmp_label
                                else:
                                    current_entity_label = _OUT_DICT_ENTITY_ID

                        entity_label_ids.append(current_entity_label)
                    # For special tokens, we set the label to -100 so it's automatically ignored in the loss function.
                    elif offset[0] == 0 and offset[1] == 0:
                        label_ids.append(-100)
                        entity_label_ids.append(-100)
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(current_label if label_all_tokens else -100)
                        entity_label_ids.append(current_entity_label if label_all_tokens else -100)

                labels.append(label_ids)
                entity_labels.append(entity_label_ids)

            tokenized_inputs["labels"] = labels
            tokenized_inputs["entity_labels"] = entity_labels

            return tokenized_inputs

        tokenized_datasets = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=1,  # set 1 for faster processing
            load_from_cache_file=False,
        )

        # 5. set up dataset format and input/output pipeline of dataset
        tokenized_datasets.set_format(type='torch', columns=_EL_COLUMNS)

        # include components in args
        args.tokenized_datasets = tokenized_datasets
        args.num_labels = num_labels
        args.tokenizer = tokenizer
        args.data_collator = data_collator

        # load entity embedding and set up shape parameters
        args.EntityEmbedding = torch.load(args.ent_vecs_filename, map_location='cpu')
        args.num_entity_labels = args.EntityEmbedding.shape[0]
        args.dim_entity_emb = args.EntityEmbedding.shape[1]

        return cls(args)

    def build_model(self, args):
        if args.task == 'BertForELClassification':
            # obtain num_label from dataset before assign model
            config = BertConfig.from_json_file(args.config_file)
            # **YD** mention detection, num_label is by default 3
            assert hasattr(args, 'num_labels')
            assert hasattr(args, 'num_entity_labels')
            assert hasattr(args, 'dim_entity_emb')
            assert hasattr(args, 'EntityEmbedding')

            model = BertForELClassification(config, args)

            # **YD** add load state_dict from pre-trained model
            # could make only master model to load from state_dict, not quite sure whether this works for single GPU
            # if distributed_utils.is_master(args) and args.hetseq_state_dict is not None:
            if args.hetseq_state_dict is not None:
                state_dict = torch.load(args.hetseq_state_dict, map_location='cpu')['model']
                if args.load_state_dict_strict:
                    model.load_state_dict(state_dict, strict=True)
                else:
                    model.load_state_dict(state_dict, strict=False)

            elif args.transformers_state_dict is not None:
                state_dict = torch.load(args.transformers_state_dict, map_location='cpu')
                if args.load_state_dict_strict:
                    model.load_state_dict(state_dict, strict=True)
                else:
                    model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError('Unknown fine_tunning task!')
        return model

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        """
        assert split in ['train', 'validation', 'test']
        if split in self.datasets:
            return self.datasets[split]
        """
        if split in self.datasets:
            return

        if 'train' in self.args.tokenized_datasets:
            self.datasets['train'] = BertELDataset(self.args.tokenized_datasets['train'], self.args)
        elif 'validation' in dataset:
            self.datasets['valid'] = BertELDataset(self.args.tokenized_datasets['validation'], self.args)
        elif 'test' in dataset:
            self.datasets['test'] = BertELDataset(self.args.tokenized_datasets['test'], self.args)
        else:
            raise ValueError('dataset must contain "train"/"validation"/"test"')

        print('| loading finished')

    # **YD** may need to write customized train_step, we will see
    def train_step(self, sample, model, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~torch.utils.data.Dataset`.
            model (~torch.nn.Module): the model
            optimizer (~optim._Optimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()

        loss = model(**sample)
        if ignore_grad:
            loss *= 0
        if sample is None or len(sample['labels']) == 0:
            sample_size = 0
        else:
            sample_size = len(sample['labels'])

        nsentences = sample_size

        logging_output = {
            'nsentences': nsentences,
            'loss': loss.data,
            'nll_loss': loss.data,
            'ntokens': 0,
            'sample_size': sample_size,
        }

        optimizer.backward(loss)
        return loss, sample_size, logging_output
