import torch
import datasets
from datasets import ClassLabel

import transformers
from transformers import BertTokenizerFast

from hetseq.tasks import Task
from hetseq.data import BertELDataset

from hetseq.data_collator import YD_DataCollatorForELClassification


_EL_COLUMNS = [
    "input_ids",
    "token_type_ids",
    "attention_mask",

    "entity_th_ids",
    "left_mention_masks",
    "right_mention_masks",
    "left_entity_masks",
    "right_entity_masks",
]


class BertForELSymmetryTask(Task):
    def __init__(self, args):
        super(BertForELsymmetryTask, self).__init__(args)

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

        # **YD** TODO: write a new data collator
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


        # 4. tokenization
        # Tokenize all texts and align the labels with them.
        # Tokenize all texts and align the **NER_labels and Entity_labels** with them.
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"],
                padding=False,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
                return_offsets_mapping=True,
            )

            offset_mappings = tokenized_inputs.pop("offset_mapping")
            ori_entity_th_ids = examples["ori_entity_th_ids"]
            ori_left_mention_masks = examples["left_mention_masks"]
            ori_right_mention_masks = examples["right_mention_masks"]
            ori_left_entity_masks = examples["left_entity_masks"]
            ori_right_entity_masks = examples["right_entity_masks"]

            entity_th_ids = []
            left_mention_masks = []
            right_mention_masks = []
            left_entity_masks = []
            right_entity_masks = []

            for (
                    offset_mapping,
                    ori_entity_th_id,
                    ori_left_mention_mask,
                    ori_right_mention_mask,
                    ori_left_entity_mask,
                    ori_right_entity_mask,
            ) in zip(
                offset_mappings,
                ori_entity_th_ids,
                ori_left_mention_masks,
                ori_right_mention_masks,
                ori_left_entity_masks,
                ori_right_entity_masks,
            ):

                label_index = 0

                entity_th_id = []
                left_mention_mask = []
                right_mention_mask = []
                left_entity_mask = []
                right_entity_mask = []

                for offset in offset_mapping:
                    # (0, 0) represents special tokens.
                    # (0, x) represents first sub-word of a token
                    # other represents second or more sub-word of a token (ignore in our pre-processing)
                    if offset[0] == 0 and offset[1] != 0:
                        entity_th_id.append(ori_entity_th_id[label_index])
                        left_mention_mask.append(ori_left_mention_mask[label_index])
                        right_mention_mask.append(ori_right_mention_mask[label_index])
                        left_entity_mask.append(ori_left_entity_mask[label_index])
                        right_entity_mask.append(ori_right_entity_mask[label_index])

                        label.append(ori_label[label_index])
                        label_index += 1

                    elif offset[0] == 0 and offset[1] == 0:
                        entity_th_id.append(-100)
                        left_mention_mask.append(-100)
                        right_mention_mask.append(-100)
                        left_entity_mask.append(-100)
                        right_entity_mask.append(-100)

                    else:
                        entity_th_id.append(-100)
                        left_mention_mask.append(-100)
                        right_mention_mask.append(-100)
                        left_entity_mask.append(-100)
                        right_entity_mask.append(-100)

                entity_th_ids.append(entity_th_id)
                left_mention_masks.append(left_mention_mask)
                right_mention_masks.append(right_mention_mask)
                left_entity_masks.append(left_entity_mask)
                right_entity_masks.append(right_entity_mask)

            tokenized_inputs["entity_th_ids"] = entity_th_ids
            tokenized_inputs["left_mention_masks"] = left_mention_masks
            tokenized_inputs["right_mention_masks"] = right_mention_masks
            tokenized_inputs["left_entity_masks"] = left_entity_masks
            tokenized_inputs["right_entity_masks"] = right_entity_masks

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
        args.tokenizer = tokenizer
        args.data_collator = data_collator

        # load entity embedding and set up shape parameters
        args.EntityEmbedding = torch.load(args.ent_vecs_filename, map_location='cpu')
        args.num_entity_labels = args.EntityEmbedding.shape[0]
        args.dim_entity_emb = args.EntityEmbedding.shape[1]

        return cls(args)

    def build_model(self, args):
        if args.task == 'BertForELSymmetry':

            model = None
            # **YD** add load state_dict from pre-trained model
            # could make only master model to load from state_dict, not quite sure whether this works for single GPU
            # if distributed_utils.is_master(args) and args.hetseq_state_dict is not None:
            if args.hetseq_state_dict is not None:
                from hetseq.bert_modeling import BertConfig
                from hetseq.model import BertForELClassification

                config = BertConfig.from_json_file(args.config_file)

                model = BertForELClassification(config)
                state_dict = torch.load(args.hetseq_state_dict, map_location='cpu')['model']
                if args.load_state_dict_strict:
                    model.load_state_dict(state_dict, strict=True)
                else:
                    model.load_state_dict(state_dict, strict=False)

            elif args.transformers_state_dict is not None:
                from transformers import BertConfig
                from hetseq.model import TransformersBertForELClassification
                config = BertConfig.from_json_file(args.config_file)

                model = TransformersBertForELClassification(config, args)
                state_dict = torch.load(args.transformers_state_dict, map_location='cpu')
                if args.load_state_dict_strict:
                    model.load_state_dict(state_dict, strict=True)
                else:
                    model.load_state_dict(state_dict, strict=False)
            else:
                model_class = getattr(args, 'model_class', 'TransformersBertForNERSymmetry')
                if model_class == 'TransformersBertForNERSymmetry':
                    from hetseq.model import TransformersBertForNERSymmetry
                    from transformers import BertConfig
                    config = BertConfig.from_json_file(args.config_file)

                    print('backbones', args.backbones)
                    model = TransformersBertForNERSymmetry.from_pretrained(
                        args.backbones, config=config, args=args,
                    )
                elif model_class == 'TransformersBertForELSymmetry':
                    from hetseq.model import TransformersBertForELSymmetry
                    from transformers import BertConfig
                    config = BertConfig.from_json_file(args.config_file)

                    print('backbones', args.backbones)
                    model = TransformersBertForELSymmetry.from_pretrained(
                        args.backbones, config=config, args=args,
                    )

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
