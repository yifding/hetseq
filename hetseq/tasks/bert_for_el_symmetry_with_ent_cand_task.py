import numpy as np

import torch
import datasets
from datasets import ClassLabel

import transformers
from transformers import BertTokenizerFast

from hetseq.tasks import Task
from hetseq.data import BertELDataset

from hetseq.data_collator import DataCollatorForSymmetry, DataCollatorForSymmetryWithCandEntities



_EL_COLUMNS = [
    "input_ids",
    "token_type_ids",
    "attention_mask",

    "entity_th_ids",
    "left_entity_masks",
    "right_entity_masks",

    "span_masks",
    "span_cand_entities",

]


class BertForELSymmetryWithEntCandTask(Task):
    def __init__(self, args):
        super(BertForELSymmetryWithEntCandTask, self).__init__(args)

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
        data_collator = DataCollatorForSymmetryWithCandEntities(tokenizer, max_length=args.max_pred_length, padding=True)

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
            ori_entity_th_ids = examples["entity_th_ids"]
            ori_left_entity_masks = examples["left_entity_masks"]
            ori_right_entity_masks = examples["right_entity_masks"]

            ori_span_masks = examples["span_masks"]
            ori_span_cand_entities = examples["span_cand_entities"]


            entity_th_ids = []
            left_entity_masks = []
            right_entity_masks = []

            span_masks = []
            span_cand_entities = []

            for (
                offset_mapping,
                ori_entity_th_id,
                ori_left_entity_mask,
                ori_right_entity_mask,
                ori_span_mask,
                ori_span_cand_entity,
            ) in zip(
                offset_mappings,
                ori_entity_th_ids,
                ori_left_entity_masks,
                ori_right_entity_masks,
                ori_span_masks,
                ori_span_cand_entities,
            ):

                label_index = 0

                entity_th_id = []
                left_entity_mask = []
                right_entity_mask = []

                span_mask = []
                span_cand_entity = []
                # len_token
                assert len(ori_span_mask) == len(ori_span_cand_entity)
                # max_len_mention
                assert len(ori_span_mask[0]) == len(ori_span_cand_entity[0])
                len_token, max_len_mention, num_cand_ent = \
                    len(ori_span_cand_entity), len(ori_span_cand_entity[0]), len(ori_span_cand_entity[0][0])
                pad_span_mask = [0 for _ in range(max_len_mention)]
                pad_span_cand_entity = [[0 for _ in range(num_cand_ent)] for _ in range(max_len_mention)]

                for offset in offset_mapping:
                    # (0, 0) represents special tokens.
                    # (0, x) represents first sub-word of a token
                    # other represents second or more sub-word of a token (ignore in our pre-processing)
                    if offset[0] == 0 and offset[1] != 0:
                        entity_th_id.append(ori_entity_th_id[label_index])
                        left_entity_mask.append(ori_left_entity_mask[label_index])
                        right_entity_mask.append(ori_right_entity_mask[label_index])

                        span_mask.append(ori_span_mask[label_index])
                        span_cand_entity.append(ori_span_cand_entity[label_index])

                        label_index += 1

                    elif offset[0] == 0 and offset[1] == 0:
                        entity_th_id.append(-100)
                        left_entity_mask.append(-100)
                        right_entity_mask.append(-100)

                        span_mask.append(list(pad_span_mask))
                        span_cand_entity.append(list(pad_span_cand_entity))

                    else:
                        entity_th_id.append(-100)
                        left_entity_mask.append(-100)
                        right_entity_mask.append(-100)

                        span_mask.append(list(pad_span_mask))
                        span_cand_entity.append(list(pad_span_cand_entity))

                entity_th_ids.append(entity_th_id)
                left_entity_masks.append(left_entity_mask)
                right_entity_masks.append(right_entity_mask)

                span_masks.append(span_mask)
                span_cand_entities.append(span_cand_entity)

            tokenized_inputs["entity_th_ids"] = entity_th_ids
            tokenized_inputs["left_entity_masks"] = left_entity_masks
            tokenized_inputs["right_entity_masks"] = right_entity_masks

            tokenized_inputs["span_masks"] = span_masks
            tokenized_inputs["span_cand_entities"] = span_cand_entities

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
        if args.task == 'BertForELSymmetryWithEntCand':

            model = None
            # **YD** add load state_dict from pre-trained model
            # could make only master model to load from state_dict, not quite sure whether this works for single GPU
            # if distributed_utils.is_master(args) and args.hetseq_state_dict is not None:
            model_class = getattr(args, 'model_class', 'TransformersBertForELSymmetry')
            if  model_class == 'TransformersBertForELSymmetry':
                from hetseq.model import TransformersBertForELSymmetry
                from transformers import BertConfig
                config = BertConfig.from_json_file(args.config_file)

                print('backbones', args.backbones)
                model = TransformersBertForELSymmetry.from_pretrained(
                    args.backbones, config=config, args=args,
                )

            else:
                raise ValueError('Unknown fine_tunning task!')

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
        if sample is None or len(sample['entity_th_ids']) == 0:
            sample_size = 0
        else:
            sample_size = len(sample['entity_th_ids'])

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
