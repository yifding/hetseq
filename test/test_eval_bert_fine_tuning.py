import os
import argparse

from datasets import load_dataset, ClassLabel

from hetseq.bert_modeling import (
    BertConfig,
    BertForPreTraining,
    BertForTokenClassification,
)
from hetseq.data_collator import YD_DataCollatorForTokenClassification

from transformers import (
    BertTokenizerFast,
    # DataCollatorForTokenClassification,
)

import torch
from torch.utils.data.dataloader import DataLoader


__DATE__ = '2020/1/3'
__AUTHOR__ = 'Yifan_Ding'
__E_MAIL = 'dyf0125@gmail.edu'

_NER_COLUMNS = ['input_ids', 'labels', 'token_type_ids', 'attention_mask']

# 1. load dataset, either from CONLL2003 or other entity linking datasets.
# 2. load model from a trained one, load model architecture and state_dict from a trained one.
# 3. predict loss, generate predicted label
# 4. compare predicted label with ground truth label to obtain evaluation results.


def main(args):
    dataset = prepare_dataset(args)
    tokenizer = prepare_tokenizer(args)
    data_collator = YD_DataCollatorForTokenClassification(tokenizer)

    if 'test' in dataset:
        column_names = dataset["test"].column_names
        features = dataset["test"].features
    else:
        raise ValueError('Evaluation must specify test_file!')

    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = ('ner_tags' if 'ner_tags' in column_names else column_names[1])

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        if 'test' in label_column_name:
            label_list = get_label_list(datasets["test"][label_column_name])
        else:
            raise ValueError('Evaluation must specify test_file!')

        label_to_id = {l: i for i, l in enumerate(label_list)}
    args.num_labels = len(label_list)

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, label_all_tokens=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_offsets_mapping=True,
        )
        #print('tokenized_inputs', tokenized_inputs)

        offset_mappings = tokenized_inputs.pop("offset_mapping")
        labels = []
        for label, offset_mapping in zip(examples[label_column_name], offset_mappings):
            label_index = 0
            current_label = -100
            label_ids = []
            for offset in offset_mapping:
                # We set the label for the first token of each word. Special characters will have an offset of (0, 0)
                # so the test ignores them.
                if offset[0] == 0 and offset[1] != 0:
                    current_label = label_to_id[label[label_index]]
                    label_index += 1
                    label_ids.append(current_label)
                # For special tokens, we set the label to -100 so it's automatically ignored in the loss function.
                elif offset[0] == 0 and offset[1] == 0:
                    label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(current_label if label_all_tokens else -100)

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        # batched=True, **YD** test non-batched results
        batched=True,
        num_proc=8,
        load_from_cache_file=False,
    )

    test_dataset = tokenized_datasets['test']
    # **YD** core code to keep only usefule parameters for model
    test_dataset.set_format(type=test_dataset.format["type"], columns=_NER_COLUMNS)

    # **YD** dataloader
    data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=None,
        collate_fn=data_collator,
        drop_last=False,
        # num_workers=8,
        num_workers=0,
    )

    model = prepare_model(args)
    for index, input in enumerate(data_loader):
        if index == 0:
            output = model(**input)
            print('output', output)

def prepare_dataset(args):
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
    else:
        raise ValueError('Evaluation must specify test_file!')

    extension = args.extension_file
    print('extension', extension)
    print('data_files', data_files)
    dataset = load_dataset(extension, data_files=data_files)
    return dataset


def prepare_tokenizer(args):
    config = BertConfig.from_json_file(args.config_file)
    # tokenizer = BertTokenizerFast(args.vocab_file, model_max_length=512)
    # print('config', type(config), config,)
    tokenizer = BertTokenizerFast(args.vocab_file, model_max_length=config.max_position_embeddings)
    return tokenizer


def prepare_model(args):
    config = BertConfig.from_json_file(args.config_file)
    model = BertForTokenClassification(config, args.num_labels)
    if args.hetseq_state_dict != '':
        # load hetseq state_dictionary
        model.load_state_dict(torch.load(args.hetseq_state_dict, map_location='cpu')['model'], strict=True)

    return model

def cli_main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # **YD** datasets argument
    parser.add_argument(
        '--train_file',
        help='local training file for BERT-ner fine-tuning',
        type=str,
    )

    parser.add_argument(
        '--validation_file',
        help='local validation file for BERT-ner fine-tuning',
        type=str,
    )

    parser.add_argument(
        '--test_file',
        help='local test file for BERT-ner fine-tuning',
        type=str,
    )

    parser.add_argument(
        '--extension_file',
        help='local extension file for building dataset similar to conll2003 using "datasets" package',
        type=str,
    )

    # **YD** BERT model related argument
    parser.add_argument(
        '--config_file',
        help='BERT config_file',
        type=str,
    )

    parser.add_argument(
        '--vocab_file',
        help='BERT vocabulary file',
        type=str,
    )

    parser.add_argument(
        '--hetseq_state_dict',
        help='hetseq pre-trained state dictionary ',
        type=str,
        default='',
    )

    parser.add_argument(
        '--batch_size',
        help='batch size for data_loader',
        type=int,
        default=8,
    )

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()