import os
import argparse

from datasets import load_dataset, ClassLabel

from hetseq.bert_modeling import (
    BertConfig,
    BertForPreTraining,
    BertForTokenClassification,
)

from transformers import (
    BertTokenizerFast,
    DataCollatorForTokenClassification,
)

import torch
from torch.utils.data.dataloader import DataLoader

from hetseq.data_collator import YD_DataCollatorForELClassification

_NER_COLUMNS = ['input_ids', 'labels', 'token_type_ids', 'attention_mask', "entity_labels"]

_UNK_ENTITY_ID = 1
_UNK_ENTITY_NAME = 'UNK_ENT'
_EMPTY_ENTITY_ID = 0
_EMPTY_ENTITY_NAME = 'EMPTY_ENT'
NER_LABEL_DICT = {'B': 0, 'I':1, 'O':2}

def main(args):
    # 1. prepare dataset from customized dataset
    dataset = prepare_dataset(args)

    # 2. prepare tokenizer from customized dictionary and build data_collator
    tokenizer = prepare_tokenizer(args)
    # data_collator = DataCollatorForTokenClassification(tokenizer)
    data_collator = YD_DataCollatorForELClassification(tokenizer)
    args.data_collator = data_collator

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

    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = 'ner_tags' if 'ner_tags' in column_names else column_names[1]
    entity_column_name = "entity_names" if "entity_names" in column_names else column_names[2]

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

    args.num_labels = len(label_list)


    # **YD** modify original tokenize function to include entities
    """
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
    """

    # **YD** preparing ent_name_id from deep_ed to
    # transform (entity name or entity wikiid) to thid (entity embedding lookup index)
    from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID
    ent_name_id = EntNameID(args)

    # Tokenize all texts and align the **NER_labels and Entity_labels** with them.
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
        entity_labels = []
        for label, offset_mapping, entity_label in zip(examples[label_column_name], offset_mappings, examples[entity_column_name]):
            label_index = 0
            current_label = -100
            label_ids = []

            current_entity_label = -100
            entity_label_ids = []
            for offset in offset_mapping:
                # We set the label for the first token of each word. Special characters will have an offset of (0, 0)
                # so the test ignores them.
                if offset[0] == 0 and offset[1] != 0:
                    current_label = label_to_id[label[label_index]]
                    label_index += 1
                    label_ids.append(current_label)

                    current_entity_label = entity_label[label_index-1]

                    if label[label_index-1] == NER_LABEL_DICT['O']:
                        current_entity_label = ent_name_id.unk_ent_thid
                    else:
                        # print(label[label_index-1])
                        # print(label_to_id)
                        assert label[label_index-1] == NER_LABEL_DICT['B'] or label[label_index-1] == NER_LABEL_DICT['I']

                        if current_entity_label == _EMPTY_ENTITY_NAME:
                            current_entity_label = ent_name_id.unk_ent_thid
                        else:
                            tmp_label = ent_name_id.get_thid(
                                ent_name_id.get_ent_wikiid_from_name(current_entity_label, True)
                            )
                            if tmp_label != ent_name_id.unk_ent_thid:
                                current_entity_label = tmp_label
                            else:
                                current_entity_label = -1

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
        # batched=True, **YD** test non-batched results
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
    )


    # print(tokenized_datasets['train'])
    # print(tokenized_datasets['train'][0])
    train_dataset = tokenized_datasets['train']

    # **YD** core code to keep only usefule parameters for model
    train_dataset.set_format(type=train_dataset.format["type"], columns=_NER_COLUMNS)


    # **YD** dataloader
    data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        #sampler=train_sampler,
        sampler=None,
        collate_fn=data_collator,
        drop_last=False,
        # num_workers=8,
        num_workers=0,
    )

    # 3. prepare bert-model loading pre-trained checkpoint
    # **YD** num_class is defined by datasets, define model after datasets, in hetseq may require pass extra parameters
    # model = prepare_model(args)

    for index, input in enumerate(data_loader):
        if index == 0:
            print('input.keys()', input.keys(), input.items())
            # print(model(**input))
        print('input_ids shape', input['input_ids'].shape, 'labels shape', input['labels'].shape)
        if index == 10:
            break
    """
    # test customized dataset
    from hetseq.data import BertNerDataset
    cus_dataset = BertNerDataset(train_dataset, args)
    print('len(cus_dataset)', len(cus_dataset))
    data_loader = DataLoader(
        cus_dataset,
        batch_size=args.batch_size,
        # sampler=train_sampler,
        sampler=None,
        collate_fn=args.data_collator,
        drop_last=False,
        # num_workers=8,
        num_workers=0,
    )
    model = prepare_model(args)
    for index, input in enumerate(data_loader):
        if index == 0:
            print('input.keys()', input.keys(), input.items())
            print(model(**input))
        print('input_ids shape', input['input_ids'].shape, 'labels shape', input['labels'].shape)
        if index == 10:
            break
    """


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
        model.load_state_dict(torch.load(args.hetseq_state_dict, map_location='cpu')['model'], strict=False)
    elif args.transformers_state_dict != '':
        model.load_state_dict(torch.load(args.transformers_state_dict, map_location='cpu'), strict=False)

    return model


def prepare_dataset(args):
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file

    extension = args.extension_file
    print('extension', extension)
    print('data_files', data_files)
    dataset = load_dataset(extension, data_files=data_files)

    return dataset


def cli_main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # **YD** datasets argument
    parser.add_argument(
        '--train_file',
        help='local training file for BERT-ner fine-tuning',
        type=str,
        default='/home/yding4/EL_resource/data/raw/AIDA-CONLL/aida_train.txt',
    )

    parser.add_argument(
        '--validation_file',
        help='local validation file for BERT-ner fine-tuning',
        type=str,
        default='/home/yding4/EL_resource/data/raw/AIDA-CONLL/testa_testb_aggregate_original',
    )

    parser.add_argument(
        '--test_file',
        help='local test file for BERT-ner fine-tuning',
        type=str,
        default='/home/yding4/EL_resource/data/raw/AIDA-CONLL/testa_testb_aggregate_original',
    )

    parser.add_argument(
        '--extension_file',
        help='local extension file for building dataset similar to conll2003 using "datasets" package',
        type=str,
        default='/home/yding4/EL_resource/preprocess/build_EL_datasets_huggingface/BIO_EL_aida.py',
    )

    # **YD** BERT model related argument
    parser.add_argument(
        '--config_file',
        help='BERT config_file',
        type=str,
        default='/home/yding4/hetseq/preprocessing/uncased_L-12_H-768_A-12/bert_config.json',
    )

    parser.add_argument(
        '--vocab_file',
        help='BERT vocabulary file',
        type=str,
        default='/home/yding4/hetseq/preprocessing/uncased_L-12_H-768_A-12/vocab.txt',
    )

    parser.add_argument(
        '--hetseq_state_dict',
        help='hetseq pre-trained state dictionary ',
        type=str,
        default='',
    )

    parser.add_argument(
        '--transformers_state_dict',
        help='transformers official pre-trained state dictionary ',
        type=str,
        default='',
    )

    # **YD** deep_ed arguments
    parser.add_argument(
        '--root_data_dir',
        type=str,
        default='/home/yding4/EL_resource/data/deep_ed_PyTorch_data/',
        help='Root path of the data, $DATA_PATH.',
    )

    parser.add_argument(
        '--entities',
        type=str,
        default='RLTD',
        choices=['RLTD', '4EX', 'ALL'],
        help='Set of entities for which we train embeddings: 4EX (tiny, for debug) |'
             ' RLTD (restricted set) | ALL (all Wiki entities, too big to fit on a single GPU)',
    )

    parser.add_argument(
        '--ent_vecs_filename',
        type=str,
        default='/home/yding4/EL_resource/data/deep_ed_PyTorch_data/generated/ent_vecs/ent_vecs__ep_9.pt',
        help='entity embedding file for given dictionary',
    )

    # **YD** control argument
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