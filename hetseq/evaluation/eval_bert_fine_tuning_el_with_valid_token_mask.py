import os
import argparse

from datasets import load_dataset, ClassLabel


# **YD** import deep_ed_PyTorch modules, to build valid mention mask
from deep_ed_PyTorch.words import StopWords
from deep_ed_PyTorch.data_gen.indexes import YagoCrosswikisWiki
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID

from hetseq.bert_modeling import (
    BertConfig,
    BertForPreTraining,
    # BertForTokenClassification,
)
from hetseq import utils
from hetseq.model import BertForELClassification
from hetseq.data_collator import YD_DataCollatorForELClassification

from transformers import (
    BertTokenizerFast,
    # DataCollatorForTokenClassification,
)

import torch
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

_EL_COLUMNS = [
    'input_ids',
    'labels',
    'token_type_ids',
    'attention_mask',
    'entity_labels',
    # 'tokens',
    # 'valid_mention_masks',
]

NER_LABEL_DICT = {'B': 0, 'I': 1, 'O': 2}
LABEL_LIST = {0: 'B', 1: 'I', 2: 'O'}
_UNK_ENTITY_ID = -2
_UNK_ENTITY_NAME = 'UNK_ENT'
_EMPTY_ENTITY_ID = 0
_EMPTY_ENTITY_NAME = 'EMPTY_ENT'
_OUT_DICT_ENTITY_ID = -1
_OUT_DICT_ENTITY_NAME = 'OUT_ENT'
_IGNORE_CLASSIFICATION_LABEL = -100



# 1. load dataset, either from CONLL2003 or other entity linking datasets.
# 2. load model from a trained one, load model architecture and state_dict from a trained one.
# 3. predict loss, generate predicted label
# 4. compare predicted label with ground truth label to obtain evaluation results.


def main(args):

    #stop_words = StopWords()
    ent_name_id = EntNameID(args)
    yago_crosswikis_wiki = YagoCrosswikisWiki(args)

    dataset = prepare_dataset(args)
    tokenizer = prepare_tokenizer(args)
    data_collator = YD_DataCollatorForELClassification(tokenizer)
    # data_collator = DataCollatorForTokenClassification(tokenizer)

    if 'test' in dataset:
        column_names = dataset["test"].column_names
        features = dataset["test"].features
    else:
        raise ValueError('Evaluation must specify test_file!')

    # **YD** valid_token_mask requires
    # 1. GT EL labels require:
    #   a. GT NER labels
    #   b. GT ED labels
    # 2. predict EL labels require:
    #   a. tokens
    #   b. valid_mention_mask
    #   c. NER predicted labels
    #   d. ED predict scores (a vector)

    text_column_name = 'tokens'
    label_column_name = 'ner_tags'
    entity_column_name = 'entity_th_ids'
    valid_mention_masks_column_name = 'valid_mention_masks'

    assert text_column_name in column_names
    assert label_column_name in column_names
    assert entity_column_name in column_names
    assert valid_mention_masks_column_name in column_names

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
    # **YD** preparing ent_name_id from deep_ed to
    # transform (entity name or entity wikiid) to thid (entity embedding lookup index)
    # ent_name_id = EntNameID(args)

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
        for offset_mapping, label, entity_label in zip(
                offset_mappings,
                examples[label_column_name],
                examples[entity_column_name],
        ):

            label_index = 0
            current_label = -100
            label_ids = []

            current_entity_label = -100
            entity_label_ids = []
            for offset in offset_mapping:
                # We set the label for the first token of each word.
                # Special characters will have an offset of (0, 0)
                # so the test ignores them.
                # print('offset', offset)
                if offset[0] == 0 and offset[1] != 0:
                    current_label = label_to_id[label[label_index]]
                    current_entity_label = entity_label[label_index]
                    label_index += 1
                    label_ids.append(current_label)
                    entity_label_ids.append(current_entity_label)

                # For special tokens, we set the label to -100 so it's automatically ignored in the loss function.
                elif offset[0] == 0 and offset[1] == 0:
                    label_ids.append(-100)
                    entity_label_ids.append(-100)

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

    # test_dataset = tokenized_datasets['test']
    test_dataset = tokenized_datasets[args.split]

    print(test_dataset)
    # **YD** core code to keep only usefule parameters for model
    # **YD** change the model forward function to allow extra useless parameters.
    # test_dataset.set_format(type=test_dataset.format["type"], columns=_EL_COLUMNS)

    # **YD** dataloader, set batch_size to 1 to recover other information at the same time.
    data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=1,
    )

    # load entity embedding and set up shape parameters
    args.EntityEmbedding = torch.load(args.ent_vecs_filename, map_location='cpu')
    args.num_entity_labels = args.EntityEmbedding.shape[0]
    args.dim_entity_emb = args.EntityEmbedding.shape[1]

    model = prepare_model(args)
    model.cuda()

    # **YD** starts with capital represents outputs variables
    NER_predictions = []
    NER_labels = []

    EL_predictions = []
    EL_labels = []

    for index, input in tqdm(enumerate(data_loader)):
        # **YD** all the elements of input are list of list
        labels, entity_labels = input['labels'], input['entity_labels']

        # **YD** inference signal
        input['labels'] = None

        # **YD** select the only batch and remove first and last special token
        labels = labels[0][1:-1].tolist()
        entity_labels = entity_labels[0][1:-1].tolist()

        # **YD** recover tokens string and valid mention mask.
        dataset_element = test_dataset[index]
        tokens = dataset_element['tokens']
        valid_mention_masks = dataset_element['valid_mention_masks']

        # assert the removed special tokens length = length of direct load ones
        # assert len(labels) == len(entity_labels) == len(tokens) == len(valid_mention_masks)

        # print(input.keys())
        input = utils.move_to_cuda(input)
        predictions, entity_predictions = model(**input)

        # **YD** select the only batch and remove first and last special token
        predictions = torch.argmax(predictions, axis=2)[0][1:-1].tolist()
        # **YD** left the only tensor to perform look ups for valid entity candidates
        entity_predictions = entity_predictions[0][1:-1]

        # shrink the size follow the labels
        entity_predictions = [entity_prediction for entity_prediction, label in zip(entity_predictions, labels) if label != -100]
        predictions = [prediction for prediction, label in zip(predictions, labels) if label != -100]
        entity_labels = [entity_label for entity_label, label in zip(entity_labels, labels) if label != -100]
        labels = [label for label in labels if label != -100]

        # double check the length
        if not(len(labels) == len(entity_labels) == len(predictions) == len(entity_predictions) == len(tokens) == len(valid_mention_masks)):
            print('labels', len(labels), labels)
            print('entity_labels', len(entity_labels), entity_labels)
            print('predictions', len(predictions), predictions)
            print('entity_predictions', len(entity_predictions), entity_predictions)
            print('tokens', len(tokens), tokens)
            print('valid_mention_masks', len(valid_mention_masks), valid_mention_masks)

        assert len(labels) == len(entity_labels) == len(predictions) == len(entity_predictions) == len(tokens) == len(valid_mention_masks)

        NER_label = generate_NER_label(labels)
        EL_label = generate_EL_label(entity_labels, labels)

        NER_prediction = generate_NER_prediction(predictions, labels)
        EL_prediction = generate_EL_prediction(entity_predictions, tokens, valid_mention_masks, predictions, labels, ent_name_id, yago_crosswikis_wiki, args)

        NER_labels.append(NER_label)
        EL_labels.append(EL_label)

        NER_predictions.append(NER_prediction)
        EL_predictions.append(EL_prediction)

    print(
        {
            "EL_accuracy_score": accuracy_score(EL_labels, EL_predictions),
            "EL_precision": precision_score(EL_labels, EL_predictions),
            "EL_recall": recall_score(EL_labels, EL_predictions),
            "EL_f1": f1_score(EL_labels, EL_predictions),
        }
    )

    print(
        {
            "accuracy_score": accuracy_score(NER_labels, NER_predictions),
            "precision": precision_score(NER_labels, NER_predictions),
            "recall": recall_score(NER_labels, NER_predictions),
            "f1": f1_score(NER_labels, NER_predictions),
        }
    )


def generate_NER_label(labels):
    return [LABEL_LIST[label] for label in labels if label != -100]


#_OUT_DICT_ENTITY_ID = -1
#_OUT_DICT_ENTITY_NAME = 'OUT_ENT'
def generate_EL_label(entity_labels, labels):
    EL_label = []
    cur_entity = 'O'
    for label, entity_label in zip(labels, entity_labels):
        if label == -100 or label == NER_LABEL_DICT['O']:
            assert entity_label == -100
            cur_entity = 'O'
            EL_label.append(cur_entity)

        elif label == NER_LABEL_DICT['B']:
            # meet out of vocabulary entity
            if entity_label == _OUT_DICT_ENTITY_ID:
                cur_entity = '-1'
                EL_label.append('B-' + cur_entity)

            # only has NER label, no EL label
            elif entity_label == -100:
                cur_entity = 'O'
                EL_label.append(cur_entity)

            # has valid EL label within vocabulary
            else:
                assert entity_label > 1
                cur_entity = str(entity_label)
                EL_label.append('B-' + cur_entity)
        else:
            assert label == NER_LABEL_DICT['I']
            # meet out of vocabulary entity
            if cur_entity == '-1':
                EL_label.append('I-' + cur_entity)

            elif cur_entity == 'O':
                EL_label.append(cur_entity)

            else:
                assert int(cur_entity) > 1
                EL_label.append('I-' + cur_entity)

    assert len(EL_label) == len(entity_labels) == len(labels)

    return EL_label


def generate_NER_prediction(predictions, labels):
    return [LABEL_LIST[pred] for pred, label in zip(predictions, labels) if label != -100]


def generate_EL_prediction(entity_predictions, tokens, valid_mention_masks, predictions, labels, ent_name_id, yago_crosswikis_wiki, args):

    new_entity_predictions = [ent_pred for ent_pred, label in zip(entity_predictions, labels) if label != -100]
    new_tokens = [token for token, label in zip(tokens, labels) if label != -100]
    new_valid_mention_masks = [valid_mention_mask for valid_mention_mask, label in zip(valid_mention_masks, labels) if label != -100]
    new_predictions = [prediction for prediction, label in zip(predictions, labels) if label != 100]
    # new_labels = [label for label in labels if label != -100]

    i, L = 0, len(new_tokens)
    cur_entity = 'O'
    EL_prediction = []

    while i < L:
        if new_valid_mention_masks[i] == 0 or new_predictions[i] == NER_LABEL_DICT['O']:
            cur_entity = 'O'
            EL_prediction.append(cur_entity)
            i += 1

        elif new_predictions[i] == NER_LABEL_DICT['I']:
            cur_entity = 'O'
            EL_prediction.append(cur_entity)
            i += 1

        else:
            assert new_valid_mention_masks[i] == 1 and new_predictions[i] == NER_LABEL_DICT['B']

            j = i + 1
            while j < L and new_valid_mention_masks[j] == 1 and new_predictions[j] == NER_LABEL_DICT['I']:
                j += 1

            k = j
            while k > i:
                mention = generate_mention(new_tokens[i:k])
                if mention in yago_crosswikis_wiki.ent_p_e_m_index and \
                        len(yago_crosswikis_wiki.ent_p_e_m_index[mention]) > 0:
                    sorted_cand = sorted(yago_crosswikis_wiki.ent_p_e_m_index[mention].items(),
                                         key=lambda x: x[1], reverse=True)
                    thids = [
                        ent_name_id.get_thid(ent_wikiid)
                        for ent_wikiid, p in sorted_cand[:args.num_cand]
                        if ent_name_id.get_thid(ent_wikiid) != ent_name_id.unk_ent_thid
                    ]

                    if len(thids) > 0:
                        max_index = torch.argmax(new_entity_predictions[i][thids]).tolist()
                        max_entity = thids[max_index]
                        assert type(max_entity) is int
                        cur_entity = str(max_entity)
                        EL_prediction.extend(['B-' + cur_entity] + ['I-' + cur_entity] * (k - i - 1))
                        cur_entity = 'O'
                        i = k
                        break
                    else:
                        k -= 1
                else:
                    k -= 1

            if k == i:
                EL_prediction.extend(['O'] * (j - i))
                i = j

    assert len(EL_prediction) == len(tokens)
    return EL_prediction


def generate_mention(tokens):
    # print('tokens', tokens)
    s = ''
    for i, token in enumerate(tokens):
        if i == 0:
            s += token
        else:
            if token == '.':
                s += token
            elif token == ',':
                s += token
            else:
                s += ' ' + token
    return s

def prepare_dataset(args):
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
    else:
        raise ValueError('Evaluation must specify test_file!')

    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    extension = args.extension_file
    print('extension', extension)
    print('data_files', data_files)
    dataset = load_dataset(extension, data_files=data_files)
    return dataset


def prepare_tokenizer(args):
    if args.source == 'hetseq':
        config = BertConfig.from_json_file(args.config_file)
    else:
        from transformers import BertConfig as TransformerBertConfig
        config = TransformerBertConfig.from_json_file(args.config_file)
    # tokenizer = BertTokenizerFast(args.vocab_file, model_max_length=512)
    # print('config', type(config), config,)
    tokenizer = BertTokenizerFast(args.vocab_file, model_max_length=config.max_position_embeddings)
    return tokenizer


def prepare_model(args):

    if args.source == 'hetseq':
        config = BertConfig.from_json_file(args.config_file)
        model = BertForELClassification(config, args)
    else:
        from transformers import BertConfig as TransformerBertConfig
        config = TransformerBertConfig.from_json_file(args.config_file)
        assert args.source == 'transformers'
        from hetseq.model import TransformersBertForELClassification
        model = TransformersBertForELClassification(config, args)

    if args.hetseq_state_dict != '':
        # load hetseq state_dictionary
        model.load_state_dict(torch.load(args.hetseq_state_dict, map_location='cpu')['model'], strict=True)

    return model


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def cli_main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # **YD** datasets argument
    parser.add_argument(
        '--train_file',
        help='local training file for BERT-ner fine-tuning',
        type=str,
        default='/scratch365/yding4/EL_resource/data/raw/AIDA-CONLL/aida_train.txt',
    )

    parser.add_argument(
        '--validation_file',
        help='local validation file for BERT-ner fine-tuning',
        type=str,
        default='/scratch365/yding4/EL_resource/data/raw/AIDA-CONLL/testa_testb_aggregate_original',
    )

    parser.add_argument(
        '--test_file',
        help='local test file for BERT-ner fine-tuning',
        type=str,
        default='/scratch365/yding4/EL_resource/data/raw/AIDA-CONLL/testa_testb_aggregate_original',
    )

    parser.add_argument(
        '--extension_file',
        help='local extension file for building dataset similar to conll2003 using "datasets" package',
        type=str,
        default='/scratch365/yding4/EL_resource/preprocess/build_EL_datasets_huggingface/BIO_EL_aida_with_valid_mention_mask.py',
    )

    # **YD** BERT model related argument
    parser.add_argument(
        '--config_file',
        help='BERT config_file',
        type=str,
        default='/scratch365/yding4/hetseq/preprocessing/uncased_L-12_H-768_A-12/bert_config.json',
    )

    parser.add_argument(
        '--vocab_file',
        help='BERT vocabulary file',
        type=str,
        default='/scratch365/yding4/hetseq/preprocessing/uncased_L-12_H-768_A-12/vocab.txt',
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

    # **YD** deep_ed arguments
    parser.add_argument(
        '--root_data_dir',
        type=str,
        default='/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/',
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
        default='/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/generated/ent_vecs/ent_vecs__ep_78.pt',
        help='entity embedding file for given dictionary',
    )

    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='testing parts of the dataset',
    )

    parser.add_argument(
        '--source',
        type=str,
        default='transformers',
        choices=['hetseq', 'transformers'],
        help='model source',
    )

    parser.add_argument(
        '--num_cand',
        type=int,
        default=100,
        help='number of top candidate entities for a given mention (surface format)',
    )

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()