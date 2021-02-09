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
from hetseq.data_collator import DataCollatorForSymmetry

from transformers import (
    BertTokenizerFast,
)

import numpy
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
    "input_ids",
    "token_type_ids",
    "attention_mask",

    "entity_th_ids",
    "left_mention_masks",
    "right_mention_masks",
    "left_entity_masks",
    "right_entity_masks",
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
    dataset = prepare_dataset(args)
    tokenizer = prepare_tokenizer(args)
    data_collator = DataCollatorForSymmetry(tokenizer)

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
        # batched=True, **YD** test non-batched results
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
    )

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
        num_workers=0,
    )

    model = prepare_model(args)
    model.cuda()
    model.eval()

    # **YD** starts with capital represents outputs variables
    NER_predictions = []
    NER_labels = []

    stop_words = StopWords()
    ent_name_id = EntNameID(args)
    yago_crosswikis_wiki = YagoCrosswikisWiki(args)

    for index, input in tqdm(enumerate(data_loader)):
        # **YD** all the elements of input are list of list
        entity_th_ids = input["entity_th_ids"][0]

        # **YD** inference signal
        input["entity_th_ids"] = None

        # **YD** recover tokens string and valid mention mask.
        dataset_element = test_dataset[index]
        tokens = dataset_element["tokens"]
        ner_tags = dataset_element["ner_tags"]

        # assert the removed special tokens length = length of direct load ones
        # assert len(labels) == len(entity_labels) == len(tokens) == len(valid_mention_masks)

        # print(input.keys())
        input = utils.move_to_cuda(input)
        left_predictions, right_predictions = model(**input)

        # **YD** select the only batch and remove first and last special token
        left_predictions = torch.argmax(left_predictions, axis=2)[0].tolist()
        right_predictions = torch.argmax(right_predictions, axis=2)[0].tolist()

        # left_predictions = left_predictions[0].cpu().detach().numpy()
        # right_predictions = right_predictions[0].cpu().detach().numpy()
        # print('left_predictions', len(left_predictions))
        # print('right_predictions', len(right_predictions))
        # print('entity_th_ids', len(entity_th_ids))
        # shrink the size follow the labels
        left_predictions = [left_prediction for left_prediction, entity_th_id in zip(
            left_predictions, entity_th_ids) if entity_th_id != -100]

        right_predictions = [right_prediction for right_prediction, entity_th_id in zip(
            right_predictions, entity_th_ids) if entity_th_id != -100]

        entity_th_ids = [entity_th_id for entity_th_id in entity_th_ids if entity_th_id != -100]

        if not(len(tokens) == len(ner_tags) == len(left_predictions) == len(right_predictions)):
            print("tokens", len(tokens), tokens)
            print("ner_tags", len(ner_tags), ner_tags)
            print("left_predictions", len(left_predictions), left_predictions)
            print("right_predictions", len(right_predictions), right_predictions)

            tmp_len = len(left_predictions)
            tokens = tokens[:tmp_len]
            ner_tags = ner_tags[:tmp_len]

        assert len(tokens) == len(ner_tags) == len(left_predictions) == len(right_predictions)

        NER_label = generate_NER_label(ner_tags, entity_th_ids)
        NER_prediction = generate_NER_prediction(
            left_predictions, right_predictions, tokens, stop_words, yago_crosswikis_wiki, ent_name_id
        )

        if not (len(NER_label) == len(NER_prediction)):
            print("NER_label", len(NER_label), NER_label)
            print("NER_prediction", len(NER_prediction), NER_prediction)
        assert len(NER_label) == len(NER_prediction)

        NER_labels.append(NER_label)
        NER_predictions.append(NER_prediction)

        if index == 0:
            print('NER_labels', NER_labels)
            print('NER_predictions', NER_predictions)
            print('tokens', tokens)

    print(
        {
            "accuracy_score": accuracy_score(NER_labels, NER_predictions),
            "precision": precision_score(NER_labels, NER_predictions),
            "recall": recall_score(NER_labels, NER_predictions),
            "f1": f1_score(NER_labels, NER_predictions),
        }
    )


def generate_NER_prediction(
    left_predictions,
    right_predictions,
    tokens,
    stop_words,
    yago_crosswikis_wiki,
    ent_name_id,
):

    NER_prediction = []
    i, L = 0, len(left_predictions)

    while i < L:
        if left_predictions[i] == 0:
            NER_prediction.append('O')
            i += 1
        else:
            assert left_predictions[i] == 1
            flag = False
            # **YD** prefere choosing the long one
            for j in range(i, min(i + 4, L)):
            #for j in range(min(i + 4, L) - 1, i - 1, -1):
                if j == i and (stop_words.is_stop_word_or_number(tokens[i])
                               or tokens[i].isspace()
                               or tokens[i].strip().isnumeric()
                               or tokens[i].strip() in '[@_!#$%^&*()<>?/\|}{~:,.\'\-"]'
                                ):
                    continue

                if right_predictions[j] == 1:
                    cur_mention = generate_mention(tokens[i: j + 1])
                    cur_mention = yago_crosswikis_wiki.preprocess_mention(cur_mention)
                    if cur_mention in yago_crosswikis_wiki.ent_p_e_m_index:
                        sorted_cand = sorted(yago_crosswikis_wiki.ent_p_e_m_index[cur_mention].items(),
                                             key=lambda x: x[1], reverse=True)
                        thids = [
                            ent_name_id.get_thid(ent_wikiid)
                            for ent_wikiid, p in sorted_cand[: 10]
                            if ent_name_id.get_thid(ent_wikiid) != ent_name_id.unk_ent_thid
                        ]
                        if len(thids) > 0:
                            NER_prediction.extend(['B'] + ['I'] * (j-i))
                            i = j + 1
                            flag = True
                            break
            if not flag:
                NER_prediction.append('O')
                i += 1

    assert len(left_predictions) == len(right_predictions) == len(NER_prediction)
    return NER_prediction


def generate_NER_label(ner_tags, entity_th_ids):
    NER_label = []
    for ner_tag, entity_th_id in zip(ner_tags, entity_th_ids):
        if ner_tag != -100:
            if entity_th_id != _EMPTY_ENTITY_ID and entity_th_id != _UNK_ENTITY_ID:
                NER_label.append(LABEL_LIST[ner_tag])
            else:
                NER_label.append('O')
    return NER_label


def generate_mention(tokens):
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
        raise ValueError('hetseq format does not support now!')
    else:
        from transformers import BertConfig as TransformerBertConfig
        config = TransformerBertConfig.from_json_file(args.config_file)
        assert args.source == 'transformers'

        model_class = getattr(args, 'model_class', 'TransformersBertForNERSymmetry')
        if model_class == 'TransformersBertForNERSymmetry':
            from hetseq.model import TransformersBertForNERSymmetry
            model = TransformersBertForNERSymmetry(config, args)
        else:
            raise ValueError('Unknown model_class')

    if args.hetseq_state_dict != '':
        print('load hetseq state_dictionary')
        model.load_state_dict(torch.load(args.hetseq_state_dict, map_location='cpu')['model'], strict=True)

    return model


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
        default='test',
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
        '--model_class',
        type=str,
        default='TransformersBertForNERSymmetry',
        choices=['TransformersBertForNERSymmetry'],
        help='model_class source',
    )

    parser.add_argument(
        '--num_cand',
        type=int,
        default=10,
        help='number of top candidate entities for a given mention (surface format)',
    )

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()