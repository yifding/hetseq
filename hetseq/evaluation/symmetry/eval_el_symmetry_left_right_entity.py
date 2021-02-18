import os
import math
import argparse
from collections import defaultdict

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
from hetseq.data_collator import DataCollatorForSymmetryWithCandEntities

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
    "left_entity_masks",
    "right_entity_masks",

    "span_masks",
    "span_cand_entities",

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
    data_collator = DataCollatorForSymmetryWithCandEntities(tokenizer)

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

    # **YD** add tokens for results print
    NER_tokens = []

    # **YD** build EL, EL_labels can obtain from entity_th_ids
    # EL_predictions obtain from NER_labels and model EL outputs.
    EL_predictions = []
    EL_labels = []

    stop_words = StopWords()
    ent_name_id = EntNameID(args)
    yago_crosswikis_wiki = YagoCrosswikisWiki(args)

    for index, input in tqdm(enumerate(data_loader)):
        # **YD** all the elements of input are list of list
        entity_th_ids = input["entity_th_ids"][0]
        span_masks = input["span_masks"][0]
        span_cand_entities = input["span_cand_entities"][0]

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
        left_predictions, right_predictions, entity_predictions = model(**input)

        # **YD** select the only batch and remove first and last special token
        left_predictions = torch.softmax(left_predictions, dim=2)[0][:, 1].tolist()
        right_predictions = torch.softmax(right_predictions, dim=2)[0][:, 1].tolist()
        entity_predictions = entity_predictions[0]

        # left_predictions = left_predictions[0].cpu().detach().numpy()
        # right_predictions = right_predictions[0].cpu().detach().numpy()
        # print('left_predictions', left_predictions)
        # print('right_predictions', right_predictions)
        # print('entity_th_ids', entity_th_ids)
        # shrink the size follow the labels
        left_predictions = [left_prediction for left_prediction, entity_th_id in zip(
            left_predictions, entity_th_ids) if entity_th_id != -100]

        right_predictions = [right_prediction for right_prediction, entity_th_id in zip(
            right_predictions, entity_th_ids) if entity_th_id != -100]

        ori_position = generate_ori_position(entity_th_ids)
        entity_th_ids = [entity_th_id for entity_th_id in entity_th_ids if entity_th_id != -100]

        if not(len(tokens) == len(ner_tags) == len(left_predictions) == len(right_predictions)):
            print("tokens", len(tokens), tokens)
            print("ner_tags", len(ner_tags), ner_tags)
            print("left_predictions", len(left_predictions), left_predictions)
            print("right_predictions", len(right_predictions), right_predictions)

            # fix the bug of too long tokens
            tmp_len = len(left_predictions)
            tokens = tokens[: tmp_len]
            ner_tags = ner_tags[: tmp_len]

        assert len(tokens) == len(ner_tags) == len(left_predictions) == len(right_predictions)

        NER_label = generate_NER_label(ner_tags, entity_th_ids)
        NER_prediction = generate_NER_prediction(
            left_predictions, right_predictions, tokens, stop_words, yago_crosswikis_wiki, ent_name_id,
            left_threshold=args.left_threshold,
            right_threshold=args.right_threshold,
            total_threshold=args.total_threshold,
        )

        EL_label = generate_EL_label(ner_tags, entity_th_ids)
        EL_prediction = generate_EL_prediction(
            entity_predictions,
            span_cand_entities,
            NER_prediction,
            ori_position,
        )

        if not (len(NER_label) == len(NER_prediction)):
            print("NER_label", len(NER_label), NER_label)
            print("NER_prediction", len(NER_prediction), NER_prediction)
        assert len(NER_label) == len(NER_prediction)


        NER_labels.append(NER_label)
        NER_predictions.append(NER_prediction)

        EL_labels.append(EL_label)
        EL_predictions.append(EL_prediction)

        # **YD** add tokens for results print
        NER_tokens.append(tokens)

        if index == 0:
            print('NER_labels', NER_labels)
            print('NER_predictions', NER_predictions)
            print('EL_labels', EL_labels)
            print('EL_predictions', EL_predictions)
            print('tokens', tokens)

    B_only_NER_preidictions = B_only(NER_predictions)
    B_only_NER_labels = B_only(NER_labels)


    B_only_performance = {
        "accuracy_score": accuracy_score(B_only_NER_labels, B_only_NER_preidictions),
        "precision": precision_score(B_only_NER_labels, B_only_NER_preidictions),
        "recall": recall_score(B_only_NER_labels, B_only_NER_preidictions),
        "f1": f1_score(B_only_NER_labels, B_only_NER_preidictions),
    }


    OUTPUT_performance = {
        "accuracy_score": accuracy_score(NER_labels, NER_predictions),
        "precision": precision_score(NER_labels, NER_predictions),
        "recall": recall_score(NER_labels, NER_predictions),
        "f1": f1_score(NER_labels, NER_predictions),
    }

    EL_performance = {
        "accuracy_score": accuracy_score(EL_labels, EL_predictions),
        "precision": precision_score(EL_labels, EL_predictions),
        "recall": recall_score(EL_labels, EL_predictions),
        "f1": f1_score(EL_labels, EL_predictions),
    }

    print('B_only NER results')
    print(B_only_performance)
    print('NER results')
    print(OUTPUT_performance)
    print('EL results')
    print(EL_performance)

    NER_label_out = stats(NER_labels)
    NER_prediction_out = stats(NER_predictions)
    EL_label_out = stats(EL_labels)
    EL_prediction_out = stats(EL_predictions)

    print('NER_labels', NER_label_out)
    print('NER_prediction', NER_prediction_out)
    print('EL_labels', EL_label_out)
    print('EL_prediction', EL_prediction_out)

    with open(args.performance_file, 'w') as writer:
        writer.write('B_only NER results \n')
        writer.write(str(B_only_performance) + '\n')
        writer.write('NER results \n')
        writer.write(str(OUTPUT_performance) + '\n')
        writer.write('NER_labels \n')
        writer.write(str(NER_label_out) + '\n')
        writer.write('NER_prediction \n')
        writer.write(str(NER_prediction_out) + '\n')
        writer.write('EL_labels \n')
        writer.write(str(EL_label_out) + '\n')
        writer.write('EL_prediction \n')
        writer.write(str(EL_prediction_out) + '\n')

    with open(args.labeling_file, 'w') as writer:
        writer.write(
            'TOKEN' + '\t' +
            'NER_LABEL' + '\t' + 'NER_Prediction' + '\t' +
            'EL_LABEL' + '\t' + 'EL_Prediction' + '\n'
        )

        for NER_token, NER_label, NER_prediction, EL_label, EL_prediction in \
                zip(NER_tokens, NER_labels, NER_predictions, EL_labels, EL_predictions):

            for token, NER_l, NER_p, EL_l, EL_p in \
                    zip(NER_token, NER_label, NER_prediction, EL_label, EL_prediction):
                writer.write(token + '\t' + NER_l + '\t' + NER_p + '\t' + EL_l + '\t' + EL_p + '\n')
            writer.write('\n')


def B_only(labels):
    B_only_labels = []
    for label in labels:
        B_only_label = []
        for l in label:
            if l.startswith('B'):
                B_only_label.append('B')
            else:
                B_only_label.append('O')
        B_only_labels.append(B_only_label)
    return B_only_labels


def stats(labels):
    len_dict = defaultdict(int)
    total = 0
    for label in labels:
        cur_len = 0
        for i, l in enumerate(label):
            # 1. deal with finished entity
            if cur_len > 0:
                if l.startswith('O') or l.startswith('B'):
                    len_dict[cur_len] += 1
                    cur_len = 0
                else:
                    assert l.startswith('I')
                    cur_len += 1

            if l.startswith('B'):
                total += 1
                cur_len = 1

            if i == len(label) - 1 and cur_len > 0:
                len_dict[cur_len] += 1

    output = {
        'total': total,
        'len_dict': len_dict,
    }

    assert total == sum(len_dict.values())

    return output


def generate_ori_position(entity_th_ids):
    ori_position = []
    for index, entity_th_id in enumerate(entity_th_ids):
        if entity_th_id != -100:
            ori_position.append(index)

    return ori_position


def generate_NER_prediction(
    left_predictions,
    right_predictions,
    tokens,
    stop_words,
    yago_crosswikis_wiki,
    ent_name_id,
    left_threshold=0.5,
    right_threshold=0.5,
    total_threshold=1.0,
):
    '''
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
    '''

    span = dict()
    L = len(tokens)

    # 1. build span candidates
    for left_index in range(L):
        for right_index in range(left_index, min(L, left_index + 4)):
            if (
                    left_predictions[left_index] > left_threshold
                    and right_predictions[right_index] > right_threshold
                    and (left_predictions[left_index] + right_predictions[right_index]) > total_threshold
            ):
                if left_index == right_index and (
                        stop_words.is_stop_word_or_number(tokens[left_index])
                        or tokens[left_index].isspace()
                        or tokens[left_index].strip().isnumeric()
                        or tokens[left_index].strip() in '[@_!#$%^&*()<>?/\|}{~:,.\'\-"]'
                                ):
                    continue

                cur_mention = generate_mention(tokens[left_index: right_index + 1])
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
                        span[(left_index, right_index)] = left_predictions[left_index] + right_predictions[right_index]

    # 2. select span candidates similar to NMS from high probability to low probability
    NER_prediction = ['O'] * L
    mask = [0] * L
    sorted_span = sorted(span.items(), key=lambda x:x[1], reverse=True)
    for (left_index, right_index), score in sorted_span:
        span_mask = mask[left_index: right_index + 1]
        assert len(span_mask) == right_index + 1 - left_index
        if all(e == 0 for e in span_mask):
            mask[left_index: right_index + 1] = [1] * (right_index + 1 - left_index)
            NER_prediction[left_index: right_index + 1] = ['B'] + ['I'] * (right_index - left_index)

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


def len_NER(NER_prediction, index):
    assert NER_prediction[index] == 'B'
    ans = 1
    for i in range(index + 1, len(NER_prediction)):
        if NER_prediction[i] == 'I':
            ans += 1
        else:
            break
    return ans


def generate_EL_prediction(
            entity_predictions,
            span_cand_entities,
            NER_prediction,
            ori_position,
        ):

    assert len(NER_prediction) == len(ori_position)
    L = len(NER_prediction)
    EL_prediction = ['O'] * L

    for index, NER in enumerate(NER_prediction):
        if NER == 'B':
            span_len = len_NER(NER_prediction, index)
            left_index = ori_position[index]
            right_index = span_len - 1
            span_cand_entity = span_cand_entities[left_index][right_index]
            entity_prediction = entity_predictions[left_index][right_index]

            thids = [thid for thid in span_cand_entity if thid > 1]

            if len(thids) > 0:
                select_thid = -1
                max_score = -math.inf
                for thid in thids:
                    if entity_prediction[thid] > max_score:
                        max_score = entity_prediction[thid]
                        select_thid = thid

                cur_entity = str(select_thid)
                EL_prediction[index: index + span_len] = ['B-' + cur_entity] + ['I-' + cur_entity] * (span_len - 1)

    return EL_prediction

def generate_EL_label(ner_tags, entity_th_ids):
    EL_label = []
    cur_entity = 'O'

    for ner_tag, entity_th_id in zip(ner_tags, entity_th_ids):
        if ner_tag == -100 or ner_tag == NER_LABEL_DICT['O']:
            assert entity_label <= 0
            cur_entity = 'O'
            EL_label.append(cur_entity)

        elif ner_tag == NER_LABEL_DICT['B']:
            # meet out of vocabulary entity
            if entity_th_id == _OUT_DICT_ENTITY_ID:
                cur_entity = '-1'
                EL_label.append('B-' + cur_entity)

            # only has NER label, no EL label
            elif entity_th_id == _UNK_ENTITY_ID:
                cur_entity = 'O'
                EL_label.append(cur_entity)

            # has valid EL label within vocabulary
            else:
                assert entity_th_id > 1
                cur_entity = str(entity_th_id)
                EL_label.append('B-' + cur_entity)
        else:
            assert ner_tag == NER_LABEL_DICT['I']
            # meet out of vocabulary entity
            if cur_entity == '-1':
                EL_label.append('I-' + cur_entity)

            elif cur_entity == 'O':
                EL_label.append(cur_entity)

            else:
                assert int(cur_entity) > 1
                EL_label.append('I-' + cur_entity)

    assert len(EL_label) == len(entity_th_ids) == len(ner_tags)

    return EL_label


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

        model_class = getattr(args, 'model_class', 'TransformersBertForELSymmetry')
        if model_class == 'TransformersBertForELSymmetry':
            from hetseq.model import TransformersBertForELSymmetry
            model = TransformersBertForELSymmetry(config, args)
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

    # arguments for inference.
    parser.add_argument(
        '--left_threshold',
        type=float,
        default=0.5,
        help='left threshold',
    )

    parser.add_argument(
        '--right_threshold',
        type=float,
        default=0.5,
        help='right threshold',
    )

    parser.add_argument(
        '--total_threshold',
        type=float,
        default=1.0,
        help='total threshold',
    )

    # **YD** output files of performance and labelling
    parser.add_argument(
        '--labeling_file',
        type=str,
        default='./labelling.log',
        help='output labelling file',
    )

    parser.add_argument(
        '--performance_file',
        type=str,
        default='./performance.log',
        help='output performance file',
    )

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()