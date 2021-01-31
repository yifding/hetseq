import os
import argparse

from datasets import load_dataset, ClassLabel

from hetseq.bert_modeling import (
    BertConfig,
    BertForPreTraining,
    # BertForTokenClassification,
)
from hetseq.model import BertForELClassification

from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID

from hetseq import utils
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

_EL_COLUMNS = ['input_ids', 'labels', 'token_type_ids', 'attention_mask', 'entity_labels']
NER_LABEL_DICT = {'B': 0, 'I': 1, 'O': 2}
_UNK_ENTITY_ID = 1
_UNK_ENTITY_NAME = 'UNK_ENT'
_EMPTY_ENTITY_ID = 0
_EMPTY_ENTITY_NAME = 'EMPTY_ENT'
_OUT_DICT_ENTITY_ID = -1
_IGNORE_CLASSIFICATION_LABEL = -100

# 1. load dataset, either from CONLL2003 or other entity linking datasets.
# 2. load model from a trained one, load model architecture and state_dict from a trained one.
# 3. predict loss, generate predicted label
# 4. compare predicted label with ground truth label to obtain evaluation results.

def EL_GT_label():
    # special tokens or (second and after subwords), do not consider this token
    pre_el_labels = []
    gt_el_labels = []

    cur_entity = 'O'
    pre_cur_entity = 'O'
    if GT_NER_label == -100:
        pass

    else:
        if GT_NER_label == 'B':
            # a mention with no corresponding entity in the GT labeling.
            if GT_EL_label == 'UNKNOW':
                gt_el_labels.append('O')
                cur_entity = 'O'
            else:
                # can be in dictionary entity, with positive number
                # or out-dictionary entity, with -1
                cur_entity = 'B' + str(GT_EL_label)
                gt_el_labels.append(cur_entity)

        elif GT_NER_label == 'I':
            gt_el_labels.append(cur_entity)

        elif GT_NER_label == 'O':
            gt_el_labels.append('O')
            cur_entity = 'O'
        else:
            raise ValueError('Unseen NER label' + GT_NER_label)

        if PRE_NER_label == 'B':
            if PRE_NER_label == 'UNKNOW':
                gt_el_labels.append('O')
                pre_cur_entity = 'O'
            else:
                pre_cur_entity = 'B' + str(PRE_EL_label)
                pre_el_labels.append(pre_cur_entity)

        elif PRE_NER_label == 'I':
            pre_el_labels.append(pre_cur_entity)

        elif PRE_NER_label == 'O':
            pre_el_labels.append('O')
            pre_cur_entity = 'O'
        else:
            raise ValueError('Unseen NER label' + PRE_NER_label)


def main(args):
    dataset = prepare_dataset(args)
    tokenizer = prepare_tokenizer(args)
    data_collator = YD_DataCollatorForELClassification(tokenizer)
    # data_collator = DataCollatorForTokenClassification(tokenizer)

    if 'test' in dataset:
        column_names = dataset["test"].column_names
        features = dataset["test"].features
    else:
        raise ValueError('Evaluation must specify test_file!')

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
        if 'test' in label_column_name:
            label_list = get_label_list(datasets["test"][label_column_name])
        else:
            raise ValueError('Evaluation must specify test_file!')

        label_to_id = {l: i for i, l in enumerate(label_list)}
    args.num_labels = len(label_list)

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
        # batched=True, **YD** test non-batched results
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
    )

    # test_dataset = tokenized_datasets['test']
    test_dataset = tokenized_datasets[args.split]

    # **YD** core code to keep only usefule parameters for model
    test_dataset.set_format(type=test_dataset.format["type"], columns=_EL_COLUMNS)

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

    # load entity embedding and set up shape parameters
    args.EntityEmbedding = torch.load(args.ent_vecs_filename, map_location='cpu')
    args.num_entity_labels = args.EntityEmbedding.shape[0]
    args.dim_entity_emb = args.EntityEmbedding.shape[1]

    model = prepare_model(args)
    model.cuda()
    model.eval()

    ner_predictions = []
    ner_labels = []

    el_predictions = []
    el_labels = []

    EL_predictions = []
    EL_labels = []

    for index, input in tqdm(enumerate(data_loader)):
        labels, input['labels'] = input['labels'].tolist(), None
        entity_labels = input['entity_labels'].tolist()

        # print(input.keys())
        input = utils.move_to_cuda(input)
        predictions, entity_predictions = model(**input)

        predictions = torch.argmax(predictions, axis=2).tolist()
        entity_predictions = torch.argmax(entity_predictions, axis=2).tolist()

        #print('entity_predictions', entity_predictions.shape, 'entity_labels', entity_labels.shape)

        # **YD** close the gap NOW!!!
        assert len(labels) == len(entity_labels) == len(predictions) == len(entity_predictions)
        for label, entity_label, prediction, entity_prediction in zip(labels, entity_labels, predictions, entity_predictions):
            assert len(label) == len(entity_label) == len(prediction) == len(entity_prediction)
            cur_entity = 'O'
            pre_cur_entity = 'O'
            gt_el_labels = []
            pre_el_labels = []

            for l, e_l, p, e_p in zip(label, entity_label, prediction, entity_prediction):
                if l == -100:
                    pass
                else:
                    if label_list[l] == 'B':
                        # a mention with no corresponding entity in the GT labeling.
                        if e_l <= 0:
                            gt_el_labels.append('O')
                            cur_entity = 'O'
                        else:
                            # can be in dictionary entity, with positive number
                            # or out-dictionary entity, with -1
                            assert e_l > 0
                            cur_entity = str(e_l)
                            gt_el_labels.append('B-' + cur_entity)

                    elif label_list[l] == 'I':
                        if cur_entity == 'O':
                            gt_el_labels.append('O')
                        else:
                            gt_el_labels.append('I-' + cur_entity)

                    elif label_list[l] == 'O':
                        gt_el_labels.append('O')
                        cur_entity = 'O'
                    else:
                        raise ValueError('Unseen NER label' + label_list[l])

                    if label_list[p] == 'B':
                        if e_p <= 0:
                            pre_el_labels.append('O')
                            pre_cur_entity = 'O'
                        else:
                            assert e_p > 0
                            pre_cur_entity = str(e_p)
                            pre_el_labels.append('B-' + pre_cur_entity)

                    elif label_list[p] == 'I':
                        if pre_cur_entity == 'O':
                            pre_el_labels.append('O')
                        else:
                            pre_el_labels.append('I-' + pre_cur_entity)

                    elif label_list[p] == 'O':
                        pre_el_labels.append('O')
                        pre_cur_entity = 'O'
                    else:
                        raise ValueError('Unseen NER label' + label_list[p])

            EL_predictions.append(pre_el_labels)
            EL_labels.append(gt_el_labels)

        ner_predictions.extend([
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ])

        ner_labels.extend([
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ])

        '''
        el_predictions.extend([
            [p for (p, l) in zip(entity_prediction, entity_label) if l != -100]
            for entity_prediction, entity_label in zip(entity_predictions, entity_labels)
        ])

        el_labels.extend([
            [l for (p, l) in zip(entity_prediction, entity_label) if l != -100]
            for entity_prediction, entity_label in zip(entity_predictions, entity_labels)
        ])
        '''

    EL_predictions = [EL_prediction for EL_prediction in EL_predictions if EL_prediction != []]
    EL_labels = [EL_label for EL_label in EL_labels if EL_label != []]
    ner_predictions = [ner_prediction for ner_prediction in ner_predictions if ner_prediction != []]
    ner_labels = [ner_label for ner_label in ner_labels if ner_label != []]

    print('ner_predictions', ner_predictions[0], ner_predictions[-1])
    print('ner_labels', ner_labels[0], ner_labels[-1])


    '''
    print('el_predictions', el_predictions[0], el_predictions[-1])
    print('el_labels', el_labels[0], el_labels[-1])
    '''

    total = 0
    correct = 0
    for el_prediction, el_label in zip(el_predictions, el_labels):
        # print('el_predictions', el_prediction, 'el_labels', el_label)
        for p, l in zip(el_prediction, el_label):
            if p == l:
                correct += 1
            total += 1
    print('total', total, 'correct', correct)


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
            "accuracy_score": accuracy_score(ner_labels, ner_predictions),
            "precision": precision_score(ner_labels, ner_predictions),
            "recall": recall_score(ner_labels, ner_predictions),
            "f1": f1_score(ner_labels, ner_predictions),
        }
    )

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

        model_class = getattr(args, 'model_class', 'TransformersBertForELClassification')
        if model_class == 'TransformersBertForELClassification':
            from hetseq.model import TransformersBertForELClassification
            model = TransformersBertForELClassification(config, args)
        elif model_class == 'TransformersBertForELClassificationCrossEntropy':
            from hetseq.model import TransformersBertForELClassificationCrossEntropy
            model = TransformersBertForELClassificationCrossEntropy(config,args)
        else:
            raise ValueError('Unknown model_class')

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
        default='hetseq',
        choices=['hetseq', 'transformers'],
        help='model source',
    )

    parser.add_argument(
        '--model_class',
        type=str,
        default='TransformersBertForELClassification',
        choices=['TransformersBertForELClassification', 'TransformersBertForELClassificationCrossEntropy'],
        help='model_class source',
    )

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()