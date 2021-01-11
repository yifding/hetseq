# 1. load wikipedia2vec entity vector and vocab.
# 2. load luke entity dictionary.
# 3. load BIO_EL_aida dataset.
# 4. for each entitty in aida, judge whether it was in the top 0.5M of luke dictionary

# **YD** biggest problem is that, the entities in the evaluation dataset do not show up in the training dataset
import argparse
import collections

import datasets
import wikipedia2vec

_NER_VOCAB = {"O": 0, "B": 1, "I": 2}
_UNK_ENTITY_NAME = '{UNK}'

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
    dataset = datasets.load_dataset(extension, data_files=data_files)

    return dataset


def load_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # **YD** datasets argument

    parser.add_argument(
        '--extension_file',
        help='local extension file for building dataset similar to conll2003 using "datasets" package',
        type=str,
        default='/scratch365/yding4/EL_resource/preprocess/build_EL_datasets_huggingface/BIO_EL_aida.py',
    )

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
        '--wikipedia2vec_file',
        help='wikipedia2vec_file',
        type=str,
        default='/scratch365/yding4/hetseq/preprocessing/wikipedia2vec/enwiki_20180420_100d.txt',
    )

    parser.add_argument(
        '--luke_entity_dict_file',
        help='luke_entity_dict_file',
        type=str,
        default='/scratch365/yding4/luke_dir/luke_large_500k/entity_vocab.tsv',
    )

    args = parser.parse_args()
    return args


def load_luke_entity_dict(args):
    dic = collections.OrderedDict()
    with open(args.luke_entity_dict_file, 'r') as reader:
        for line in reader:
            line = line.rstrip()
            parts = line.split('\t')
            assert len(parts) == 2
            dic[parts[0]] = parts[1]

    return dic


def collect_entity_freq_from_aida(dataset):
    ans = dict()
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            dataset_split = dataset[split]
            for index in range(len(dataset_split)):
                instance = dataset_split[index]
                for entity_name, ner_tag in zip(instance['entity_names'], instance['ner_tags']):
                    if ner_tag == _NER_VOCAB['B'] and entity_name != _UNK_ENTITY_NAME:
                        entity_name = entity_name.replace('_', ' ')
                        if entity_name not in ans:
                            ans[entity_name] = 0
                        ans[entity_name] += 1
    return ans


if __name__ == '__main__':
    """
    args = load_args()
    
    # 1. load wikipedia2vec entity vector and vocab.
    print('====> loading wikipedia2vec entity vector')
    wikipedia2vec_dict = wikipedia2vec.Wikipedia2Vec.load_text(args.wikipedia2vec_file)
    print('--------> loading wikipedia2vec entity vector finished')

    # 2. load luke entity dictionary.
    luke_entity_dict = load_luke_entity_dict(args)
    
    # 3.load BIO_EL_aida dataset.
    dataset = prepare_dataset(args)

    # 4. for each entitty in aida, judge whether it was in the top 0.5M of luke dictionary
    # 4-1. collect entity_names from BIO_EL_aida dataset, count the entity frequency in a dictionary.
    entity_freq_from_aida = collect_entity_freq_from_aida(dataset)
    """

    """
    # 4-1.a: change the space '_' with space
    # 4-2. collect entity_names from BIO_EL_aida dataset, count the entity frequency in a dictionary.
    find_entity_freq, not_find_entity_freq = 0, 0
    find_entity_set, not_find_entity_set = set(), set()
    for key, value in entity_freq_from_aida.items():
        if wikipedia2vec_dict.get_entity(key):
            find_entity_set.add(key)
            find_entity_freq += value
        else:
            not_find_entity_set.add(key)
            not_find_entity_freq += value

    print(find_entity_freq, not_find_entity_freq)

    find_entity_freq, not_find_entity_freq = 0, 0
    find_entity_set, not_find_entity_set = set(), set()
    for key, value in entity_freq_from_aida.items():
        if wikipedia2vec_dict.get_entity(key):
            if key in luke_entity_dict:
                find_entity_set.add(key)
                find_entity_freq += value
            else:
                not_find_entity_set.add(key)
                not_find_entity_freq += value
        else:
            not_find_entity_set.add(key)
            not_find_entity_freq += value

    print(find_entity_freq, not_find_entity_freq)
    """

    find_entity_freq, not_find_entity_freq = 0, 0
    find_entity_set, not_find_entity_set = set(), set()
    for key in luke_entity_dict:
        if wikipedia2vec_dict.get_entity(key):
            find_entity_set.add(key)
            find_entity_freq += value
        else:
            not_find_entity_set.add(key)
            not_find_entity_freq += value

    print(find_entity_freq, not_find_entity_freq)