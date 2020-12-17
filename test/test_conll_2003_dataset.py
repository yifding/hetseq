import os
import argparse

from bert_modeling import (
    BertConfig,
    BertForPreTraining,
    BertForTokenClassification,
)


def main(args):
    # 1. prepare dataset from customized dataset
    # 2. prepare bert-model loading pre-trained checkpoint
    # 3. prepare tokenizer from customized dictionary

    pass


def cli_main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        '',
        help='',
        type=str,
    )

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()