from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase


@dataclass
class DataCollatorForSymmetry:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`,
            `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

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

    INPUT_IDS_PAD = 0
    LABELS_PAD = -100
    ENTITY_LABELS_PAD = -100
    TOKEN_TYPE_ID_PAD = 0
    ATTENTION_MASK_PAD = 0

    def __call__(self, features):
        label_name = "entity_th_ids"

        len_list = [len(feature[label_name]) for feature in features]
        max_len = max(len_list)

        # print(len(features), features)
        def process_label(label):
            if type(label) is torch.Tensor:
                return label.tolist()
            else:
                assert type(label) is list
                return label

        # **YD** manually padding to solve the NER padding issues.
        batch = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],

            "entity_th_ids": [],
            "left_mention_masks": [],
            "right_mention_masks": [],
            "left_entity_masks": [],
            "right_entity_masks": [],
        }

        for feature in features:
            if self.tokenizer.padding_side == "right":
                batch["input_ids"].append(
                    process_label(feature["input_ids"]) + [self.INPUT_IDS_PAD] * (max_len - len(feature["input_ids"]))
                )

                batch["token_type_ids"].append(
                    process_label(feature["token_type_ids"]) + [self.TOKEN_TYPE_ID_PAD] * (
                            max_len - len(feature["token_type_ids"])
                    )
                )

                batch["attention_mask"].append(
                    process_label(feature["attention_mask"]) + [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["attention_mask"]))
                )

                batch["entity_th_ids"].append(
                    process_label(feature["entity_th_ids"]) + [self.ENTITY_LABELS_PAD] * (
                            max_len - len(feature["entity_th_ids"]))
                )

                batch["left_mention_masks"].append(
                    process_label(feature["left_mention_masks"]) + [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["left_mention_masks"]))
                )

                batch["right_mention_masks"].append(
                    process_label(feature["right_mention_masks"]) + [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["right_mention_masks"]))
                )

                batch["left_entity_masks"].append(
                    process_label(feature["left_entity_masks"]) + [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["left_entity_masks"]))
                )

                batch["right_entity_masks"].append(
                    process_label(feature["right_entity_masks"]) + [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["right_entity_masks"]))
                )

            else:
                batch["input_ids"].append(
                    [self.INPUT_IDS_PAD] * (max_len - len(feature["input_ids"])) + process_label(feature["input_ids"])
                )

                batch["token_type_ids"].append(
                     [self.TOKEN_TYPE_ID_PAD] * (
                        max_len - len(feature["token_type_ids"])
                     ) + process_label(feature["token_type_ids"])
                )

                batch["attention_mask"].append(
                    [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["attention_mask"])) + process_label(feature["attention_mask"])
                )

                batch["entity_th_ids"].append(
                    [self.ENTITY_LABELS_PAD] * (
                            max_len - len(feature["entity_th_ids"])) + process_label(feature["entity_th_ids"])
                )

                batch["left_mention_masks"].append(
                    [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["left_mention_masks"])) + process_label(feature["left_mention_masks"])
                )

                batch["right_mention_masks"].append(
                    [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["right_mention_masks"])) +
                    process_label(feature["right_mention_masks"])
                )

                batch["left_entity_masks"].append(
                    [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["left_entity_masks"])) + process_label(feature["left_entity_masks"])
                )

                batch["right_entity_masks"].append(
                    [self.ATTENTION_MASK_PAD] * (
                            max_len - len(feature["right_entity_masks"])) + process_label(feature["right_entity_masks"])
                )

        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        # change the input features from a list of dict to dict of list.
        # features = {key: [example[key] for example in features] for key in features[0].keys()}

        if labels is None:
            return batch

        batch = {k: torch.from_numpy(np.asarray(v, dtype=np.int64)) for k, v in batch.items()}
        return batch
