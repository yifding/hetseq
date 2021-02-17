from .tasks import Task, LanguageModelingTask, MNISTTask
from .bert_for_el_classification_task import BertForELClassificationTask
from .bert_for_token_classification_task import BertForTokenClassificationTask
from .bert_for_el_symmetry_task import BertForELSymmetryTask
from .bert_for_el_symmetry_with_ent_cand_task import BertForELSymmetryWithEntCandTask

__all__ = [
    'Task',
    'LanguageModelingTask',
    'BertForTokenClassificationTask',
    'MNISTTask',
    'BertForELClassificationTask',
    'BertForELSymmetryTask',
    'BertForELSymmetryWithEntCandTask',
]