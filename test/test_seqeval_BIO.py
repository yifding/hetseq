from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
y_true = [['O', 'O', 'O', 'B--1', 'I--1', 'I--1', 'O'], ['B-234', 'I-234', 'O']]
y_pred = [['O', 'O', 'B--1', 'I--1', 'I--1', 'I--1', 'O'], ['B-234', 'I-234', 'O']]

print(f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))