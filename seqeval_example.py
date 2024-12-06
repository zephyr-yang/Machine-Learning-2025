#
# copied from github.com/chakki-work/seqeval
#

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC','O',], ['B-PER', 'I-PER', 'O']]

print('f1_score =', f1_score(y_true, y_pred))
print("classification result is \n", classification_report(y_true, y_pred))
print("accuracy_score =", accuracy_score(y_true, y_pred))

# strict mode
print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))

# different example
y_true = [['B-NP', 'I-NP', 'O']]
y_pred = [['I-NP', 'I-NP', 'O']]

print(classification_report(y_true, y_pred))
print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
