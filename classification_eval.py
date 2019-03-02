from cams import HalfModel, FullModel
from utils import opt, io_utils
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# For manually overwriting opt:
model_type = opt.type
data_dir = opt.data_dir
weights = None

test_dir = os.path.join(data_dir, 'test')

if os.path.isdir(test_dir):
    classes = len(os.listdir(test_dir))

# Load a model
if model_type == 'half':
    if not weights:
        weights = opt.half_weights
    model = HalfModel(weights=weights, num_classes=io_utils.num_classes)
elif model_type == 'full':
    if not weights:
        weights = opt.weights
    model = FullModel(weights=weights, num_classes=io_utils.num_classes)
else:
    raise TypeError('option "type" needs to be either "full" or "half.')

# Use the model to make predictions on the test set
labels, preds = [], []
for i in tqdm(range(io_utils.num_images)):
    x, y = io_utils.read_image(i)
    y_hat = model.predict(x)
    labels.append(y)
    preds.append(y_hat)

# Compute the results
prec, rec, f1, supp = precision_recall_fscore_support(labels, preds)

mask = np.where(supp != 0, 1, np.nan)

micro_preceision = np.nanmean(prec * supp * mask) / supp.max()
micro_recall = np.nanmean(rec * supp * mask) / supp.max()
micro_f1 = np.nanmean(f1 * supp * mask) / supp.max()

macro_preceision = np.nanmean(prec * mask)
macro_recall = np.nanmean(rec * mask)
macro_f1 = np.nanmean(f1 * mask)

acc = accuracy_score(labels, preds)

# Print the scores
print('{:<16} {:.2f}%'.format('top-1 accuracy:', acc * 100))
print('{:<16} {:.2f}%'.format('micro-precision:', micro_preceision * 100))
print('{:<16} {:.2f}%'.format('micro-recall:', micro_recall * 100))
print('{:<16} {:.2f}%'.format('micro-f1:', micro_f1 * 100))
print('{:<16} {:.2f}%'.format('macro-precision:', macro_preceision * 100))
print('{:<16} {:.2f}%'.format('macro-recall:', macro_recall * 100))
print('{:<16} {:.2f}%'.format('macro-f1:', macro_f1 * 100))

