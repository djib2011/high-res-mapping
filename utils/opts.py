from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import __main__

import argparse
import os
import keras.backend as K

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--im_size', type=int, default=224, help='image dimensions')
parser.add_argument('--channels', type=int, default=3, help='number of channels (3 for rgb, 1 for grayscale)')
parser.add_argument('--num_classes', type=int, default=398, help='number of classes in the dataset')
parser.add_argument('--weight_dir', type=str, default=None, help='directory where the weights are stored')
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
parser.add_argument('--backend_format', type=str, default='channels_last', help='keras image data format')
parser.add_argument('--type', type=str, default='full', help='type of the model to evaluate for classification')
parser.add_argument('--target_name', type=str, default=None, help='name of the directory storing the results')
parser.add_argument('--data_dir', type=str, default='/home/thanos/Machine Learning/Datasets/animals',
                    help='directory where training data is located')
parser.add_argument('--weight_dir', type=str, default='results/densenet_fcn/animals/best_weights.h5',
                    help='path to a valid weight file')
parser.add_argument('--half_weight_dir', type=str, default='results/half_densenet_fcn/animals/best_weights.h5',
                    help='path to a valid weight file')

if hasattr(__main__, '__file__'):
    opt = parser.parse_args()
else:
    opt = parser.parse_args(args=[])

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


K.set_image_data_format(opt.backend_format)

if opt.weight_dir and not os.path.isdir(opt.weight_dir):
    os.mkdir(opt.weight_dir)

