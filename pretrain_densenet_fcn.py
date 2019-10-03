#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import custom_models
from imgaug import augmenters as iaa
from utils.opts import opt

# For manually overwriting opt
num_classes = opt.num_classes
batch_size = opt.batch_size
epochs = opt.epochs
img_dims = (opt.im_size, opt.im_size)
input_shape = img_dims + (opt.channels,)
data_dir = opt.data_dir
half_weight_dir = opt.half_weight_dir
results_name = opt.target_name
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

# Check if user gave mandatory opts
if not data_dir:
    raise ValueError('You must specify a data directory. Add "--data_dir /path/to/data_dir"')
if not results_name:
    raise ValueError('You must specify a name for the results directory. Add "--results_name name"')

# Define useful paths
train_path = os.path.join(data_dir, 'train')
test_path = os.path.join(data_dir, 'test')

results_dir = os.path.join(os.getcwd(), 'results', results_name)
log_dir = os.path.join(os.getcwd(), 'logs', results_name)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define image augmentation scheme
seq = iaa.Sequential([
      iaa.Sometimes(0.5, iaa.Affine(
                  scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # scale images to 80-120% of their size
                  translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                  rotate=(-180, 180),  # rotate by -180 to +180 degrees
                  shear=(-5, 5),
                  order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                  cval=0,
                  mode='constant'
                  )),
      iaa.Fliplr(0.5)
      ])

# Initialize Image Generators for batching
tg = ImageDataGenerator(preprocessing_function=seq.augment_image)
vg = ImageDataGenerator()

train_gen = tg.flow_from_directory(train_path, target_size=img_dims, batch_size=batch_size)
test_gen = vg.flow_from_directory(test_path, target_size=img_dims, batch_size=batch_size)

# Model
model = custom_models.HalfDenseNetFCN(input_shape, num_classes)

# Load previously trained model
if half_weight_dir:
    previous_weights = half_weight_dir
    model.load_weights(previous_weights)

# Define keras callbacks
lr_reducer = ReduceLROnPlateau(monitor='loss', factor=np.sqrt(0.1), patience=10, cooldown=0, min_lr=1e-5, verbose=1)
tb = TensorBoard(log_dir=log_dir, batch_size=batch_size)
model_chp = ModelCheckpoint(os.path.join(results_dir, 'best_weights.h5'), monitor='val_acc',
                            save_best_only=True, save_weights_only=True)

# Train the model
model.fit_generator(train_gen, epochs=epochs, callbacks=[tb, model_chp, lr_reducer], validation_data=test_gen)

# In case of completion
model.save_weights(os.path.join(results_dir, 'last_weights.h5'))

