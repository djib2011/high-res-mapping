#!/usr/bin/python3
# -*- coding: utf-8 -*-

from imgaug import augmenters as iaa
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import custom_models
from utils.opts import opt
from utils.train import CustomImageGenerator, transfer_weights
import os
import numpy as np

# For manually overwriting opt
num_classes = opt.num_classes
batch_size = opt.batch_size
epochs = opt.epochs
image_dims = (opt.im_size, opt.im_size)
input_shape = image_dims + (opt.channels,)
weight_dir = opt.half_weight_dir
data_dir = opt.data_dir
results_name = opt.results_name
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

# Check if user gave mandatory opts
if not results_name:
    raise ValueError('You must specify a name for the results directory. Add "--results_name name"')

# Where to store results
results_dir = os.path.join(os.getcwd(), 'results', results_name)
log_dir = os.path.join(os.getcwd(), 'logs', results_name)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Read train/validation data
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Apply image transforms
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

# Train/Validation generators
generator = CustomImageGenerator(preprocessing_function=seq.augment_image)
generator_val = CustomImageGenerator()

train = generator.flow_from_directory(train_dir, target_size=image_dims, batch_size=batch_size)
val = generator_val.flow_from_directory(test_dir, target_size=image_dims, batch_size=batch_size)

# Define keras callbacks
lr_reducer = ReduceLROnPlateau(monitor='loss', factor=np.sqrt(0.1), patience=10, cooldown=0, min_lr=1e-5)
model_chp = ModelCheckpoint(os.path.join(results_dir, 'best_weights.h5'), monitor='loss',
                            save_best_only=True, save_weights_only=True)
tb = TensorBoard(log_dir=log_dir, batch_size=batch_size)

# Instantiate full and half models
model = custom_models.AuxiliaryDenseNetFCN(input_shape, num_classes, weights=None)
pretrained = custom_models.HalfDenseNetFCN(input_shape, num_classes, weights=weight_dir)

# Transfer weights from half to full and freeze them
transfer_weights(pretrained, model)
del pretrained

for layer in model.layers[:121] + [model.layers[-5], model.layers[-3], model.layers[-1]]:
    layer.trainable = False

# Re-compile the model
model.compile(optimizer='adadelta',
              loss={'main_output': 'mean_squared_error', 'aux_output': 'categorical_crossentropy'},
              metrics={'main_output': ['accuracy'], 'aux_output': ['accuracy']})

# Train the model
out = model.fit_generator(train, steps_per_epoch=train.n//batch_size, epochs=epochs,
                          callbacks=[tb, model_chp, lr_reducer], validation_data=val,
                          validation_steps=val.n//batch_size)

# In case of completion
model.save_weights(os.path.join(results_dir, 'last_weights.h5'))

