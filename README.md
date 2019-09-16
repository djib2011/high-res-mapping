### Repository of the paper:

## High Resolution Class Activation Mapping for Discriminative Feature Localization
### by Thanos Tagaris, Maria Sdraka and Andreas Stafylopatis

Abstract:
> Insufficient reasoning for their predictions has for long been a major drawback of neural networks 
and has proved to be a major obstacle for their adoption by several fields of application. This paper 
presents a framework for discriminative localization, which helps shed some light into the decision-making 
of Convolutional Neural Networks (CNN). Our framework generates robust, refined and high-quality Class 
Activation Maps, without impacting the CNN’s performance.

Presented at [ICIP 2019](http://www.2019.ieeeicip.org/):  

- [full paper](https://ieeexplore.ieee.org/document/8803474)
- [poster](https://sigport.org/documents/high-resolution-class-activation-mapping)

![](https://github.com/djib2011/high-res-mapping/blob/master/sample_images/collage.png)

## Quick start:

First add the your default parameters to [`utils.opts`](https://github.com/djib2011/high-res-mapping/blob/master/utils/opts.py). These include the location of the directory storing your training/test images (`data_dir`), the size of the images (`im_size`), the location of the weights (`weight_dir`), etc.

```python
from utils.io_utils import *
from utils import plotting
import postprocessing
from cams import FullModel

# The read_image can load images along with their labels
x, y = read_image()  # read a random image 

# The FullModel can make predictions and generate low and high-res CAMs 
fm = FullModel()
high, low, preds = fm.generate_cams(x)  # generate the cams and preds for x

# Apply the postprocessing pipeline to generate a refined CAM
steps = postprocessing.pipeline(x, low, high)
steps['merged']  # the refined CAM

# Draw the whole pipeline
plotting.pipeline(x, low, high)
```

## Requirements:

- python 3
- numpy
- scipy
- skimage (required for postprocessing filters)
- tensorflow (if you want to train a model, tensorflow-gpu is recommended)
- keras (all models are keras instances)
- imgaug (used for data augmentation)
- tqdm (only required for `classification_eval`)
- SimpleITK (required for `ConnectedThreshold` segmentation)
- matplotlib (only required for plotting the postprocessing pipeline)
- seaborn (same as matplotlib)

## Detailed description:

### a. *HalfModel*

The *HalfModel* is a simple CNN for classification, with a single Fully Connected (FC) layer and a Global Average Pooling (GAP) layer after its final convolutional layer. This architecture was proposed by [Zhou et al.](https://arxiv.org/pdf/1512.04150.pdf).

![](https://github.com/djib2011/high-res-mapping/blob/master/figures/HalfModel.png)

The dimensions shown in the figure represent those of the proposed DenseNet-based *HalfModel*. *M* is the number of classes.

To generate the (resized) low-res CAMs, one must follow the methodology illustrated in the figure below:

![](https://github.com/djib2011/high-res-mapping/blob/master/figures/HalfModel_localization.png)

The image is passed through the network and the activations of the final convolution layer are extracted and resized to the original dimensions. Meanwhile the weight matrix from the final FC layer is sliced in order to keep the weights for the desired class. For example, if we want the CAM for class *c*, we'll slice the respective column (see figure below).

![](https://github.com/djib2011/high-res-mapping/blob/master/figures/weight_slicing.png)

The (resized) activations from the final convolution layer along with the slice of the weight matrix are multiplied to produce the low-res CAM. It is called a low-res CAM because before the resize, the CAM has dimensions of *14x14*. 

### b. *FullModel*

The proposed *FullModel* is depicted in the figure below:

![](https://github.com/djib2011/high-res-mapping/blob/master/figures/FullModel.png)

The *FullModel* consists of two parts, the *localization network* (in black and blue) and the *expansion network* (in orange). The first is responsible for making predictions and generating low-res CAMs, while the second produces the high-res CAMs. The procedure for producing the high-res CAMs is similar to the one for their lower resolution counterparts. The only difference is that instead of the output of the final convolution of the *localization network* (which has dimensions of *14x14*), we'll use the output of the convolution of the *expansion network*, which is upscaled to *224x224*.

## Guide:

#### Tip:

If you want to run a single experiment (use one dataset), go to [`utils.opts`](https://github.com/djib2011/high-res-mapping/blob/master/utils/opts.py) and change the default values to meet your specifications. More importantly change the default value of `data_dir`, so that you don't have to specify it every time.

### 1. Data
#### 1.1. Data acquisition
#### 1.1.a. *Animals* dataset
Due to the large size of the dataset, we didn't upload it to the repo, however you can create it by following these instructions:
- Download the [ILSVRC 2012 dataset](http://image-net.org/challenges/LSVRC/2012/). 
- Keep only the subset of classes concerning animals. These should be 398 classes which you can find in the 
[animal_synsets.txt](https://github.com/djib2011/high-res-mapping/blob/master/animal_synsets.txt) file.
- The images should amount to a total of 510530 images.
#### 1.1.b. custom dataset
You can use your own dataset, just keep in mind that you need have *enough* in order tor the model to be trained.  
How much is *enough*? I'd say somewhere in the tens of thousands, but I can't be sure.

#### 1.2. Pre-processing
During this step the images should be:
- Resized to the same resolution (we used 224 × 224).
- Normalized to [0, 1].
- Split into a training and a test set. We used 396416 images for training and 114114 for test). 

#### 1.3. Structuring the data
In order to work as intended you should have two main directories for the training and test images respectively. 
Each should have the images organized in classes. You should have a directory for each class that contains the images of that class.
For example if you have 5 classes, each of the train and test directories should have 5 sub-directories (one for each class),
containing the train/test images of that class. 

The data needs to be structured using the following scheme:
```
data_dir
   |  
   |____train  
   |      |  
   |      |_____n01440764  
   |      |       |__n01440764_5861.JPEG    
   |      |       |__n01440764_9973.JPEG  
   |      |       ...  
   |      |_____n02326432  
   |      |       |__n02326432_38335.JPEG  
   |      |       |__n02326432_9314.JPEG  
   |      |       ...  
   |      ...  
   |      ...  
   |____test  
   |      |  
   |      |_____n01440764  
   |      |       |__n01440764_5103.JPEG     
   |      |       |__n01440764_9567.JPEG  
   |      |       ...  
   |      |_____n02326432  
   |      |       |__n02326432_17608.JPEG  
   |      |       |__n02326432_9173.JPEG  
   |      |       ...  
   |      ...  
   |      ...  
```

This scheme is selected so that the data can be read by keras' [ImageDataGenerator](https://keras.io/preprocessing/image/).

### 2. Classification model pre-training

#### 2.1. Training 

In this step we'll train the classification model (which produces the low-res CAMs). This is referred to as the *half model*.
The script you'll need to run is [`pretrain_densenet_fcn.py`](https://github.com/djib2011/high-res-mapping/blob/master/pretrain_densenet_fcn.py).

The default parameters are:
```python
batch_size=16    # batch size
epochs=50        # number of epochs
im_size=224      # desired dimension of the input image 
channels=3       # number of channels (3 for rgb, 1 for grayscale)
num_classes=398  # number of classes in the dataset
weight_dir=None  # directory where the weights are stored
gpu='0'          # which gpu to use (only relevant for multi-gpu systems)
```

Two parameters that will need to be specified are:

```python
data_dir         # location where we can find the data_dir (see 1.3. for what the data_dir should look like)
results_name     # name of the directory that will store the results. can include subdirs (which will be created if they don't exist)
half_weight_dir  # if we want to continue the training of the model, specify the location of the weights
```

For example:
```
python pretrain_densenet_fcn.py --data_dir /path/to/data_dir --results_name half/experiment1/run3 --batch_size 32 --epochs 100
```
This would start training the DenseNet with a batch size of 32, for 100 epochs. The training images should be under `/path/to/data_dir/train/`, the weights will be stored under `results/half/experiment1/run3/` and the logs under `logs/half/experiment1/run3`.

#### 2.2. Evaluation 

If you want to evaluate the performance of the *half* model, you'll need to run [`https://github.com/djib2011/high-res-mapping/blob/master/classification_eval.py`](https://github.com/djib2011/high-res-mapping/blob/master/classification_eval.py).

Two parameters need to be specified:
```python
data_dir         # location where we can find the data_dir (see 1.3. for what the data_dir should look like)
half_weight_dir  # the path to the weights we want to use
type             # the type of the model: half/full. In this case it needs to be set to half
```

Example:
```
python classification_eval.py --data_dir path/to/data_dir --type half --half_weight_dir /path/to/half/weights.h5
```

The script will print the accuracy of the model as well as the macro and micro averages of its precision, recall and f1 scores. If you want other custom metrics (e.g. top-5 accuracy), you'll have to chance this script.

Note: the evaluation will be performed on the test set images.  

#### 2.3. Interpreting results

As mentioned previously, the *half* model is simply a classification model that can produce low-res CAMs. The scores will show how potent it is for classification. If the results are not satisfactory, this means that either that the problem is very difficult (e.g. very few images, images are hard to classify) or that the model hasn't been trained properly (e.g. few epochs, wrong weights).

If this is the case, there is a good chance that neither the CAMs will be able to pick up on any meaningful features.

#### 2.4. Using the *half* model

How can I use the trained *half* model? 

First, you have to import its wrapper class (which makes it easier to use) and then instantiate it. You'll need to pass the location of the weights file during instantiation.

```python
from cams import HalfModel

hm = HalfModel(weights='path/to/half_weights_dir/best_weights.h5')
```

What can I do with the `HalfModel`?

You can do two main things: make a prediction and draw a low-res CAM. Given an input image `x` with a shape of `(1, height, width, channels)`:

```python
# Precictions. Will return the class' index:
pred = hm.predict(x)                     # make a prediction for image x
preds = hm.predictN(x, n=3)              # make the top-3 predictions for image x

# Low-res CAMs. Shape will be (height/14, width/14):
prediction_cam = hm.cam(x)               # generate the low-res CAM for the predicted class 
custom_cam = hm.cam(x, custom_class=13)  # generate the low-res CAM for class 13
```

### 3. FCN model training

#### 3.1. Training 

Now, it's time to train the FCN model which is capable of classification, low and high-res mapping. This is referred to as the *full model*.

The main script you'll need to run is [`train_densenet_fcn.py`](https://github.com/djib2011/high-res-mapping/blob/master/train_densenet_fcn.py). Most options are the same, with the exception of:

```python
half_weight_dir  # specify the location of the weights of the pre-trained *half* model. 
```

which is now mandatory.

For example:
```
python train_densenet_fcn.py --data_dir /path/to/data_dir --results_name full/experiment1/run3 --half_weight_dir results/half/experiment1/run3/best_weights.h5 --batch_size 32 --epochs 100
```

This will train the *full* model with images from `/path/to/data_dir/train/` and weights initialized from `results/half/experiment1/run3/best_weights.h5`. After training the *full* model with the above options, the weights will be stored under `results/full/experiment1/run3`, while the logs will be stored in `logs/full/experiment1/run3`.

#### 3.2. Evaluation 

If you want to evaluate the performance of the *full* model for the classification task, you'll need to run [`https://github.com/djib2011/high-res-mapping/blob/master/classification_eval.py`](https://github.com/djib2011/high-res-mapping/blob/master/classification_eval.py).

Again, two parameters need to be specifeid:

```python
data_dir         # location where we can find the data_dir (see 1.3. for what the data_dir should look like)
weight_dir       # the path to the weights we want to use
type             # the type of the model: half/full. In this case it needs to be set to full
```

Example:
```
python classification_eval.py --data_dir path/to/data_dir --type full --weight_dir /path/to/full/weights.h5
```

The script will print the accuracy of the model as well as the macro and micro averages of its precision, recall and f1 scores. If you want other custom metrics (e.g. top-5 accuracy), you'll have to chance this script.

Note: the evaluation will be performed on the test set images.  

#### 3.3. Interpreting results

The scores here should be identical to the ones achieved during step 2.2. If not, this means that the *full* model's weights were not properly initialized from the pre-trained *half* model's weights. 

If this is the case, steps 3.1 and 3.2. should be repeated. Be careful on entering the proper weights after the `half_weight_dir` parameter. You can debug this through `utils.train.check_weights()` (see 5.2.).

#### 3.4. Using the *full* model

How can I use the trained *full* model? 

Again, you have to import the wrapper class and instantiate it. The location of the weights file needs to be passed as a parameter.

```python
from cams import FullModel

fm = FullModel(weights='path/to/weights_dir/best_weights.h5')
```

What can I do with the `FullModel`?

You can do three main things: make predictions and draw both low and high-res CAMs. Given an input image `x` with a shape of `(1, height, width, channels)`:

```python
# Predictions. Will return the class' index:
pred = fm.predict(x)

# Generate low-res CAMs. The shape of the low-res CAMs will be (height/14, width/14):
low_cam = fm.low_res_cam(x)                     # low-res CAM for predicted class 
low_cam = fm.low_res_cam(x, custom_class=13)    # low-res CAM for class 13

# Generate high-res CAMs. The shape of the high-res CAMs will be (height, width):
high_cam = fm.high_res_cam(x)                   # high-res CAM for predicted class
high_cam = fm.high_res_cam(x, custom_class=13)  # high-res CAM for class 13
```

### 4. Post-processing

#### 4.1. Option 1: Pipeline outputs

In order to apply the proposed post-processing pipeline that combines both high and low-res CAMs in order to produce the *refined* CAMs, you have to use the [`postprocessing.py`](https://github.com/djib2011/high-res-mapping/blob/master/postprocessing.py) script. This will return in detail the outputs of each of the steps in the postprocessing pipeline.

**Usage**:

Assuming an input image `x` with a shape of `(1, height, width, channels)`, a low-res CAM `low_res_cam`and a high-res CAM `high_res_cam`.

```python
from postprocessing import pipeline

steps = pipeline(x, low_res_cam, high_res_cam)

steps['blurred']       # the blurred high-res CAM
steps['low']           # the original low-res CAM
steps['low_resized']   # the resized low-res CAM (through bilinear interpolation)
steps['edges']         # the result of the sobel filter on the blurred high-res map
steps['roi']           # the Region Of Interest extracted from the low-res map
steps['bounds']        # the lower and upper bounds for the region-growing segmentation
steps['focal_points']  # the focal_points extracted from the low-res map
steps['segmented']     # the output of the region-growing segmentation
steps['full_segment']  # the filled segmentation
steps['merged']        # the merger of the segmentation with the low-res map, i.e. the refined CAM
steps['result']        # the part of the original image that has been cropped according to the regined CAM
```

If we just care about the *refined* CAM we can just use `steps['merged']`.

**Postprocessing pipeline parameters**:

```python
roi_percentile=85          # the percentile above which the ROI will be estimated. roi_percentile=85 means that the 15% highest
                           # intensity pixels of the low-res map will constitute the ROI (int in (0, 100)).
focal_scope='global'       # the scope in which the focal points will be identified. 'global' looks for global maxima, 
                           # while 'local' looks for local maxima. Accepted values: ['global', 'local']
maxima_areas='small'       # can either be 'large' or 'small', depending on whether or not we want larger or smaller areas.
                           # Only relevant for 'local' scopes. Accepted values: ['global', 'local']
merge_type='blend'         # selection on whether to multiply or blend the high with the low-res CAMs after processing. 
                           # Accepted values: ['blend', 'merge']
merge_alpha=0.5            # parameter for the blend merge method. Higher values result in more influence from the high-res map.
                           # Should be a float in [0, 1].
filter_type='percentage'   # selects how to crop the original image according to the refined CAM. Two options are available:
                           # - 'percentage', which keeps a percentage of the highest-instensity values of the refined CAM
                           # - 'threshold', which keeps the intensities above a certain threshold
filter_percentage=15       # a float representing the percentage of pixels to be kept (should be in [0, 1]).
                           # Only relevant when filter_type='percentage'
filter_threshold=0.6       # a float in [0, 1] over which the intensities of the refined CAM will be kept. 
                           # Only relevant when filter_type='threshold'
```

#### 4.2. Option 2: Visualization

You can visualize the whole pipeline procedure through [`utils.plotting.py`](https://github.com/djib2011/high-res-mapping/blob/master/utils/plotting.py).

**Usage**:

Assuming an input image `x` with a shape of `(1, height, width, channels)`, a low-res CAM `low_res_cam`and a high-res CAM `high_res_cam`.

```python
from utils.plotting import pipeline

pipeline(x, low_res_cam, high_res_cam)
```

This will create a figure visualizing how both maps are combined, step-by-step, to create the *refined* CAM.

Parameters are the same as in 4.1. 

### 5. Useful utils

#### 5.1. Input

Useful utils for data input. You need to first specify a `data_dir`:

```python
from io_utils import *

# Info: 
test_images  # a list of the paths for the images in the test_dir
num_images   # number of images in the test_dir
num_classes  # number of classes

# Reading images:
x, y = read_image()      # loads a random image from the test set, along with its label's index
x, y = read_image(13)     # loads image with index 13 from the test set, along with its label's index
x, y = read_image(13, return_class_name=True)  # same as before,
                                               # returns the label's name (only works for animals dataset)

# AnimalsMapper class (it is already instantiated):
mapper.name(5)            # will return the name for the class with an index of 5 
mapper.index('stingray')  # will return the index for the class with a name of 'stingray'
```

If you haven't set a default `data_dir` from `utils.opts`.

```python
# First define useful paths:
from utils.opts import opt
from pathlib import Path

opt.data_dir = '/path/to/data_dir'
test_dir = str(Path(opt.data_dir) / 'test')

# Then instantiate the AnimalsMapper class:
from utils.io_utils import AnimalsMapper

synset_path = '/path/to/synset_words.txt'
mapper = AnimalsMapper(synset_path, test_dir=test_dir)

# Finally import read_image:
from utils.io_utils import read_image
```

#### 5.2. Debug tools

#### 5.2.a. Find the memory usage of a model

Given a keras `model` and the `batch_size` we want to use: 

```python
from utils.debug import get_model_memory_usage

print(get_model_memory_usage(model, batch_size))
```

Will return the required memory to train the `model` with the given `batch_size` in GB.

#### 5.2.b. Compere the weights of two models

Given two keras models `model1` and `model2`:

```python
from utils.debug import check_weights

check_weights(model1, model2)
```

Will compare the weights between `model1` and `model2` to see up to which point they are equal. Useful for checking if the *half* model's weights were properly transferred to the *full* model.
