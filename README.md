### Repository of the paper:

## High Resolution Class Activation Mapping for Discriminative Feature Localization
### by Thanos Tagaris, Maria Sdraka and Andreas Stafylopatis

Abstract:
> Insufficient reasoning for their predictions has for long been a major drawback of neural networks 
and has proved to be a major obstacle for their adoption by several fields of application. This paper 
presents a framework for discriminative localization, which helps shed some light into the decision-making 
of Convolutional Neural Networks (CNN). Our framework generates robust, refined and high-quality Class 
Activation Maps, without impacting the CNN’s performance.

## Quick start:

### 1. Data
#### 1.1. Data aquisition
#### 1.1.a. *Animals* dataset
Due to the large size of the dataset, we didn't upload it to the repo, however you can create it by following these instructions:
- Download the [ILSVRC 2012 dataset](http://image-net.org/challenges/LSVRC/2012/). 
- Keep only the subset of classes concerning animals. These should be $398$ classes which you can find in the 
[animal_synsets.txt](https://github.com/djib2011/high-res-mapping/blob/master/animal_synsets.txt) file.
- The images should amount to a total of $510530$ images.
#### 1.1.b. custom dataset
You can use your own dataset, just keep in mind that you need have *enough* in order tor the model to be trained.  
How much is *enough*? I'd say somewhere in the tens of thousands, but I can't be sure.

#### 1.2. Pre-processing
During this step the images should be:
- Resized to the same resolution (we used $224 \times 224$).
- Normalized to $[0, 1]$.
- Split into a training and a test set. We used $396416$ images for training and $114114$ for test). 

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

In this step we'll train the classification model (which produces the low-res CAMs). The script you'll need to run is 
[`pretrain_densenet_fcn.py`](https://github.com/djib2011/high-res-mapping/blob/master/pretrain_densenet_fcn.py).

The default parameters are:
```python
batch_size=16    # batch size
epochs=50        # number of epochs
im_size=224      # desired dimension of the input image 
channels=3       # number of channels (3 for rgb, 1 for grayscale)
num_classes=398  # number of classes in the dataset
weight_dir=None  # directory where the weights are stored
gpu='0'          # which gpu to use (unly relevant for multi-gpu systems)
```

Two parameters that will need to be specified are:

```python
data_dir         # location where we can find the data_dir (see 1.3. for what the data_dir should look like)
half_weight_dir  # if we want to continue the training of the model, specify the location of the weights
```

## Requirements:


## Detailed description of experiments:
