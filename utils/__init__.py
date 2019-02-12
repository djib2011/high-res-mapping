from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.opts import opt
from scipy import stats
import numpy as np
from PIL import Image

# Input dimensions
image_dims = (opt.im_size, opt.im_size)
input_shape = image_dims + (opt.channels,)


def resize(arr, target_size=image_dims, resample_method='bilinear'):
    """
    Resize given image to the target size. Supported interpolation methods: 'bilinear' and 'nearest'
    :param arr: Image in numpy.ndarray format
    :param target_size: Tuple representing the target shape: (height, weight)
    :param resample_method: Method used for resampling. Valid options: 'bilinear' and 'nearest'.
    :return: Resized image as a numpy.ndarray
    """
    if resample_method == 'bilinear':
        resample = Image.BILINEAR
    elif resample_method == 'nearest':
        resample = Image.NEAREST
    else:
        raise NotImplementedError("Only 'bilinear' and 'nearest' resample methods available")
    return np.array(Image.fromarray(arr).resize(target_size, resample=resample))


def normalize_image(img, dtype='f'):
    """
    Normalize an image to [0, 1] or to [0, 255]. If normalized to [0, 1] the data type will be np.float32. If normalized
    to [0, 255] the data type will be np.uint8.
    :param img: An image in a numpy.ndarray format (dtype either np.float32 or np.unit8 depending on normalization).
    :return: The normalized image array.
    """
    if dtype == 'f':
        return (img - img.min()) / (img.max() - img.min())
    elif dtype in ('u', 'i'):
        return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        raise TypeError("dtype can be either 'u', 'i' for [0, 255] images or 'f' for [0, 1].")


def repeat3(img):
    """
    Repeat an array 3 times along its last axis
    :param img: A numpy.ndarray
    :return: A numpy.ndarray with a shape of: img.shape + (3,)
    """
    return np.repeat(img[..., np.newaxis], 3, axis=-1)


def percentile(arr, percent):
    """
    Accepts an arbitrary percentage (let's say x) and returns the value that lies on the x-th percentile (i.e. below
    that value is x percent of the values of the array.
    :param arr: An array (numpy.ndarray).
    :param percent: A percentile (int in (0, 100)).
    :return: The value that lies on the given percentile (float).
    """
    return arr.min() + stats.iqr(arr, rng=(0, percent))

