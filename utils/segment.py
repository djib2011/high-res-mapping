from utils import percentile
from skimage import segmentation
import SimpleITK as sitk
import numpy as np


def felzenszwalb(image, params=None):
    """
    Wrapper for scikit-image's felzenszwalb segmentation. Mainly added for namespace uniformity and for ease with the
    default parameters
    :param image: An image (numpy.ndarray).
    :param params: A dictionary containing parameters to be used for the segmentation (dict).
    :return: The segmented image (numpy.ndarray).
    """

    # Type check
    if not isinstance(params, dict):
        raise TypeError('Argument "params" should be a dictionary.')

    # Define some default parameters. If these are not specified in the 'params' dictionary, then use the default.
    default_params = {'scale': 100, 'sigma': 0.5, 'min_size': 100}
    if not params:
        params = default_params
    else:
        for param, value in default_params:
            if param not in params:
                params[param] = value

    # Segment the image and return the result
    return (segmentation.felzenszwalb(image, scale=params['scale'],
                                      sigma=params['sigma'],
                                      min_size=params['min_size']) > 0).astype(np.uint8)


def region_growing(image, seeds, lower=None, upper=None):
    """
    Wrapper for SimpleITK's ConnectedThreshold region-growing segmentation. This was created for ease of use as
    ConnectedThreshold requires an proprietary SimpleITK image format. This function handles converting the image (in
    the conventional numpy.ndarray format) to sitk's proprietary one and back, as well as performing the actual
    segmentation.
    :param image: An image (numpy.ndarray).
    :param seeds: The seeds from which the region growing will start (list of tuples).
    :param lower: The lower boundary, under which the 'growing' process stops.
    :param upper: The upper boundary, over which the 'growing' process stops.
    :return: The segmented image (numpy.ndarray)
    """

    # Doesn't work with lower=0. Instead try lower=0.00001.
    if not lower:

        lower = percentile(image, 25)

    if not upper:

        upper = percentile(image, 75)

    if all([isinstance(x, int) for x in seeds]):
        seeds = [seeds]

    image = sitk.GetImageFromArray(image)

    segm = sitk.ConnectedThreshold(image1=image, seedList=seeds, lower=lower, upper=upper, replaceValue=1)

    return sitk.GetArrayFromImage(segm)

