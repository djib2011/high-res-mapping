from scipy.ndimage.filters import convolve
import numpy as np
from skimage.filters import sobel
from utils import normalize_image


def blur(image, filter_size=5):
    """
    Function that uses a convolution filter to blur an image.
    :param image: An image (numpy.ndarray)
    :param filter_size: The size of each dimension of the filter
                        For example, for filter_size=5 a 5x5 convolution filter will be applied.
    :return: The blurred image (numpy.ndarray).
    """
    return convolve(normalize_image(image), np.zeros((filter_size, filter_size)) + 1 / filter_size ** 2)


# For namespace purposes (actually the line isn't necessary)
sobel = sobel

