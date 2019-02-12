import numpy as np
from utils import *


def filter_image(img, mask, threshold=0.6):
    """
    Apply a continuous mask to an image. The mask and the image should have the same size. For every value of the mask
    that is smaller than a threshold, its corresponding image pixel is set to zero.
    This is used to keep the part of the image that is deemed relevant from a CAM.
    :param img: An image in numpy.ndarray format, shape=(1, height, width, 3).
    :param mask: A mask as a numpy.ndarray(dtype=np.float32, shape=(height, width)).
    :param threshold: A float in [0, 1].
    :return: The filtered image with the same dimensions as the original image.
    """

    if threshold < 0 or threshold > 1:
        raise ValueError('threshold should be between 0 and 1.')

    # Identify which pixels to keep
    mask = normalize_image(mask)  # normalize mask for the threshold to make sense
    keep_px = repeat3(mask > threshold)

    # Replace the pixels to-be-discarded with 0
    filtered = np.copy(np.squeeze(img))  # copy array or else it will replace the original image
    filtered.setflags(write=True)
    filtered[~keep_px] = 0

    return filtered


def keep_percentage(img, mask, percentage=0.3):
    """
    Apply a continuous mask to an image. Keeps only a percentage of image pixels whose corresponding mask pixels have
    the highest values. The rest are set to zero.
    :param img: An image in numpy.ndarray format, shape=(1, height, width, 3).
    :param mask: A mask as a numpy.ndarray(dtype=np.dloat32, shape=(height, width)).
    :param percentage: A float representing the percentage of pixels to be kept (should be in [0, 1]).
    :return: The filtered image with the same dimensions as the original image.
    """

    if percentage < 0 or percentage > 1:
        raise ValueError('percentage should be a float between 0 and 1.')

    # Find how many pixels need to be kept to be consistent with the designated percentage
    position = int((1 - percentage) * mask.size)

    # Find the pixel intensity that corresponds with the percentage
    # i.e. the intensity value that if we keep all intensities above, we'll derive to the desired percentage
    threshold = np.sort(mask.flatten())[position]

    return filter_image(img, mask, threshold=threshold)


def merge_images(image1, image2, method='blend', alpha=0.5):
    """
    Merge two images into one. Two options are available:
    - 'blend': generates a linear combination of the two images based on a parameter 'alpha'. High values of alpha
               result in more influence from the first image and low from the second. The default value is alpha=0.5,
               which results in a simple average of the two images.
    - 'multiply': multiplies the two images. This method benefits pixels that have a high value in both images. For
                  example two pixels with values of 0.5 and 0.5 will produce a higher merged intensity than if they were
                  0.7 and 0.3 (i.e. 0.5 * 0.5 = 0.25; 0.7 * 0.3 = 0.21). In 'blend' mode with alpha=0.5 these would
                  result in the same value.
    The two images are normalized and of they don't have the same size they are resized to the default.

    :param image1: First image represented as a np.ndarray.
    :param image2: Second image represented as a np.ndarray.
    :param method: Method to merge the two images. Options: 'multiply' and 'blend'.
    :param alpha: Parameter for the blend merge method. Higher values result in more influence from image1.
                  Should be a float in [0, 1].
    :return: The np.ndarray resulting from the merge.
    """

    # If either image is in tensor format, i.e. shape=(1, height, width, channels)
    # convert it to a regular image, i.e. shape=(height, width, channels)
    if image1.shape[0] == 1:
        image1 = image1[0]
    if image2.shape[0] == 1:
        image2 = image2[0]

    # If one of the images is RGB and the other grayscale,
    # repeat the grayscale image 3 times so that the dimensions match
    if len(image1.shape) == 2 and len(image2.shape) == 3:
        image1 = repeat3(image1)
    if len(image1.shape) == 3 and len(image2.shape) == 2:
        image2 = repeat3(image2)

    # Check if any of the two images are blank; if yes, return the other. If both are blank raise an error
    c1 = not np.any(image1 > 0)
    c2 = not np.any(image2 > 0)
    if c1 and c2:
        raise ValueError('Both images are completely blank.')
    elif c1:
        return image2
    elif c2:
        return image1

    # Normalize the two images to [0, 255] range to be conpatible with resizing
    image1 = normalize_image(image1, dtype='i')
    image2 = normalize_image(image2, dtype='i')

    # If the two images aren't the same size, resize them both to the default
    if image1.shape != image2.shape:

        # Normalize the two images to [0, 255] range to be conpatible with resizing
        image1 = normalize_image(image1, dtype='i')
        image2 = normalize_image(image2, dtype='i')

        image1 = resize(image1)
        image2 = resize(image2)

    # Normalize the two images
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    # Method 1: multiplication
    if method == 'multiply':
        return image1 * image2

    # Method 2: blending (linear combination of the two images)
    elif method == 'blend':

        # Check the validity of parameter alpha
        if alpha < 0 or alpha > 1:
            raise ValueError('Parameter alpha should be between 0 and 1.')

        return alpha * image1 + (1 - alpha) * image2

    else:
        raise ValueError("Invalid method. Must be either 'multiply' or 'blend'.")

