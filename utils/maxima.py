import numpy as np
from utils import resize, opt
from skimage import morphology


def find_focal_points(image, scope='local', maxima_areas='large', local_maxima_threshold=None, num_points=None):
    """
    Finds the 'focal_points' of a model, given a low resolution CAM. Has two modes: a 'local' scope and a 'global' one.

    If a 'local' scope is selected, the function looks for local maxima in the CAM. Due to the high sensitivity of the
    algorithm finding the local maxima, usually a large number of maxima is identified (which is, in most cases,
    undesirable. An interest_threshold can be selected that filters out possibly unwanted maxima (i.e. maxima whose
    intensity is lower than the threshold). Due to the resizing of the CAM, these local maxima produce large areas in
    the new image. If this is not desired, the option maxima_areas='small' should be selected, which "skeletonizes" the
    large areas to shrink them.

    The 'global' scope looks for global maxima in the CAM. This is accompanied by the parameter num_points, which
    designates the number of points returned by the function.

    :param image: An input image. Ideally this should be a low resolution CAM.
    :param scope: Can either be 'local' or 'global'. A 'local' scope looks for local maxima in the image, while a
                  'global' scope looks for global ones.
    :param maxima_areas: Can either be 'large' or 'small', depending on whether or not we want larger or smaller areas.
                         Only relevant for 'local' scopes.
    :param local_maxima_threshold: A float that filters out any local maxima that are below the threshold. Its default
                                   value is the average of the lowest-intensity local maxima with the highest-intensity
                                   one. Only relevant for 'local' scopes.
    :param num_points: An integer that specifies the number of points with the maximum intensity.
                       Only relevant for 'global' scopes.
    :return: A list of tuples, each containing the x and y coordinates of the 'focal points' in the input CAM.
    """

    # Global scope: looks for 'num_points' global maxima in the input image.
    if scope == 'global':

        # If 'num_points' is not defined, picks the square root of one of its dimensions:
        # e.g. for a 224x224 image: num_points = sqrt(224) = 15
        if num_points:
            if not isinstance(num_points, int):
                raise TypeError('num_points can only take integer values')
        else:
            num_points = int(round(np.sqrt(opt.im_size)))

        # Resizes the image to the desired size  and returns the coordinates of the top 'num_points' pixels that have
        # the largest values. They are cast as python's default 32-bit integers to be compatible with SimpleITK's
        # ConnectedThreshold function. The two axes are also reversed.
        top_points = np.argpartition(resize(image).ravel(), -num_points)[-num_points:]
        return [(int(x % opt.im_size), int(x // opt.im_size)) for x in top_points]

    # Local scope: looks for local maxima in the input image.
    elif scope == 'local':

        # Identifies the image's local maxima.
        candidate_points = morphology.local_maxima(image).astype(bool)

        # Because of the high sensitivity of scikit-image's morphology.local_maxima function, it is often desired to
        # filter some of the local maxima out via a threshold. If this is not passed explicitly the average of the
        # local maxima with the minimum and maximum intensities is used.
        if not isinstance(local_maxima_threshold, float):
            local_maxima_threshold = (image[candidate_points].max() + image[candidate_points].min()) / 2

        # Any local maxima that, whose intensity fails to exceed the threshold is ignored.
        focal_points = candidate_points * image > local_maxima_threshold

        # Resizes the map of the local maxima to the desired dimensions. This results in the enlargement of the areas
        # of the each maxima. If this is undesired, as indicated by the option maxima_areas='small', scikit-image's
        # morphology.skeletonize is applied, which shrinks the maxima areas.
        focal_points = resize(focal_points.astype(float), resample_method='nearest')
        if maxima_areas not in ('small', 'large'):
            raise ValueError("maxima_areas can either be 'small' or 'large'")
        elif maxima_areas == 'small':
            focal_points = morphology.skeletonize(focal_points)

        # Finally, the coordinates of the maxima are returned. They are cast as python's default 32-bit integers to be
        # compatible with SimpleITK's ConnectedThreshold function. The two axes are also reversed.
        focal_point_coods = np.where(focal_points)
        return [(int(focal_point_coods[1][i]), int(focal_point_coods[0][i])) for i in range(len(focal_point_coods[0]))]


def remove_small_holes(image, max_hole_size=256):
    """
    Wrapper to scikit-image's morphology.remove_small_holes that returns an image array with numbers instead of a
    boolean array.
    :param image: A segmentation mask (numpy.ndarray).
    :param max_hole_size: The maximum size (in pixels) of a hole to fill (int).
    :return: The filled segmentation mask (numpy.ndarray).
    """
    return morphology.remove_small_holes(image > 0, area_threshold=max_hole_size).astype(float)

