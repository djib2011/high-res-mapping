from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils
from utils import filters, maxima, segment, merge


def pipeline(img, low, high, roi_percentile=85, focal_scope='global', maxima_areas='small', merge_type='blend',
             merge_alpha=0.5, filter_type='percentage', filter_percentage=15, filter_threshold=0.6):
    """
    The whole postprocessing pipeline, returning step-by-step results.

    In detail the postprocessing pipeline involves the following steps:

    1. Applies a filter to blur the high-res map.
    2. Extracts the ROI from the low-res map through a percentile.
    3. Identifies the focal points of the low-res map by locating it's local maxima.
    4. Computes the gradient of the high-res map through a sobel filter.
    5. Draws a histogram of the gradient. Only considers areas corresponding to the ROI extracted from the low-res map.
    6. Calculates a 'lower' and 'upper' bound on the 25th and 75th percentile, respectively.
    7. Performs a region-growing segmentation algorithm on the gradient. The boundaries are the previous percentiles,
       while the focal points are set as the initial seeds (from where to start growing).
    8. Merges the result of the segmentation with the low-res map.
    9. Segments the original image according to the result of the previous merger.

    :param img: The original image (numpy.ndarray).
    :param low: The low-resolution Class Activation Map (numpy.ndarray).
    :param high: The high-resolution Class Activation Map (numpy.ndarray).
    :param roi_percentile: The percentile above which the ROI will be estimated. roi_percentile=85 means that the 15%
                           highest intensity pixels of the low-res map will constitute the ROI (int in (0, 100)).
    :param focal_scope: The scope in which the focal points will be identified. 'global' looks for global maxima, while
                        'local' looks for local maxima. Accepted values: ['global', 'local']
    :param maxima_areas: Can either be 'large' or 'small', depending on whether or not we want larger or smaller areas.
                         Only relevant for 'local' scopes. Accepted values: ['global', 'local']
    :param merge_type: Selection on whether to multiply or blend the high with the low-res CAMs after processing.
                       Accepted values: ['blend', 'merge']
    :param merge_alpha: Parameter for the blend merge method. Higher values result in more influence from the high-res
                        map. Should be a float in [0, 1].
    :param filter_type: Selects how to crop the original image according to the refined CAM. Two options are available:
                        - 'percentage', which keeps a percentage of the highest-instensity values of the refined CAM
                        - 'threshold', which keeps the intensities above a certain threshold
    :param filter_percentage: A float representing the percentage of pixels to be kept (should be in [0, 1]). Only
                              relevant when filter_type='percentage'
    :param filter_threshold: A float in [0, 1] over which the intensities of the refined CAM will be kept. Only relevant
                             when filter_type='threshold'
    :return: A dictionary with all intermediate results from the postprocessing pipeline. In detail:
             - 'blurred': The blurred high-res CAM.
             - 'low': The original low-res CAM.
             - 'low_resized': The resized low-res CAM (through bilinear interpolation).
             - 'edges': The result of the sobel filter on the blurred high-res map.
             - 'roi': The Region Of Interest extracted from the low-res map.
             - 'bounds': The lower and upper bounds for the region-growing segmentation.
             - 'focal_points': The focal_points extracted from the low-res map.
             - 'segmented': The output of the region-growing segmentation.
             - 'full_segment': The filled segmentation.
             - 'merged': The merger of the segmentation with the low-res map, i.e. the refined CAM.
             - 'result': The part of the original image that has been cropped according to the regined CAM.
    """

    # High-res processing
    blurred = filters.blur(high)
    grad = utils.normalize_image(filters.sobel(blurred))

    # Low-res processing
    roi = utils.resize(low) > utils.percentile(utils.resize(low), roi_percentile)
    upper = utils.percentile(grad[roi], 75)
    lower = utils.percentile(grad[roi], 25)
    focal_points = maxima.find_focal_points(low, scope=focal_scope, maxima_areas=maxima_areas)

    # Region growing segmentation
    segm = segment.region_growing(grad, seeds=focal_points, lower=lower, upper=upper)

    # Segment processing
    edges = (grad >= upper).astype(float)
    roi_edges = edges * roi
    segm_with_edges = segm + roi_edges
    filled = maxima.remove_small_holes(segm_with_edges)

    # Merger
    merged = merge.merge_images(filled, low, method=merge_type, alpha=merge_alpha)

    if filter_type == 'percentage':
        result = merge.keep_percentage(img, merged, percentage=filter_percentage/100)
    elif filter_type == 'threshold':
        result = merge.filter_image(img, merged, threshold=filter_threshold)

    return {'blurred': blurred, 'low': low, 'low_resized': utils.resize(low), 'edges': grad, 'roi': roi,
            'bounds': (lower, upper), 'focal_points': focal_points, 'segmented': segm, 'full_segment': filled,
            'merged': merged, 'result': result}

