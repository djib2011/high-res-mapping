import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Bbox
import seaborn as sns
import utils
from utils import filters, maxima, segment, merge
import warnings


def pipeline(img, low, high, roi_percentile=85, focal_scope='global', maxima_areas='small', merge_type='blend',
             merge_alpha=0.5, filter_type='percentage', filter_percentage=15, filter_threshold=0.6):
    """
    Visualization of the whole workflow. Requires the original image and the high and low res CAMs to work. Performs
    the following steps:

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

    Note: it would be more efficient and elegant if I went for 'axes fraction' instead of 'data' for the coordinates
          of the ConnectionPatches, but it's too much of a hassle to change.

    :param img: Original RBG image, default shape=(224, 224, 3).
    :param low: Low-resolution CAM, default shape=(14, 14).
    :param high: High-resolution CAM, default shape=(224, 224).
    :param roi_percentile: Percentile based on which the ROI will be extracted. The default percentile=85 means that
                           the ROI will include the 15% highest-intensity pixels from the low-res map.
    :param focal_scope: The scope in which the focal points will be identified. 'global' looks for global maxima, while
                        'local' looks for local maxima. Accepted values: ['global', 'local']
    :param maxima_areas: Specifies the size of the focal points. Two options available: 'small' and 'large'.
    :param merge_type: Specifies the method of merging the high-res segment map with the low-res map.
                       Two methods available: 'blend' and 'multiply'. The first is a possibly weighted linear
                       combination of the two, while the second simply multiplies them.
    :param merge_alpha: If merge_type='blend', alpha regulates the importance of each of the two images (i.e. the low
                        and the high-res maps). Should be a float in [0, 1]. High values result in more influence from
                        the high-res map.
    :param filter_type: Specifies the method of segmenting the original image based on the combined CAM. Two methods are
                        available: 'percentage' and 'threshold'. The first keeps a percentage of the original image's
                        pixels while the second relies solely on the values of the combined CAM exceeding a threshold.
    :param filter_percentage: Selects the percentage of pixels to be included in the final segment. Only relevant if
                              filter_type='percentage'. Should be a number between 0 and 100.
    :param filter_threshold: Selects the threshold based on which the final segmentation will be performed. Only pixels
                             of the combined CAM that have an intensity greater than this threshold will be included.
                             Based on this mask, the original image will be segmented. Should be a float in [0, 1].
    """

    # Value checks

    # Categorical arguments
    if maxima_areas not in ('small', 'large'):
        raise ValueError("available options for maxima_areas are: 'small' and 'large'.")

    if merge_type not in ('blend', 'multiply'):
        raise ValueError("available options for merge_type are: 'blend' and 'multiply'.")

    if filter_type not in ('percentage', 'threshold'):
        raise ValueError("vailable options for filter_type are: 'percentage' and 'threshold'.")

    # Percentage arguments
    if roi_percentile <= 0 or roi_percentile >= 100:
        raise ValueError('roi_percentile should be a percentage in (0, 100)')
    elif roi_percentile < 1:
        warnings.warn('roi_percentile value in [0, 1). Should be defined as a percentage in (0, 100), '
                      'e.g. If the desired percentage is 13%, pass 33 instead of 0.33!')

    if filter_percentage <= 0 or filter_percentage >= 100:
        raise ValueError('filter_percentage should be a percentage in (0, 100)')
    elif filter_percentage < 1:
        warnings.warn('filter_percentage value in [0, 1). Should be defined as a percentage in (0, 100), '
                      'e.g. If the desired percentage is 13%, pass 33 instead of 0.33!')

    # Value arguments
    if merge_alpha < 0 or merge_alpha > 1:
        raise ValueError('merge_alpha should be a float in [0, 1]')

    if filter_threshold < 0 or filter_threshold > 1:
        raise ValueError('filter_threshold should be a float in [0, 1]')

    # Coordinates of the top/bottom/left/right/middle of the input image
    left = (0, img.shape[1] / 2)
    right = (img.shape[1], img.shape[1] / 2)
    bottom = (img.shape[1] / 2, img.shape[1])
    top = (img.shape[1] / 2, 0)
    midpoint = (img.shape[1] / 2, img.shape[1] / 2)

    # Create two 'blank' images for filling empty positions
    blank = np.ones(img[0].shape, dtype=np.uint8)
    half_blank = blank[::2]

    # Initialize 5x7 grid
    fig, ax = plt.subplots(5, 7, figsize=(16, 16))

    ##############################
    ######## First column ########
    ##############################

    # Fill first, second, fourth and fifth rows with blank images
    ax[0, 0].imshow(blank, alpha=0)
    ax[0, 0].axis('off')
    ax[1, 0].imshow(blank, alpha=0)
    ax[1, 0].axis('off')
    ax[3, 0].imshow(blank, alpha=0)
    ax[3, 0].axis('off')
    ax[4, 0].imshow(half_blank, alpha=0)
    ax[4, 0].axis('off')

    # Add original image to the third row
    ax[2, 0].imshow(img[0], zorder=3)
    ax[2, 0].axis('off')
    ax[2, 0].set_title('Original image', backgroundcolor='white', zorder=2)

    # Three crooked lines starting from the first row, represented by thirteen (!) connection patches
    # Connection of 'original image' to 'high-res map'
    con1a = ConnectionPatch(xyA=top, xyB=midpoint, coordsA='data', coordsB='data',
                            axesA=ax[2, 0], axesB=ax[1, 0], color='black', lw=2, zorder=1)
    con1b = ConnectionPatch(xyA=midpoint, xyB=left, coordsA='data', coordsB='data',
                            axesA=ax[1, 0], axesB=ax[1, 1], color='black', lw=2, arrowstyle='->')

    # Connection of 'original image' to 'low-res map'
    con2a = ConnectionPatch(xyA=bottom, xyB=midpoint, coordsA='data', coordsB='data',
                            axesA=ax[2, 0], axesB=ax[3, 0], color='black', lw=2)
    con2b = ConnectionPatch(xyA=midpoint, xyB=left, coordsA='data', coordsB='data',
                            axesA=ax[3, 0], axesB=ax[3, 1], color='black', lw=2, arrowstyle='->')

    # Connection of 'original image' to 'result'
    con3b = ConnectionPatch(xyA=midpoint, xyB=bottom, coordsA='data', coordsB='data',
                            axesA=ax[1, 0], axesB=ax[0, 0], color='black', lw=2)
    con3c = ConnectionPatch(xyA=bottom, xyB=bottom, coordsA='data', coordsB='data',
                            axesA=ax[0, 0], axesB=ax[0, 1], color='black', lw=2)
    con3d = ConnectionPatch(xyA=bottom, xyB=bottom, coordsA='data', coordsB='data',
                            axesA=ax[0, 1], axesB=ax[0, 2], color='black', lw=2)
    con3e = ConnectionPatch(xyA=bottom, xyB=bottom, coordsA='data', coordsB='data',
                            axesA=ax[0, 2], axesB=ax[0, 3], color='black', lw=2)
    con3f = ConnectionPatch(xyA=bottom, xyB=bottom, coordsA='data', coordsB='data',
                            axesA=ax[0, 3], axesB=ax[0, 4], color='black', lw=2)
    con3g = ConnectionPatch(xyA=bottom, xyB=bottom, coordsA='data', coordsB='data',
                            axesA=ax[0, 4], axesB=ax[0, 5], color='black', lw=2)
    con3h = ConnectionPatch(xyA=bottom, xyB=bottom, coordsA='data', coordsB='data',
                            axesA=ax[0, 5], axesB=ax[0, 6], color='black', lw=2)
    con3i = ConnectionPatch(xyA=bottom, xyB=midpoint, coordsA='data', coordsB='data',
                            axesA=ax[0, 6], axesB=ax[1, 6], color='black', lw=2)
    con3k = ConnectionPatch(xyA=midpoint, xyB=midpoint, coordsA='data', coordsB='data',
                            axesA=ax[1, 6], axesB=ax[2, 6], color='black', lw=2)
    con3l = ConnectionPatch(xyA=midpoint, xyB=top, coordsA='data', coordsB='data',
                            axesA=ax[2, 6], axesB=ax[3, 6], color='black', lw=2, arrowstyle='->', zorder=1)

    # Add each patch to its respective axis
    ax[2, 0].add_artist(con1a)
    ax[1, 0].add_artist(con1b)

    ax[2, 0].add_artist(con2a)
    ax[3, 0].add_artist(con2b)

    ax[1, 0].add_artist(con3b)
    ax[0, 0].add_artist(con3c)
    ax[0, 1].add_artist(con3d)
    ax[0, 2].add_artist(con3e)
    ax[0, 3].add_artist(con3f)
    ax[0, 4].add_artist(con3g)
    ax[0, 5].add_artist(con3h)
    ax[0, 6].add_artist(con3i)
    ax[1, 6].add_artist(con3k)
    ax[2, 6].add_artist(con3l)

    ###############################
    ######## Second column ########
    ###############################

    # High-res map on the second line
    ax[1, 1].imshow(high)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('High-res CAM')

    # Low-res map on the fourth line
    ax[3, 1].imshow(utils.resize(low), zorder=3)
    ax[3, 1].axis('off')
    ax[3, 1].set_title('Low-res CAM', backgroundcolor='white', zorder=2)

    # Fill the first, third and fifth lines with blank images
    ax[0, 1].imshow(blank, alpha=0)
    ax[0, 1].axis('off')
    ax[2, 1].imshow(blank, alpha=0)
    ax[2, 1].axis('off')
    ax[4, 1].imshow(half_blank, alpha=0)
    ax[4, 1].axis('off')

    # Four lines represented by eleven (!) connection patches
    # Connection of 'high-res map' to 'gradient'
    con4 = ConnectionPatch(xyA=right, xyB=left, coordsA='data', coordsB='data',
                           axesA=ax[1, 1], axesB=ax[1, 2], color='black', lw=2, arrowstyle='->')

    # Connection of 'low-res map' to 'roi'
    con5a = ConnectionPatch(xyA=top, xyB=midpoint, coordsA='data', coordsB='data',
                            axesA=ax[3, 1], axesB=ax[2, 1], color='black', lw=2, zorder=1)
    con5b = ConnectionPatch(xyA=midpoint, xyB=left, coordsA='data', coordsB='data',
                            axesA=ax[2, 1], axesB=ax[2, 2], color='black', lw=2, arrowstyle='->')

    # Connection of 'low-res map' to 'focal points'
    con6 = ConnectionPatch(xyA=right, xyB=left, coordsA='data', coordsB='data',
                           axesA=ax[3, 1], axesB=ax[3, 2], color='black', lw=2, arrowstyle='->')

    # Connection of 'low-res map' to 'merger'
    con7a = ConnectionPatch(xyA=bottom, xyB=top, coordsA='data', coordsB='data',
                            axesA=ax[3, 1], axesB=ax[4, 1], color='black', lw=2, zorder=1)
    con7b = ConnectionPatch(xyA=top, xyB=top, coordsA='data', coordsB='data',
                            axesA=ax[4, 1], axesB=ax[4, 2], color='black', lw=2, zorder=1)
    con7c = ConnectionPatch(xyA=top, xyB=top, coordsA='data', coordsB='data',
                            axesA=ax[4, 2], axesB=ax[4, 3], color='black', lw=2, zorder=1)
    con7d = ConnectionPatch(xyA=top, xyB=top, coordsA='data', coordsB='data',
                            axesA=ax[4, 3], axesB=ax[4, 4], color='black', lw=2, zorder=1)
    con7e = ConnectionPatch(xyA=top, xyB=top, coordsA='data', coordsB='data',
                            axesA=ax[4, 4], axesB=ax[4, 5], color='black', lw=2, zorder=1)
    con7f = ConnectionPatch(xyA=top, xyB=bottom, coordsA='data', coordsB='data',
                            axesA=ax[4, 5], axesB=ax[3, 5], color='black', lw=2, zorder=1, arrowstyle='->')

    # Add the patches to their respective axes
    ax[1, 1].add_artist(con4)
    ax[3, 1].add_artist(con5a)
    ax[2, 1].add_artist(con5b)
    ax[3, 1].add_artist(con6)
    ax[3, 1].add_artist(con7a)
    ax[4, 1].add_artist(con7b)
    ax[4, 2].add_artist(con7c)
    ax[4, 3].add_artist(con7d)
    ax[4, 4].add_artist(con7e)
    ax[4, 5].add_artist(con7f)

    ##############################
    ######## Third column ########
    ##############################

    # High-res blur
    blurred = filters.blur(high)
    ax[1, 2].imshow(blurred)
    ax[1, 2].axis('off')
    ax[1, 2].set_title('Blurred')

    # Region of Interest
    roi = utils.resize(low) > utils.percentile(utils.resize(low), roi_percentile)
    a = ax[2, 2].imshow(roi)
    ax[2, 2].axis('off')
    ax[2, 2].set_title('Region of Interest')

    # Focal Points
    focal_points = maxima.find_focal_points(low, scope=focal_scope, maxima_areas=maxima_areas)
    bg, dots = a.get_cmap().colors[0], a.get_cmap().colors[-1]
    ax[3, 2].imshow((blank.reshape(-1, 3) * bg).reshape(img.shape[1], img.shape[1], 3))
    ax[3, 2].scatter([x[0] for x in focal_points], [x[1] for x in focal_points], marker='x', s=30, c=dots)
    ax[3, 2].axis('off')
    ax[3, 2].set_title('Focal Points')

    # Fill first and fifth rows with blank images
    ax[0, 2].imshow(blank, alpha=0)
    ax[0, 2].axis('off')
    ax[4, 2].imshow(half_blank, alpha=0)
    ax[4, 2].axis('off')

    # Three lines represented by five connection patches
    con8 = ConnectionPatch(xyA=right, xyB=left, coordsA='data', coordsB='data',
                           axesA=ax[1, 2], axesB=ax[1, 3], color='black', lw=2, arrowstyle='->')
    con9 = ConnectionPatch(xyA=right, xyB=(0, 0.5), coordsA='data', coordsB='axes fraction',
                           axesA=ax[2, 2], axesB=ax[2, 3], color='black', lw=2, arrowstyle='->')
    con10a = ConnectionPatch(xyA=right, xyB=midpoint, coordsA='data', coordsB='data',
                             axesA=ax[3, 2], axesB=ax[3, 3], color='black', lw=2)
    con10b = ConnectionPatch(xyA=midpoint, xyB=midpoint, coordsA='data', coordsB='data',
                             axesA=ax[3, 3], axesB=ax[3, 4], color='black', lw=2)
    con10c = ConnectionPatch(xyA=midpoint, xyB=left, coordsA='data', coordsB='data',
                             axesA=ax[3, 4], axesB=ax[3, 5], color='black', lw=2, arrowstyle='->')

    # Add the patches to their respective axes
    ax[1, 2].add_artist(con8)
    ax[2, 2].add_artist(con9)
    ax[3, 2].add_artist(con10a)
    ax[3, 3].add_artist(con10b)
    ax[3, 4].add_artist(con10c)

    ###############################
    ######## Fourth column ########
    ###############################

    # High-res edge detection
    grad = utils.normalize_image(filters.sobel(blurred))
    ax[1, 3].imshow(grad)
    ax[1, 3].axis('off')
    ax[1, 3].set_title('Edge detection')

    # Gradient percentiles
    roi_grad = grad[roi]
    lower = utils.percentile(roi_grad, 25)
    upper = utils.percentile(roi_grad, 75)
    ax[2, 3] = sns.distplot(roi_grad.ravel(), ax=ax[2, 3])
    ax[2, 3].plot([lower, lower], [0, 4], c='C1')
    ax[2, 3].plot([upper, upper], [0, 4], c='C1')
    ax[2, 3].text(lower, -0.5, 'lower', color='C1', horizontalalignment='center')
    ax[2, 3].text(upper, 4.5, 'upper', color='C1', horizontalalignment='center')
    ax[2, 3].axis('off')
    ttl = ax[2, 3].set_title('Edge Histogram')
    ttl.set_bbox(dict(color='white', alpha=0.5, zorder=2))
    square_axes(ax[2, 3])  # custom function that shrinks the axis object to a square box

    # Fill first, fourth and fifth rows
    ax[0, 3].imshow(blank, alpha=0)
    ax[0, 3].axis('off')
    ax[3, 3].imshow(blank, alpha=0)
    ax[3, 3].axis('off')
    ax[4, 3].imshow(half_blank, alpha=0)
    ax[4, 3].axis('off')

    # Three lines represented by four connection patches
    con11 = ConnectionPatch(xyA=bottom, xyB=(0.5, 1), coordsA='data', coordsB='axes fraction',
                            axesA=ax[1, 3], axesB=ax[2, 3], color='black', lw=2, arrowstyle='->')
    con12a = ConnectionPatch(xyA=right, xyB=midpoint, coordsA='data', coordsB='data',
                             axesA=ax[1, 3], axesB=ax[1, 4], color='black', lw=2)
    con12b = ConnectionPatch(xyA=midpoint, xyB=top, coordsA='data', coordsB='data',
                             axesA=ax[1, 4], axesB=ax[2, 4], color='black', lw=2, arrowstyle='->', zorder=1)

    con13 = ConnectionPatch(xyA=(1, 0.5), xyB=left, coordsA='axes fraction', coordsB='data',
                            axesA=ax[2, 3], axesB=ax[2, 4], color='black', lw=2, arrowstyle='->')

    # Add the patches to their respective axes
    ax[1, 3].add_artist(con11)
    ax[1, 3].add_artist(con12a)
    ax[1, 4].add_artist(con12b)
    ax[2, 3].add_artist(con13)

    ##############################
    ######## Fifth column ########
    ##############################

    # Region Growing Segmentation
    segm = segment.region_growing(grad, seeds=focal_points, lower=lower, upper=upper)
    ax[2, 4].imshow(segm, zorder=3)
    ax[2, 4].axis('off')
    ttl = ax[2, 4].set_title('Region Growing\nSegmentation')
    ttl.set_bbox(dict(color='white', alpha=0.5, zorder=2))

    # Fill first, second fourth and fifth rows
    ax[0, 4].imshow(blank, alpha=0)
    ax[0, 4].axis('off')
    ax[1, 4].imshow(blank, alpha=0)
    ax[1, 4].axis('off')
    ax[3, 4].imshow(blank, alpha=0)
    ax[3, 4].axis('off')
    ax[4, 4].imshow(half_blank, alpha=0)
    ax[4, 4].axis('off')

    # Just one connection! :)
    con14 = ConnectionPatch(xyA=right, xyB=left, coordsA='data', coordsB='data',
                            axesA=ax[2, 4], axesB=ax[2, 5], color='black', lw=2, arrowstyle='->')

    ax[2, 4].add_artist(con14)

    ##############################
    ######## Sixth column ########
    ##############################

    # Add edges and fill small holes
    edges = (grad >= upper).astype(float)
    roi_edges = edges * roi
    segm_with_edges = segm + roi_edges
    filled = maxima.remove_small_holes(segm_with_edges)
    ax[2, 5].imshow(filled)
    ax[2, 5].axis('off')
    ax[2, 5].set_title('Remove small holes')

    # High-Low merger
    merged = merge.merge_images(filled, low, method=merge_type, alpha=merge_alpha)
    ax[3, 5].imshow(merged)
    ax[3, 5].axis('off')
    ttl = ax[3, 5].set_title('High-Low Merger')
    ttl.set_bbox(dict(color='white', alpha=0.5, zorder=2))

    # Fill remaining rows
    ax[0, 5].imshow(blank, alpha=0)
    ax[0, 5].axis('off')
    ax[1, 5].imshow(blank, alpha=0)
    ax[1, 5].axis('off')
    ax[3, 5].imshow(blank, alpha=0)
    ax[3, 5].axis('off')
    ax[4, 5].imshow(half_blank, alpha=0)
    ax[4, 5].axis('off')

    # Last connection patches...
    con15 = ConnectionPatch(xyA=bottom, xyB=top, coordsA='data', coordsB='data',
                            axesA=ax[2, 5], axesB=ax[3, 5], color='black', lw=2, zorder=-1, arrowstyle='->')
    con16 = ConnectionPatch(xyA=right, xyB=left, coordsA='data', coordsB='data',
                            axesA=ax[3, 5], axesB=ax[3, 6], color='black', lw=2, zorder=-1, arrowstyle='->')

    ax[2, 5].add_artist(con15)
    ax[3, 5].add_artist(con16)

    ################################
    ######## Seventh column ########
    ################################

    # Result
    if filter_type == 'percentage':
        result = merge.keep_percentage(img, merged, percentage=filter_percentage/100)
    else:
        result = merge.filter_image(img, merged, threshold=filter_threshold)
    ax[3, 6].imshow(result, zorder=3)
    ax[3, 6].axis('off')
    ttl = ax[3, 6].set_title('Result')
    ttl.set_bbox(dict(color='white', alpha=0.5, zorder=2))

    # Fill remaining rows
    ax[0, 6].imshow(blank, alpha=0)
    ax[0, 6].axis('off')
    ax[1, 6].imshow(blank, alpha=0)
    ax[1, 6].axis('off')
    ax[2, 6].imshow(blank, alpha=0)
    ax[2, 6].axis('off')
    ax[4, 6].imshow(half_blank, alpha=0)
    ax[4, 6].axis('off')


def plt_to_static(axes):
    """
    Should convert an axis object to an image in a numpy.array. Doesn't work as intended!
    :param axes: A matplotlib.axes.Axes object
    :return: The same object as a numpy.array
    """
    fig = plt.figure()
    fig.axes.append(axes)
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)


def square_axes(axes):
    """
    Takes a matplotlib.axes.Axes object, finds its height and width and shrinks the largest dimension to match the
    smallest one. Caution: it actually changes the object (in-place)!
    :param axes: A matplotlib.axes.Axes object.
    :return: The new Bbox coordinates.
    """

    bbox = axes.get_position()._points.copy()
    width = bbox[1, 0] - bbox[0, 0]
    height = bbox[1, 1] - bbox[0, 1]

    if width < height:
        center = bbox[0, 1] + height / 2
        bbox[0, 1] = center - width / 2
        bbox[1, 1] = center + width / 2
    else:
        center = bbox[0, 0] + width / 2
        bbox[0, 0] = center - height / 2
        bbox[1, 0] = center + height / 2

    axes.set_position(Bbox(bbox))
    return bbox

