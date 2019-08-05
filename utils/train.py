from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np


class BalancedGenerator(ImageDataGenerator):
    """
    Generate minibatches of image data with real-time data augmentation,
        while randomly over-sampling to fix the class imbalance.

        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        # Initialize variables and methods from base class (ImageDataGenerator)
        super(BalancedGenerator, self).__init__(featurewise_center=featurewise_center,
                                                samplewise_center=samplewise_center,
                                                featurewise_std_normalization=featurewise_std_normalization,
                                                samplewise_std_normalization=samplewise_std_normalization,
                                                zca_whitening=zca_whitening,
                                                zca_epsilon=zca_epsilon,
                                                rotation_range=rotation_range,
                                                width_shift_range=width_shift_range,
                                                height_shift_range=height_shift_range,
                                                shear_range=shear_range,
                                                zoom_range=zoom_range,
                                                channel_shift_range=channel_shift_range,
                                                fill_mode=fill_mode,
                                                cval=cval,
                                                horizontal_flip=horizontal_flip,
                                                vertical_flip=vertical_flip,
                                                rescale=rescale,
                                                preprocessing_function=preprocessing_function,
                                                data_format=data_format)

    def flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb', classes=None,
                            class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None,
                            save_prefix='', save_format='png', follow_links=False, subset=None,
                            interpolation='nearest'):
        """
        Used for generating batches of images directly from a directory. Supports
        on-line data augmentation and random over-sampling.

        Returns a DirectoryIterator that thinks it has more samples (from the
        under-represented classes) than it actually does
        """

        # Initialize the DirectoryIterator
        it = DirectoryIterator(directory, self, target_size=target_size, color_mode=color_mode,
                               classes=classes, class_mode=class_mode, data_format=self.data_format,
                               batch_size=batch_size, shuffle=shuffle, seed=seed, save_to_dir=save_to_dir,
                               save_prefix=save_prefix, save_format=save_format, follow_links=follow_links,
                               interpolation=interpolation)

        # Define target number of images for each class to reach
        self.target = np.bincount(it.classes).max()

        # Create lists containing the images and their respective labels,
        # sampled multiple times if necessary in order to reach the target
        # number for each class.
        new_filenames = []
        new_classes = []
        print('\n ' + '-' * 31 + ' ')
        print('|         |  Number of images   |')
        print('|  Class  |  ------------------ |')
        print('|         | Previous | Current  |')
        print(' ' + '-' * 31 + ' ')
        for c in range(it.num_classes):
            new_filenames += self.balance(np.array(it.filenames)[it.classes == c], class_name=c, seed=seed)
            new_classes += [c] * self.target
        print(' ' + '-' * 31 + ' \n')

        # Replaces the DirectoryIterator's lists with the ones we created
        it.filenames = new_filenames
        it.classes = np.array(new_classes)
        del new_filenames, new_classes

        # Replaces the maximum number of samples in the DirectoryIterator
        it.n = it.samples = len(it.filenames)

        print('Total number of images after Over-Sampling:', it.n)

        return it

    def balance(self, index_slice, class_name, seed=None):
        """
        Function for randomly sampling images multiple times in order to reach
        a target.
        """
        np.random.seed(seed)
        current = len(index_slice)

        print('|  {:^6} | {:^8} | {:^8} |'.format(class_name, current, self.target))

        return list(index_slice) * (self.target // current) + \
               list(index_slice[np.random.randint(current,
                                                  size=(self.target % current))])


class CustomImageGenerator(ImageDataGenerator):
    """
    Customized ImageDataGenerator.
    # Arguments
        featurewise_center: Boolean. Set input mean to 0 over the dataset, feature-wise.
        samplewise_center: Boolean. Set each sample mean to 0.
        featurewise_std_normalization: Boolean. Divide inputs by std of the dataset, feature-wise.
        samplewise_std_normalization: Boolean. Divide each input by its std.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening: Boolean. Apply ZCA whitening.
        rotation_range: Int. Degree range for random rotations.
        width_shift_range: Float (fraction of total width). Range for random horizontal shifts.
        height_shift_range: Float (fraction of total height). Range for random vertical shifts.
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        zoom_range: Float or [lower, upper]. Range for random zoom. If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: Float. Range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.  Default is 'nearest'.
        Points outside the boundaries of the input are filled according to the given mode:
            'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
            'nearest':  aaaaaaaa|abcd|dddddddd
            'reflect':  abcddcba|abcd|dcbaabcd
            'wrap':  abcdabcd|abcd|abcdabcd
        cval: Float or Int. Value used for points outside the boundaries when `fill_mode = "constant"`.
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling is applied,
                otherwise we multiply the data by the value provided (before applying
                any other transformation).
        preprocessing_function: function that will be implied on each input.
                The function will run after the image is resized and augmented.
                The function should take one argument:
                one image (Numpy tensor with rank 3),
                and should output a Numpy tensor with the same shape.
        data_format: One of {"channels_first", "channels_last"}.
            "channels_last" mode means that the images should have shape `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: Float. Fraction of images reserved for validation (strictly between 0 and 1).
    # Examples
    Example of using `.flow(x, y)`:
    ```python
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs)
    # here's a more "manual" example
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
    ```
    Example of using `.flow_from_directory(directory)`:
    ```python
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
    ```
    Example of transforming images and masks together.
    ```python
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    image_generator = image_datagen.flow_from_directory(
        'data/images',
        class_mode=None,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        'data/masks',
        class_mode=None,
        seed=seed)
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50)
    ```
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 output_names=None):
        self.output_names = output_names

        super(CustomImageGenerator, self).__init__(featurewise_center=featurewise_center,
                                                   samplewise_center=samplewise_center,
                                                   featurewise_std_normalization=featurewise_std_normalization,
                                                   samplewise_std_normalization=samplewise_std_normalization,
                                                   zca_whitening=zca_whitening,
                                                   zca_epsilon=zca_epsilon,
                                                   rotation_range=rotation_range,
                                                   width_shift_range=width_shift_range,
                                                   height_shift_range=height_shift_range,
                                                   shear_range=shear_range,
                                                   zoom_range=zoom_range,
                                                   channel_shift_range=channel_shift_range,
                                                   fill_mode=fill_mode,
                                                   cval=cval,
                                                   horizontal_flip=horizontal_flip,
                                                   vertical_flip=vertical_flip,
                                                   rescale=rescale,
                                                   preprocessing_function=preprocessing_function,
                                                   data_format=data_format)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest',
                            output_names=None):
        """Takes the path to a directory, and generates batches of augmented/normalized data.
        # Arguments
                directory: path to the target directory.
                 It should contain one subdirectory per class.
                 Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator.
                See [this script](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d) for more details.
                target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
                 The dimensions to which all images found will be resized.
                color_mode: one of "grayscale", "rbg". Default: "rgb".
                 Whether the images will be converted to have 1 or 3 color channels.
                classes: optional list of class subdirectories (e.g. `['dogs', 'cats']`).
                 Default: None. If not provided, the list of classes will
                 be automatically inferred from the subdirectory names/structure under `directory`,
                 where each subdirectory will be treated as a different class
                 (and the order of the classes, which will map to the label indices, will be alphanumeric).
                 The dictionary containing the mapping from class names to class
                 indices can be obtained via the attribute `class_indices`.
                class_mode: one of "categorical", "binary", "sparse", "input" or None.
                 Default: "categorical". Determines the type of label arrays that are
                 returned: "categorical" will be 2D one-hot encoded labels, "binary" will be 1D binary labels,
                 "sparse" will be 1D integer labels, "input" will be images identical to input images (mainly used to work with autoencoders).
                 If None, no labels are returned (the generator will only yield batches of image data, which is useful to use
                 `model.predict_generator()`, `model.evaluate_generator()`, etc.).
                  Please note that in case of class_mode None,
                   the data still needs to reside in a subdirectory of `directory` for it to work correctly.
                batch_size: size of the batches of data (default: 32).
                shuffle: whether to shuffle the data (default: True)
                seed: optional random seed for shuffling and transformations.
                save_to_dir: None or str (default: None). This allows you to optionally specify a directory to which to save
                 the augmented pictures being generated (useful for visualizing what you are doing).
                save_prefix: str. Prefix to use for filenames of saved pictures (only relevant if `save_to_dir` is set).
                save_format: one of "png", "jpeg" (only relevant if `save_to_dir` is set). Default: "png".
                follow_links: whether to follow symlinks inside class subdirectories (default: False).
        # Returns
            A CustomDirectoryIterator yielding tuples of `(x, y)` where `x` is a numpy array of image data and
             `y` is a numpy array of corresponding labels.
        """
        return CustomDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            output_names=output_names)


class CustomDirectoryIterator(DirectoryIterator):
    """
    Custom DirectoryIterator.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 output_names=None):
        self.output_names = output_names

        super(CustomDirectoryIterator, self).__init__(directory, image_data_generator,
                                                      target_size=target_size, color_mode=color_mode,
                                                      classes=classes, class_mode=class_mode,
                                                      batch_size=batch_size, shuffle=shuffle, seed=seed,
                                                      data_format=data_format, save_to_dir=save_to_dir,
                                                      save_prefix=save_prefix, save_format=save_format,
                                                      follow_links=follow_links, interpolation=interpolation)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x, batch_y = super(CustomDirectoryIterator, self)._get_batches_of_transformed_samples(index_array)
        # return batch_x, [batch_x, batch_y]
        if self.output_names:
            return batch_x, {self.output_names[0]: batch_x, self.output_names[1]: batch_y}
        return batch_x, {'main_output': batch_x, 'aux_output': batch_y}


def transfer_weights(pretrained_model, new_model):
    """
    Function meant to transfer the weights from a HalfDenseNetFCN to an AuxiliaryDenseNetFCN
    :param pretrained_model: The model from which we want to get the weights.
    :param new_model: The model to which we want to set the weights.
    :return: True, if the transfer is successful.
    """
    # As long as the models' architectures match, transfer the weights
    i = 0
    if pretrained_model.input_shape != new_model.input_shape:
        raise ValueError('Models should have matching input shapes.')
    while i < len(pretrained_model.layers) and \
            pretrained_model.layers[i].output_shape == new_model.layers[i].output_shape:
        new_model.layers[i].set_weights(pretrained_model.layers[i].get_weights())
        i += 1
    # For the rest of the layers (i.e. the auxiliary ones), we need to find them by name.
    for layer in pretrained_model.layers[i:]:
        new_model.get_layer(layer.name).set_weights(layer.get_weights())
    print('Successfully transferred weights from {} layers.'.format(len(pretrained_model.layers)))
    return True


def normalize_imagenet(img):
    """
    This function is meant to be run with ImageDataGenerator
    in order to normalize the images with the mean values of
    the ImageNet dataset and feed them to an ImageNet-pretrained
    model.

    The exact normalization technique is taken from:
        https://github.com/flyyufelix/cnn_finetune
    since we use that pretrained model.
    """
    img[:, :, 0] = (img[:, :, 0] - 123.68) * 0.017
    img[:, :, 1] = (img[:, :, 1] - 116.78) * 0.017
    img[:, :, 2] = (img[:, :, 2] - 103.94) * 0.017

    return img

