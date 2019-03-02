import custom_models
from utils import opt
import numpy as np
from keras.models import Model


class FullModel:

    def __init__(self, weights=opt.weight_dir, num_classes=opt.num_classes, im_size=opt.im_size, channels=opt.channels):
        """
        Wrapper for easier predictions and debugging of an AuxiliaryDenseNetFCN model.
        :param weights: Path to a file containing the weights of a pre-trained AuxiliaryDenseNetFCN.
        :param num_classes: An integer representing the number of classes.
        :param im_size: An integer representing the size of a single dimension each image.
        :param channels: How many channels does the image have (e.g. 1 -> grayscale, 3 -> RGB)
        """

        # Print info during instantiation
        print('Instantiating a Full Model:')
        print('weights:     {}'.format(weights))
        print('classes:     {}'.format(num_classes))

        # Calculate image dimensions
        image_dims = (im_size, im_size)
        input_shape = image_dims + (channels,)
        print('input shape: {}'.format(input_shape))

        # Main model
        self.model = custom_models.AuxiliaryDenseNetFCN(input_shape, num_classes, weights)

        # High and low-res Ak models
        self.Ak_model = Model(inputs=self.model.input, outputs=self.model.layers[-4].output)
        self.Ak_model_low = Model(inputs=self.model.input, outputs=self.model.layers[-5].output)

        # Wk array
        self.Wk = self.model.layers[-1].get_weights()[0]

    def generate_cams(self, image, custom_class=None):
        """
        Generate the high and low resolution CAMs from a given image, as well as model's output.
        :param img: An image in numpy.ndarray format with a shape of (1, height, width, channels).
        :return: The high-resolution CAM, the low-resolution CAM and the prediction array.
        """

        # Main model's output
        preds = self.model.predict(image)[1]  # keep the output of the 'half' model that does the classification
        prediction = np.argmax(preds)  # identify the model's prediction

        # Ak models' outputs
        Ak = self.Ak_model.predict(image)
        Ak_low = self.Ak_model_low.predict(image)

        if custom_class or custom_class == 0:
            prediction = custom_class

        # Create high and low-res CAMs
        high_res_cam = np.dot(np.squeeze(Ak), self.Wk)[..., prediction]
        low_res_cam = np.dot(np.squeeze(Ak_low), self.Wk)[..., prediction]

        return high_res_cam, low_res_cam, preds

    def predict(self, image):
        """
        Predict the class of an image.
        :param image: An image (numpy.ndarray).
        :return: The model's prediction (int).
        """

        preds = self.model.predict(image)[1]

        return np.argmax(preds)

    def high_res_cam(self, image, custom_class=None):
        """
        Generates the high-resolution Class Activation Map (CAM) for a given class. If no class is specified, the
        prediction for the given image will be used.
        :param image: An image (numpy.ndarray).
        :param custom_class: Which class do we want the CAM for (int).
        :return: The high-res CAM (numpy.ndarray).
        """

        # If custom_class is not specified, use the model's prediction for the input image.
        if not custom_class and custom_class != 0:
            prediction = self.predict(image)
        else:
            prediction = custom_class

        Ak = self.Ak_model.predict(image)

        return np.dot(np.squeeze(Ak), self.Wk)[..., prediction]

    def low_res_cam(self, image, custom_class=None):
        """

        :param image:  An image (numpy.ndarray).
        :param custom_class:  Which class do we want the CAM for (int).
        :return:  The low-res CAM (numpy.ndarray).
        """

        # If custom_class is not specified, use the model's prediction for the input image.
        if not custom_class and custom_class != 0:
            prediction = self.predict(image)
        else:
            prediction = custom_class

        Ak_low = self.Ak_model_low.predict(image)

        return np.dot(np.squeeze(Ak_low), self.Wk)[..., prediction]

    def summary(self):
        """
        Print out the AuxiliaryDenseNetFCN model's summary. Useful for debugging.
        """
        self.model.summary()


class HalfModel:

    def __init__(self, weights=opt.half_weight_dir, num_classes=opt.num_classes, im_size=opt.im_size, channels=opt.channels):
        """
        Wrapper for easier predictions and debugging of a HalfDenseNetFCN model.
        :param weights: Path to a file containing the weights of a pre-trained HalfDenseNetFCN.
        :param num_classes: An integer representing the number of classes.
        :param im_size: An integer representing the size of a single dimension each image.
        :param channels: How many channels does the image have (e.g. 1 -> grayscale, 3 -> RGB)
        """
        print('Instantiating a Half Model:')
        print('weights:     {}'.format(weights))
        print('classes:     {}'.format(num_classes))

        image_dims = (im_size, im_size)
        input_shape = image_dims + (channels,)
        print('input shape: {}'.format(input_shape))

        # Main model
        self.model = custom_models.HalfDenseNetFCN(input_shape, num_classes, weights)

        # Ak model
        self.Ak_model = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)

        # Wk array
        self.Wk = self.model.layers[-1].get_weights()[0]

    def generate_cams(self, image, custom_class=None):
        """
        Generate the CAM from a given image, as well as model's output.
        :param img: An image in numpy.ndarray format with a shape of (1, height, width, channels).
        :return: The CAM and the prediction array.
        """

        preds = self.model.predict(image)
        prediction = np.argmax(preds)

        if custom_class or custom_class == 0:
            prediction = custom_class

        Ak_low = self.Ak_model.predict(image)
        cam = np.dot(np.squeeze(Ak_low), self.Wk)[..., prediction]

        return cam, preds

    def predict(self, image):
        """
        Predict the class of an image.
        :param image: An image (numpy.ndarray).
        :return: The model's prediction (int).
        """

        preds = self.model.predict(image)

        return np.argmax(preds)

    def predictN(self, image, n=5):
        """
        Return the top-N predictions for an image.
        :param image: An image (numpy.ndarray).
        :return: The model's prediction (int).
        """

        preds = self.model.predict(image)

        return np.argpartition(preds, -n)[-n:]

    def cam(self, image, custom_class=None):
        """
        Generate the CAM from a given image.
        :param img: An image in numpy.ndarray format with a shape of (1, height, width, channels).
        :return: The CAM (numpy.array).
        """
        if not custom_class and custom_class != 0:
            prediction = self.predict(image)
        else:
            prediction = custom_class

        Ak_low = self.Ak_model.predict(image)

        return np.dot(np.squeeze(Ak_low), self.Wk)[..., prediction]

    def summary(self):
        """
        Print out the HalfDenseNetFCN model's summary. Useful for debugging.
        """
        self.model.summary()
