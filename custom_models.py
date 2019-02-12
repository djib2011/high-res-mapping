from densenet import DenseNetFCN
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.models import Model


def HalfDenseNetFCN(input_shape, num_classes, weights=None):
    """
    Model following the 'DenseNetFCN' architecture that ends after the bottleneck (i.e. just before the first
    'transition up' layer). It uses the model provided by keras_contrib and should be used for classification purposes.
    Due to the existence of the GAP layer, this model can be used to produce Class Activation Maps (CAMs).
    Detailed description:
    Following the output of the concatenation layer, a 1x1 convolution is placed to reduce the number of filters to 16;
    the dimensions at this point should be (batch_size, input_height/16, input_width/16, 16). After that a Global
    Average Pooling layer is added to average the 'pixels' of the first two dimensions, leading to a shape of
    (batch_size, 16). Finally, a Fully Connected (Dense) layer is added for classification.
    :param input_shape: Shape of the input image: (height, width, channels).
    :param num_classes: An integer representing the number of classes.
    :param weights: Path to a file containing the weights of a pre-trained HalfDenseNetFCN.
    :return: A compiled keras model, using the Functional keras API.
    """

    # Create the model:
    model_dn = DenseNetFCN(input_shape, nb_dense_block=4, early_transition=True)
    conv_aux = Conv2D(16, (1, 1), kernel_initializer='he_normal', padding='same', name='conv_aux',
                      use_bias=False)(model_dn.layers[121].output)  # The input is the 'concatenation' layer after the
                                                                    # bottleneck (just before the first 'transition up'
                                                                    # layer), named 'concatenate_27'.
    gap = GlobalAveragePooling2D(name='gap')(conv_aux)
    new_top_layer = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4), name='aux_output')(gap)
    model = Model(inputs=model_dn.input, outputs=new_top_layer)

    # If a weights file is provided, load it.
    if weights:
        model.load_weights(weights)

    # Compile the model (it can be recompiled afterwards)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def AuxiliaryDenseNetFCN(input_shape, num_classes, weights=None):
    """
    Model following the DenseNetFCN architecture with an auxiliary exit after the bottleneck (i.e. just before the first
    'transition up' layer) for classification. It uses the model provided by keras_contrib and should be used for
    classification purposes. Due to the existence of the GAP layer, this model can be used to produce Class Activation
    Maps (CAMs). The rest of the model remains the same and is used to provide high-resolution CAMs.
    Detailed description of the modifications to the DenseNetFCN:
    The model has two outputs (i.e. the main output and an auxiliary one).
    - Main output:
    After the final 'transition up' (i.e. Conv2DTrnaspose) layer, a 1x1 convolutional layer is added, which reduces the
    number of filters from 16 to the number of channels of the input image. This is done so that the output image has
    the same shape as the original. This output is trained in an autoencoder-like fashion.
    - Auxiliary output:
    Following the output of the concatenation layer, a 1x1 convolution is placed to reduce the number of filters to 16;
    the dimensions at this point should be (batch_size, input_height/16, input_width/16, 16). After that a Global
    Average Pooling layer is added to average the 'pixels' of the first two dimensions, leading to a shape of
    (batch_size, 16). Finally, a Fully Connected (Dense) layer is added for classification. This output is trained for
    classification.

    :param input_shape: Shape of the input image: (height, width, channels).
    :param num_classes: An integer representing the number of classes.
    :param weights: Path to a file containing the weights of a pre-trained AuxiliaryDenseNetFCN.
    :return: A compiled keras model, using the Functional keras API.
    """

    # Create the model:
    model_dn = DenseNetFCN(input_shape, nb_dense_block=4, early_transition=True)

    # Auxiliary output:
    conv_aux = Conv2D(16, (1, 1), kernel_initializer='he_normal', padding='same', name='conv_aux',
                      use_bias=False)(model_dn.layers[121].output)  # The input is the 'concatenation' layer after the
                                                                    # bottleneck (just before the first 'transition up'
                                                                    # layer), named 'concatenate_27'.
    gap = GlobalAveragePooling2D(name='gap')(conv_aux)
    new_top_layer = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4), name='aux_output')(gap)

    # Main output:
    output = Conv2D(input_shape[2], (1, 1), kernel_initializer='he_normal', padding='same',
                    use_bias=False, name='main_output')(model_dn.layers[-5].output)

    model = Model(inputs=model_dn.input, outputs=[output, new_top_layer])

    # If a weights file is provided, load it.
    if weights:
        model.load_weights(weights)

    # Compile the model (it can be recompiled afterwards). Two loss functions need to be used. The main output's purpose
    # is to be trained to reproduce the original image from the bottleneck (line an autoencoder). The auxiliary output
    # is trained for classification.
    model.compile(optimizer='adadelta',
                  loss={'main_output': 'mean_squared_error', 'aux_output': 'categorical_crossentropy'},
                  metrics={'main_output': ['accuracy'], 'aux_output': ['accuracy']})

    return model

