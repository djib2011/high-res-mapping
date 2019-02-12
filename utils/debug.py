import numpy as np


def get_model_memory_usage(batch_size, model):
    """
    Finds how much memory a keras model of a given batch size requires.
    :param batch_size: The desired batch size (int).
    :param model: An instance of a keras model (keras.models.model).
    :return: A keras model's memory footprint in GB
    """
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def check_weights(model1, model2):
    """
    Check if two models have the same weights.
    :param model1: A keras model (keras.models.model).
    :param model2: Another keras model (keras.models.model).
    """

    print('First model has {} layers.'.format(len(model1.model.layers)))
    print('Second model has {} layers.'.format(len(model2.model.layers)))

    first = True

    for i, layer1 in enumerate(model1.model.layers):

        if 'BatchNormalization' in str(layer1):
            continue

        try:
            layer2 = model2.model.get_layer(layer1.name)
        except ValueError:
            continue

        if not np.array_equal(layer1.get_weights(), layer2.get_weights()):

            if first:
                print('{:^36} | {:^36}'.format('First model', 'Second model'))
                print('{:^5} {:^15} {:^14} | {:^5} {:^15} {:^15}'.format('index', 'name', 'class',
                                                                         'index', 'name', 'class'))
                first = False

            print(' {:<3} {:<15} {:<15} |  {:<3} {:<15} {:<15}'.format(i, layer1.name, layer1.__class__.__name__,
                                                                       model2.model.layers.index(layer2),
                                                                       layer2.name, layer2.__class__.__name__))

