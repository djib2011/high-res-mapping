from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from pathlib import Path
import numpy as np
from utils import opt


class AnimalsMapper:

    def __init__(self, synset_path, gen=None, test_dir=None):
        """
        Class that maps class indices with names and vice versa.
        :param synset_path: Path under which synset_words.txt cen be found (str).
        :param gen: An instance of a keras ImageDataGenerator (keras.preprocessing.image.ImageDataGenerator).
        :param test_dir: Directory for the test set images.
        """
        if not gen:
            if test_dir:
                gen = ImageDataGenerator().flow_from_directory(test_dir)
            else:
                raise ValueError('You need to specify at least one of the two arguments (gen/test_dir).')

        self.num_classes = gen.num_classes
        self.class_indices = gen.class_indices
        self.classes = {v: k for k, v in self.class_indices.items()}

        self.class_names = {}
        with open(synset_path, 'r') as f:
            for line in f:
                self.class_names[line[:9]] = line[10:].replace('\n', '')

    def name(self, index):
        """
        Map the class index with its respective name.
        :param index: A class index (int).
        :return: The name of the class (str).
        """

        if not isinstance(index, int) or index < 0 or index > self.num_classes:
            raise ValueError('The class index needs to be an integer in [0, {}]'.format(self.num_classes-1))

        return self.class_names[self.classes[index]]

    def index(self, name):
        """
        Map the class name with its respective index.
        :param name: A class name (str).
        :return: The index of the class (int).
        """

        return self.class_indices[name]


def read_image(ind=None, return_class_name=False):
    """
    Read an image, given its index
    :param ind: A valid index (int)
    :param return_class_name:
    :return: A numpy array with a shape of (1, image_height, image_width, channels) (np.ndarray(dtype=np.float32))
    """
    if not ind:
        ind = np.random.randint(0, len(test_images))
    # Open image
    img = Image.open(test_images[ind])
    img = img.resize(image_dims)
    x = np.asarray(img)
    x.setflags(write=True)
    # Insert a new dimensions for the batch_size
    x = np.expand_dims(x, axis=0)
    if len(x.shape) < 4:
        x = np.repeat(x[..., np.newaxis], 3, axis=-1)
    # Identify the class
    wordnet_id = str(test_images[ind]).split('/')[-2]
    if return_class_name:
        y = mapper.class_names[wordnet_id]
    else:
        y = mapper.class_indices[wordnet_id]
    return x[..., :3], y


# Test directory
if not opt.data_dir:
    raise ValueError('You need to specify a data_dir.')
    
test_dir = str(Path(opt.data_dir) / 'test')

# Input dimensions
image_dims = (opt.im_size, opt.im_size)
input_shape = image_dims + (opt.channels,)

# Locate test images
test_images = list(Path(test_dir).rglob('*.JPEG'))

# Select random image
rand_img_ind = np.random.randint(0, len(test_images))

# Create generator
gen = ImageDataGenerator().flow_from_directory(test_dir)

# Class index-to-name mapper
synset_file_path = 'synset_words.txt'
mapper = AnimalsMapper(synset_path=synset_file_path, gen=gen)

# Create variable
num_images = gen.n
num_classes = gen.num_classes
