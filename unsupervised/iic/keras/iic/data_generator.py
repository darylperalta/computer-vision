"""Data generator for original and affine MNIST images

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import numpy as np
import os
import skimage
from skimage.io import imread
from skimage.util import random_noise
from skimage import exposure


class DataGenerator(Sequence):
    def __init__(self,
                 dataset=mnist,
                 batch_size=512,
                 shuffle=True,
                 normalize=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._dataset()
        self.on_epoch_end()
        # self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))


    def __getitem__(self, index):
        # indexes of the batch
        start_index = index * self.batch_size
        end_index = (index+1) * self.batch_size
        x1, x2, y1, y2 = self.__data_generation(start_index, end_index)
        return x1, x2, y1, y2

    def _dataset(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset.load_data()
        if self.dataset == mnist:
            num_channels = 1
        else:
            num_channels = self.x_train.shape[3]

        # from sparse label to categorical
        self.num_labels = len(np.unique(self.y_train))
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        # reshape and normalize input images
        image_size = self.x_train.shape[1]
        self.x_train = np.reshape(self.x_train,[-1, image_size, image_size, num_channels])
        self.x_test = np.reshape(self.x_test,[-1, image_size, image_size, num_channels])
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255
        self.indexes = [i for i in range(self.x_train.shape[0])]
        print(self.x_train.shape)


    def on_epoch_end(self):
        # shuffle after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def apply_random_noise(self, image, percent=30):
        random = np.random.randint(0, 100)
        if random < percent:
            image = random_noise(image)
        return image


    def apply_random_intensity_rescale(self, image, percent=30):
        random = np.random.randint(0, 100)
        if random < percent:
            v_min, v_max = np.percentile(image, (0.2, 99.8))
            image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        return image


    def __data_generation(self, start_index, end_index):

        x1 =  self.x_train[self.indexes[start_index, end_index]]
        y1 =  self.y_train[self.indexes[start_index, end_index]]

        return x1, y1, x2. y2


if __name__ == '__main__':
    datagen = DataGenerator()

