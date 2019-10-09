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
from skimage.transform import resize


class DataGenerator(Sequence):
    def __init__(self,
                 dataset=mnist,
                 train=True,
                 batch_size=512,
                 shuffle=True,
                 siamese=False,
                 normalize=False):
        self.dataset = dataset
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.siamese = siamese
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
        if self.siamese:
            x1, x2, y1, y2 = self.__data_generation(start_index, end_index)
            return x1, x2, y1, y2
        else:
            x, y = self.__data_generation(start_index, end_index)
            return x, y

    def _dataset(self):
        if self.train:
            (self.data, self.label), (_, _) = self.dataset.load_data()
        else:
            (_, _), (self.data, self.label) = self.dataset.load_data()
        if self.dataset == mnist:
            self.num_channels = 1
        else:
            self.num_channels = self.data.shape[3]

        # from sparse label to categorical
        self.num_labels = len(np.unique(self.label))
        self.label = to_categorical(self.label)

        # reshape and normalize input images
        image_size = self.data.shape[1]
        self.data = np.reshape(self.data,[-1, image_size, image_size, self.num_channels])
        self.data = self.data.astype('float32') / 255
        self.indexes = [i for i in range(self.data.shape[0])]


    def on_epoch_end(self):
        # shuffle after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def apply_random_noise(self, image, percent=30):
        random = np.random.randint(0, 100)
        if random < percent:
            image = random_noise(image)
        return image


    def random_crop(self, image, target_shape, crop_sizes):
        height, width = image.shape[0], image.shape[1]
        choice = np.random.randint(0, len(crop_sizes))
        dx = dy = crop_sizes[choice]
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        image = image[y:(y+dy), x:(x+dx), :]
        image = resize(image, target_shape)
        return image


    def __data_generation(self, start_index, end_index):

        crop_size = 4
        d = crop_size // 2
        image_size = self.data.shape[1] - crop_size
        x =  self.data[self.indexes[start_index : end_index]]
        y1 =  self.label[self.indexes[start_index : end_index]]
        target_shape = (x.shape[0], image_size, image_size, self.num_channels)
        x1 = np.zeros(target_shape)
        if self.siamese:
            y2 = y1 
            x2 = np.zeros(target_shape)

        for i in range(x1.shape[0]):
            image = x[i]
            x1[i] = image[d: image_size + d, d: image_size + d]
            if self.siamese:
                x2[i] = self.random_crop(image, target_shape[1:], [8, 10, 12])

        if self.siamese:
            return x1, x2, y1, y2
        return x1, y1


if __name__ == '__main__':
    datagen = DataGenerator()

