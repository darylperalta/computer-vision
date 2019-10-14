"""Data generator for original and affine MNIST images

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import numpy as np
import os
import skimage
from skimage.transform import resize


class DataGenerator(Sequence):
    def __init__(self,
                 args,
                 dataset=mnist,
                 train=True,
                 batch_size=512,
                 shuffle=True,
                 siamese=False,
                 normalize=False):
        self.args = args
        self.dataset = dataset
        self.train = train
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.siamese = siamese
        self._dataset()
        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.indexes) / self._batch_size))


    def __getitem__(self, index):
        # indexes of the batch
        start_index = index * self._batch_size
        end_index = (index+1) * self._batch_size
        return self.__data_generation(start_index, end_index)


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
        d = crop_sizes[choice]
        x = height - d
        y = width - d
        center = np.random.randint(0, 2)
        if center:
            dx = dy = d // 2
            image = image[dy:(y + dy), dx:(x + dx), :]
        else:
            dx = np.random.randint(0, d + 1)
            dy = np.random.randint(0, d + 1)
            image = image[dy:(y + dy), dx:(x + dx), :]
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
            x_train = np.concatenate([x1, x2], axis=0)
            y_train = np.concatenate([y1, y2], axis=0)
            y = []
            for i in range(self.args.heads):
                y.append(y_train)
            return x_train, y

        return x1, y1


    @property
    def batch_size(self):
        return self._batch_size


if __name__ == '__main__':
    datagen = DataGenerator()

