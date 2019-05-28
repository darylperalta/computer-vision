"""Data generator

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras
import layer_utils
import label_utils
import os
import skimage
from layer_utils import get_gt_data
from skimage.io import imread
from layer_utils import anchor_boxes
import config
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGenerator(Sequence):

    def __init__(self,
                 dictionary,
                 n_classes,
                 params=config.params,
                 input_shape=(300, 480, 3),
                 feature_shapes=[],
                 n_anchors=0,
                 batch_size=32,
                 shuffle=True):
        self.dictionary = dictionary
        self.n_classes = n_classes
        self.keys = np.array(list(self.dictionary.keys()))
        self.params = params
        self.input_shape = input_shape
        self.feature_shapes = feature_shapes
        # self.feature_shape = (1, *feature_shape)
        # print("feature shape: ", self.feature_shape)
        self.index = index
        self.n_anchors = n_anchors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.dictionary) / self.batch_size))


    def __getitem__(self, index):
        # indexes of the batch
        start_index = index * self.batch_size
        end_index = (index+1) * self.batch_size
        keys = self.keys[start_index : end_index]
        x, y = self.__data_generation(keys)
        return x, y


    def on_epoch_end(self):
        # shuffle after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.keys)


    def __data_generation(self, keys):
        data_path = self.params['data_path']
        x = np.empty((self.batch_size, *self.input_shape))
        n_boxes = np.prod(self.feature_shape) // self.n_anchors
        gt_class = np.empty((self.batch_size, n_boxes, self.n_classes))
        gt_offset = np.empty((self.batch_size, n_boxes, 4))
        gt_mask = np.empty((self.batch_size, n_boxes, 4))

        y = []
        for i, key in enumerate(keys):
            image_path = os.path.join(data_path, key)
            image = skimage.img_as_float(imread(image_path))
            x[i] = image
            anchors = anchor_boxes(self.feature_shape,
                                   image.shape,
                                   index=self.index,
                                   is_K_tensor=False)
            anchors = np.reshape(anchors, [-1, 4])
            labels = self.dictionary[key]
            labels = np.array(labels)
            boxes = labels[:,0:-1]
            iou = layer_utils.iou(anchors, boxes)
            ret  = get_gt_data(iou,
                               n_classes=self.n_classes,
                               anchors=anchors,
                               labels=labels)
            gt_class[i], gt_offset[i], gt_mask[i] = ret
            y_ = [gt_class, np.concatenate((gt_offset, gt_mask), axis=-1)]
            y.append(y_)

        return x, y
        # return x, [gt_class, np.concatenate((gt_offset, gt_mask), axis=-1)]
