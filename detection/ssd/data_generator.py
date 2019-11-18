"""Data generator
This is a scalable and efficient way of reading huge images
as dataset of SSD model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.utils.data_utils import Sequence

import numpy as np
import keras
import layer_utils
import label_utils
import config
import os
import skimage

from layer_utils import get_gt_data
from skimage.io import imread
from layer_utils import anchor_boxes

from skimage.util import random_noise
from skimage import exposure


class DataGenerator(Sequence):
    def __init__(self,
                 dictionary,
                 n_classes,
                 input_shape=(480, 640, 3),
                 feature_shapes=[],
                 n_anchors=4,
                 n_layers=6,
                 batch_size=4,
                 shuffle=True,
                 aug_data=False,
                 normalize=False):
        self.dictionary = dictionary
        self.n_classes = n_classes
        self.keys = np.array(list(self.dictionary.keys()))
        self.input_shape = input_shape
        self.feature_shapes = feature_shapes
        self.n_anchors = n_anchors
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug_data = aug_data
        self.n_layers = len(feature_shapes)
        self.normalize = normalize
        self.on_epoch_end()
        self.get_n_boxes()

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


    def get_n_boxes(self):
        # the number of bounding boxes
        self.n_boxes = 0
        for shape in self.feature_shapes:
            self.n_boxes += np.prod(shape) // self.n_anchors
        return self.n_boxes


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


    def apply_random_exposure_adjust(self, image, percent=30):
        random = np.random.randint(0, 100)
        if random < percent:
            image = exposure.adjust_gamma(image, gamma=0.4, gain=0.9)
            # another exposure algo
            # image = exposure.adjust_log(image)
        return image


    def __data_generation(self, keys):
        data_path = config.params['data_path']
        x = np.zeros((self.batch_size, *self.input_shape))
        dim = (self.batch_size, self.n_boxes, self.n_classes)
        gt_class = np.zeros(dim)
        dim = (self.batch_size, self.n_boxes, 4)
        gt_offset = np.zeros(dim)
        gt_mask = np.zeros(dim)

        for i, key in enumerate(keys):
            # images are assumed to be stored in config data_path
            # key is the image filename 
            image_path = os.path.join(data_path, key)
            image = skimage.img_as_float(imread(image_path))

            # if augment data is enabled
            if self.aug_data:
                image = self.apply_random_noise(image)
                image = self.apply_random_intensity_rescale(image)
                image = self.apply_random_exposure_adjust(image)

            x[i] = image
            labels = self.dictionary[key]
            labels = np.array(labels)
            # 4 boxes coords are 1st four items of labels
            # last item is the obj class
            boxes = labels[:,0:-1]
            for index, feature_shape in enumerate(self.feature_shapes):
                # feature_shape = feature_shape
                # generate anchor boxes
                anchors = anchor_boxes(feature_shape,
                                       image.shape,
                                       index=index,
                                       n_layers=self.n_layers)
                anchors = np.reshape(anchors, [-1, 4])
                # compute IoU of each anchor box 
                # with respect to each bounding boxes
                iou = layer_utils.iou(anchors, boxes)

                # generate ground truth class and offsets
                ret = get_gt_data(iou,
                                  n_classes=self.n_classes,
                                  anchors=anchors,
                                  labels=labels,
                                  normalize=self.normalize)
                gt_cls, gt_off, gt_msk = ret
                if index == 0:
                    cls = np.array(gt_cls)
                    off = np.array(gt_off)
                    msk = np.array(gt_msk)
                else:
                    cls = np.append(cls, gt_cls, axis=0)
                    off = np.append(off, gt_off, axis=0)
                    msk = np.append(msk, gt_msk, axis=0)

            gt_class[i] = cls
            gt_offset[i] = off
            gt_mask[i] = msk


        y = [gt_class, np.concatenate((gt_offset, gt_mask), axis=-1)]

        return x, y
