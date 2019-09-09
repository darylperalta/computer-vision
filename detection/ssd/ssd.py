"""SSD class to build, train, eval SSD models

1)  ResNet50 (v2) backbone.
    Train with 6 layers of feature maps.
    Pls adjust batch size depending on your GPU memory.
    For 1060, -b=1. For V100 32GB, -b=4

python3 ssd.py -t -b=4

2)  ResNet50 (v2) backbone.
    Train from a previously saved model:

python3 ssd.py --weights=saved_models/ResNet56v2_4-layer_weights-200.h5 -t -b=4

2)  ResNet50 (v2) backbone.
    Evaluate:

python3 ssd.py -e --weights=saved_models/ResNet56v2_4-layer_weights-200.h5 \
        --image_file=dataset/drinks/0010000.jpg

3) TinyNet backbone

python3 ssd.py -t -b=4 --tiny

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber
# from tensorflow.keras.utils import plot_model

import tensorflow as tf
import layer_utils
import label_utils
import config

import os
import skimage
import numpy as np
import argparse

from skimage.io import imread
from data_generator import DataGenerator
from label_utils import build_label_dictionary
from boxes import show_boxes
from model import build_tinynet, build_ssd
from resnet import build_resnet

def lr_scheduler_resnet(epoch):
    lr = 1e-3
    epoch_offset = config.params['epoch_offset']
    if epoch > (180 - epoch_offset):
        lr *= 0.5e-3
    elif epoch > (140 - epoch_offset):
        lr *= 1e-3
    elif epoch > (100 - epoch_offset):
        lr *= 1e-2
    elif epoch > (60 - epoch_offset):
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def lr_scheduler(epoch):
    lr = 1e-3
    epoch_offset = config.params['epoch_offset']
    if epoch > (200 - epoch_offset):
        lr *= 1e-4
    elif epoch > (180 - epoch_offset):
        lr *= 5e-4
    elif epoch > (160 - epoch_offset):
        lr *= 1e-3
    elif epoch > (140 - epoch_offset):
        lr *= 5e-3
    elif epoch > (120 - epoch_offset):
        lr *= 1e-2
    elif epoch > (100 - epoch_offset):
        lr *= 5e-2
    elif epoch > (80 - epoch_offset):
        lr *= 1e-1
    elif epoch > (60 - epoch_offset):
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr

class SSD():
    def __init__(self,
                 n_layers=6,
                 batch_size=4,
                 epochs=200,
                 workers=8,
                 build_basenet=build_resnet,
                 normalize=False):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.train_generator = None
        self.test_generator = None
        self.normalize = normalize
        self.build_model(build_basenet)


    def build_model(self, build_basenet):
        self.build_dictionary()
        # load 1st image and build base network
        image_path = os.path.join(config.params['data_path'],
                                  self.keys[0])
        image = skimage.img_as_float(imread(image_path))
        self.input_shape = image.shape
        self.basenetwork = build_basenet(self.input_shape,
                                         n_layers=self.n_layers)
        self.basenetwork.summary()
        #plot_model(self.basenetwork,
        #           to_file="basenetwork.png",
        #           show_shapes=True)

        ret = build_ssd(self.input_shape,
                        self.basenetwork,
                        n_layers=self.n_layers,
                        n_classes=self.n_classes)
        # n_anchors = num of anchors per feature point (eg 4)
        # feature_shapes is the feature map shape
        # feature maps of ssd - basis of class and offset predictions
        self.n_anchors, self.feature_shapes, self.ssd = ret
        self.ssd.summary()
        #plot_model(self.ssd,
        #           to_file="ssd.png",
        #           show_shapes=True)


    def build_generator(self):
        # multi-thread train data generator
        gen = DataGenerator(dictionary=self.dictionary,
                            n_classes=self.n_classes,
                            input_shape=self.input_shape,
                            feature_shapes=self.feature_shapes,
                            n_anchors=self.n_anchors,
                            n_layers=self.n_layers,
                            batch_size=self.batch_size,
                            shuffle=True,
                            normalize=self.normalize)
        self.train_generator = gen

        return
        # we skip the test data generator since it is time consuming
        # multi-thread test data generator
        self.test_generator = DataGenerator(dictionary=self.test_dictionary,
                                            n_classes=self.n_classes,
                                            input_shape=self.input_shape,
                                            feature_shapes=self.feature_shapes,
                                            n_anchors=self.n_anchors,
                                            n_layers=self.n_layers,
                                            batch_size=self.batch_size,
                                            shuffle=True)


    def build_dictionary(self):
        # load dataset path
        csv_path = os.path.join(config.params['data_path'],
                                config.params['train_labels'])

        # build dictionary and key
        self.dictionary, self.classes  = build_label_dictionary(csv_path)
        self.n_classes = len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))


    def focal_loss_ce(self, y_true, y_pred):
        # only missing in this FL is y_pred clipping
        weight = (1 - y_pred)
        weight *= weight
        # alpha = 0.25
        weight *= 0.25
        return K.categorical_crossentropy(weight*y_true, y_pred)

    def focal_loss_binary(self, y_true, y_pred):
        gamma = 2.0
        alpha = 0.25

        pt_1 = tf.where(tf.equal(y_true, 1),
                        y_pred,
                        tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0),
                        y_pred,
                        tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        weight = alpha * K.pow(1. - pt_1, gamma)
        fl1 = -K.sum(weight * K.log(pt_1))
        weight = (1 - alpha) * K.pow(pt_0, gamma)
        fl0 = -K.sum(weight * K.log(1. - pt_0))

        return fl1 + fl0


    def focal_loss_categorical(self, y_true, y_pred):
        gamma = 2.0
        alpha = 0.25

        # scale to ensure sum of prob is 1.0
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)

        # calculate focal loss
        weight = alpha * K.pow(1 - y_pred, gamma)
        cross_entropy *= weight

        return K.sum(cross_entropy, axis=-1)

    def mask_offset(self, y_true, y_pred): 
        # 1st 4 are offsets
        offset = y_true[..., 0:4]
        # last 4 are mask
        mask = y_true[..., 4:8]
        # pred is actually duplicated for alignment
        # either we get the 1st or last 4 offset pred
        # and apply the mask
        pred = y_pred[..., 0:4]
        offset *= mask
        pred *= mask
        return offset, pred


    def l1_loss(self, y_true, y_pred):
        offset, pred = self.mask_offset(y_true, y_pred)
        # we can use L1
        return K.mean(K.abs(pred - offset), axis=-1)


    def smooth_l1_loss(self, y_true, y_pred):
        offset, pred = self.mask_offset(y_true, y_pred)
        # Huber loss as approx of smooth L1
        return Huber()(offset, pred)


    def train_model(self,
                    improved_loss=False,
                    smooth_l1=False):
        if self.train_generator is None:
            self.build_generator()

        optimizer = Adam(lr=1e-3)
        print("# classes", self.n_classes)
        if improved_loss:
            print("Focal loss and smooth L1")
            loss = [self.focal_loss_categorical, self.smooth_l1_loss]
        elif smooth_l1:
            print("Smooth L1")
            loss = ['categorical_crossentropy', self.smooth_l1_loss]
        else:
            print("Cross-entropy and L1")
            loss = ['categorical_crossentropy', self.l1_loss]

        self.ssd.compile(optimizer=optimizer, loss=loss)

        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = self.basenetwork.name
        model_name += '_' + str(self.n_layers) + "layer"
        if self.normalize:
            model_name += "-norm"
        if improved_loss:
            model_name += "-improved_loss"
        elif smooth_l1:
            model_name += "-smooth_l1"

        threshold = config.params['gt_label_iou_thresh']
        if threshold < 1.0:
            model_name += "-extra_anchors" 

        model_name += "-" 
        dataset = config.params['dataset']
        model_name += dataset
        model_name += '-{epoch:03d}.h5'

        print("Batch size: ", self.batch_size)
        print("Weights filename: ", model_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # prepare callbacks for model saving
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     save_weights_only=True)
        scheduler = LearningRateScheduler(lr_scheduler)

        callbacks = [checkpoint, scheduler]
        self.ssd.fit_generator(generator=self.train_generator,
                               use_multiprocessing=True,
                               callbacks=callbacks,
                               epochs=self.epochs,
                               workers=self.workers)


    def load_weights(self, weights):
        print("Loading weights: ", weights)
        self.ssd.load_weights(weights)


    # evaluate image based on image (np tensor) or filename
    def evaluate(self, image_file=None, image=None):
        show = False
        if image is None:
            image = skimage.img_as_float(imread(image_file))
            show = True

        image = np.expand_dims(image, axis=0)
        classes, offsets = self.ssd.predict(image)
        # print("Classes shape: ", classes.shape)
        # print("Offsets shape: ", offsets.shape)
        image = np.squeeze(image, axis=0)
        # classes = np.argmax(classes[0], axis=1)
        classes = np.squeeze(classes)
        # classes = np.argmax(classes, axis=1)
        offsets = np.squeeze(offsets)
        class_names, rects, _, _ = show_boxes(image,
                                              classes,
                                              offsets,
                                              self.feature_shapes,
                                              show=show,
                                              normalize=self.normalize)
        return class_names, rects


    def evaluate_test(self):
        # test labels csv path
        csv_path = os.path.join(config.params['data_path'],
                                config.params['test_labels'])
        # test dictionary
        dictionary, _ = build_label_dictionary(csv_path)
        keys = np.array(list(dictionary.keys()))
        # number of gt bbox overlapping predicted bbox
        n_iou = 0
        # sum of IoUs
        s_iou = 0
        # true positive
        tp = 0
        # false positiove
        fp = 0
        for key in keys:
            labels = dictionary[key]
            labels = np.array(labels)
            # 4 boxes coords are 1st four items of labels
            gt_boxes = labels[:, 0:-1]
            # last one is class
            gt_class_ids = labels[:, -1]
            # load image id by key
            image_file = os.path.join(config.params['data_path'], key)
            image = skimage.img_as_float(imread(image_file))
            image = np.expand_dims(image, axis=0)
            # perform prediction
            classes, offsets = self.ssd.predict(image)
            image = np.squeeze(image, axis=0)
            classes = np.squeeze(classes)
            offsets = np.squeeze(offsets)
            # perform nms
            _, _, class_ids, boxes = show_boxes(image,
                                                classes,
                                                offsets,
                                                self.feature_shapes,
                                                show=False,
                                                normalize=self.normalize)

            boxes = np.reshape(np.array(boxes), (-1,4))
            # compute IoUs
            iou = layer_utils.iou(gt_boxes, boxes)
            # skip empty IoUs
            if iou.size ==0:
                continue
            # the class of predicted box w/ max iou
            maxiou_class = np.argmax(iou, axis=1)
            n = iou.shape[0]
            n_iou += n
            s = []
            for j in range(n):
                # list of max ious
                s.append(iou[j, maxiou_class[j]])
                # true positive has the same class and gt
                if gt_class_ids[j] == class_ids[maxiou_class[j]]:
                    tp += 1
                else:
                    fp += 1

            # extra predictions belong to false positives
            fp += abs(len(class_ids) - len(gt_class_ids))
            s = np.sum(s)
            s_iou += s

        print("sum:", s_iou) 
        print("num:", n_iou) 
        print("mIoU:", s_iou/n_iou)
        print("tp:" , tp)
        print("fp:" , fp)
        print("precision:" , tp/(tp+fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Use tinynet as base network"
    parser.add_argument("--tiny",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Num of ssd top feature map layers"
    parser.add_argument("-l",
                        "--layers",
                        default=6,
                        type=int,
                        help=help_)
    help_ = "Batch size"
    parser.add_argument("-b",
                        "--batch_size",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Number of workers thread"
    parser.add_argument("--workers",
                        default=8,
                        type=int,
                        help=help_)
    help_ = "Train model"
    parser.add_argument("-t",
                        "--train",
                        action='store_true',
                        help=help_)
    help_ = "Use focal and smooth l1 loss functions"
    parser.add_argument("--improved_loss",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Use smooth l1 loss function"
    parser.add_argument("-s",
                        "--smooth_l1",
                        default=False,
                        action='store_true', 
                        help=help_)

    help_ = "Use normalize predictions"
    parser.add_argument("-n",
                        "--normalize",
                        default=False,
                        action='store_true', 
                        help=help_)

    help_ = "Load h5 model trained weights"
    parser.add_argument("-w",
                        "--weights",
                        help=help_)

    help_ = "Evaluate model"
    parser.add_argument("-e",
                        "--evaluate",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Image file for evaluation"
    parser.add_argument("--image_file",
                        default=None,
                        help=help_)


    args = parser.parse_args()

    # build ssd using simple cnn backbone
    if args.tiny:
        build_basenet = build_tinynet
    # build ssd using resnet50 backbone
    else:
        build_basenet = build_resnet

    ssd = SSD(n_layers=args.layers,
              batch_size=args.batch_size,
              workers=args.workers,
              build_basenet=build_basenet,
              normalize=args.normalize)

    if args.weights:
        ssd.load_weights(args.weights)
        if args.evaluate:
            if args.image_file is None:
                ssd.evaluate_test()
            else:
                ssd.evaluate(image_file=args.image_file)
            
    if args.train:
        ssd.train_model(improved_loss=args.improved_loss,
                        smooth_l1=args.smooth_l1)
