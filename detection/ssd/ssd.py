"""SSD class to build, train, eval an SSD network

1)  ResNet50 (v2) backbone.
    Train with 6 layers of feature maps.
    Pls adjust batch size depending on your GPU memory.
    For 1060 with 6GB, -b=1. For V100 with 32GB, -b=4

python3 ssd.py -t -b=4

2)  ResNet50 (v2) backbone.
    Train from a previously saved model:

python3 ssd.py --restore-weights=saved_models/ResNet56v2_4-layer_weights-200.h5 -t -b=4

2)  ResNet50 (v2) backbone.
    Evaluate:

python3 ssd.py -e --restore-weights=saved_models/ResNet56v2_4-layer_weights-200.h5 \
        --image-file=dataset/drinks/0010000.jpg

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber

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
from model import build_ssd
from resnet import build_resnet
from loss import focal_loss_categorical, smooth_l1_loss, l1_loss


def lr_scheduler(epoch):
    """Learning rate scheduler - called every epoch"""
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


class SSD:
    """Made of an ssd network model and a dataset generator.
    SSD defines functions to train and validate 
    an ssd network model.

    Arguments:
        args: User-defined configurations

    Attributes:
        ssd (model): SSD network model
        train_generator: multi-threaded data generator for training
        test_generator: multi-threaded data generator for testing

    """
    def __init__(self, args):
        """Copy user-defined configs
        Build backbone and ssd network models
        """
        self.args = args
        seld.ssd = None
        self.train_generator = None
        self.test_generator = None
        self.build_model()


    def build_model(self):
        """Build backbone and SSD networks
        """

        # store in a dictionary the list of image files and labels
        self.build_dictionary()

        # input shape is (480, 640, 3) by default
        self.input_shape = (self.args.height, 
                            self.args.width,
                            self.args.channels)

        # build the backbone network (eg ResNet50)
        # the number of feature layers is equal to n_layers
        # feature layers are inputs to SSD network heads
        # for class and offsets predictions
        self.backbone = self.args.backbone(self.input_shape,
                                           n_layers=self.args.layers)

        # using the backbone, build ssd network
        # outputs of ssd are class and offsets predictions
        anchors, features, ssd = build_ssd(self.input_shape,
                                           self.backbone,
                                           n_layers=self.args.layers,
                                           n_classes=self.n_classes)
        # n_anchors = num of anchors per feature point (eg 4)
        self.n_anchors = anchors
        # feature_shapes is a list of feature map shapes
        # per output layer - used for computing anchor boxes sizes
        self.feature_shapes = features
        # ssd network model
        self.ssd = ssd


    def build_dictionary(self):
        """Read the input image filenames and obj detection labels
        from a csv file and store in a dictionary
        """
        # train dataset path
        csv_path = os.path.join(self.args.data_path,
                                self.args.train_labels)

        # build dictionary: 
        # key=image filaname, value=box coords + class label
        # self.classes is a list of class labels
        self.dictionary, self.classes  = build_label_dictionary(csv_path)
        self.n_classes = len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))


    def print_summary(self):
        """Print network summary for debugging purposes
        """
        from tensorflow.keras.utils import plot_model
        if self.args.summary:
            self.backbone.summary()
            self.ssd.summary()
            plot_model(self.backbone,
                       to_file="backbone.png",
                       show_shapes=True)


    def build_generator(self):
        """Build a multi-thread train data generator
        """
        self.train_generator = DataGenerator(args=self.args,
                                             dictionary=self.dictionary,
                                             n_classes=self.n_classes,
                                             feature_shapes=self.feature_shapes,
                                             n_anchors=self.n_anchors,
                                             shuffle=True)

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


    def train(self):
        """Train the SSD network
        """
        # build the train data generator
        if self.train_generator is None:
            self.build_generator()

        optimizer = Adam(lr=1e-3)
        print("# classes", self.n_classes)
        # choice of loss functions via args
        if self.args.improved_loss:
            print("Focal loss and smooth L1")
            loss = [focal_loss_categorical, smooth_l1_loss]
        elif self.args.smooth_l1:
            print("Smooth L1")
            loss = ['categorical_crossentropy', smooth_l1_loss]
        else:
            print("Cross-entropy and L1")
            loss = ['categorical_crossentropy', l1_loss]

        self.ssd.compile(optimizer=optimizer, loss=loss)

        # model weights are saved for future validation
        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), self.args.save_dir)
        model_name = self.backbone.name
        model_name += '_' + str(self.args.layers) + "layer"
        if self.args.normalize:
            model_name += "-norm"
        if self.args.improved_loss:
            model_name += "-improved_loss"
        elif self.args.smooth_l1:
            model_name += "-smooth_l1"

        if self.args.threshold < 1.0:
            model_name += "-extra_anchors" 

        model_name += "-" 
        model_name += self.args.dataset
        model_name += '-{epoch:03d}.h5'

        print("Batch size: ", self.args.batch_size)
        print("Weights filename: ", model_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # prepare callbacks for model saving
        # and learning rate scheduler
        # learning rate is divided by half every 20epochs
        # after 60th epoch
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     save_weights_only=True)
        scheduler = LearningRateScheduler(lr_scheduler)

        callbacks = [checkpoint, scheduler]
        self.ssd.fit_generator(generator=self.train_generator,
                               use_multiprocessing=True,
                               callbacks=callbacks,
                               epochs=self.args.epochs,
                               workers=self.args.workers)


    def restore_weights(self):
        """Load previously trained model weights
        """
        if self.args.restore_weights:
            save_dir = os.path.join(os.getcwd(), self.args.save_dir)
            filename = os.path.join(save_dir, self.args.restore_weights)
            print("Loading weights: ", filename)
            self.ssd.load_weights(filename)


    def evaluate(self, image_file=None, image=None):
        """Evaluate image based on image (np tensor) or filename
        """
        show = False
        if image is None:
            image = skimage.img_as_float(imread(image_file))
            show = True

        image = np.expand_dims(image, axis=0)
        classes, offsets = self.ssd.predict(image)
        image = np.squeeze(image, axis=0)
        classes = np.squeeze(classes)
        offsets = np.squeeze(offsets)
        class_names, rects, _, _ = show_boxes(args,
                                              image,
                                              classes,
                                              offsets,
                                              self.feature_shapes,
                                              show=show)
        return class_names, rects


    def evaluate_test(self):
        # test labels csv path
        csv_path = os.path.join(self.args.data_path,
                                self.args.test_labels)
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
            image_file = os.path.join(self.args.data_path, key)
            image = skimage.img_as_float(imread(image_file))
            image = np.expand_dims(image, axis=0)
            # perform prediction
            classes, offsets = self.ssd.predict(image)
            image = np.squeeze(image, axis=0)
            classes = np.squeeze(classes)
            offsets = np.squeeze(offsets)
            # perform nms
            _, _, class_ids, boxes = show_boxes(args,
                                                image,
                                                classes,
                                                offsets,
                                                self.feature_shapes,
                                                show=False)

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
    parser = argparse.ArgumentParser(description='SSD for object detection')

    # arguments for model building and training
    help_ = "Number of feature extraction layers of SSD head after backbone"
    parser.add_argument("--layers",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Batch size during training"
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Number of epochs to train"
    parser.add_argument("--epochs",
                        default=200,
                        type=int,
                        help=help_)
    help_ = "Number of data generator worker threads"
    parser.add_argument("--workers",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Labels IoU threshold"
    parser.add_argument("--threshold",
                        default=0.6,
                        type=float,
                        help=help_)
    help_ = "Backbone or base network"
    parser.add_argument("--backbone",
                        default=build_resnet,
                        help=help_)
    help_ = "Train the model"
    parser.add_argument("--train",
                        action='store_true',
                        help=help_)
    help_ = "Print model summary (text and png)"
    parser.add_argument("--summary",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Use focal and smooth L1 loss functions"
    parser.add_argument("--improved-loss",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Use smooth L1 loss function"
    parser.add_argument("--smooth-l1",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Use normalized predictions"
    parser.add_argument("--normalize",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Directory for saving filenames"
    parser.add_argument("--save-dir",
                        default="weights",
                        help=help_)
    help_ = "Dataset name"
    parser.add_argument("--dataset",
                        default="drinks",
                        help=help_)

    # inputs configurations
    help_ = "Input image height"
    parser.add_argument("--height",
                        default=480,
                        type=int,
                        help=help_)
    help_ = "Input image width"
    parser.add_argument("--width",
                        default=640,
                        type=int,
                        help=help_)
    help_ = "Input image channels"
    parser.add_argument("--channels",
                        default=3,
                        type=int,
                        help=help_)

    # dataset configurations
    help_ = "Path to dataset directory"
    parser.add_argument("--data-path",
                        default="dataset/drinks",
                        help=help_)
    help_ = "Train labels csv file name"
    parser.add_argument("--train-labels",
                        default="labels_train.csv",
                        help=help_)
    help_ = "Test labels csv file name"
    parser.add_argument("--test-labels",
                        default="labels_test.csv",
                        help=help_)

    # configurations for evaluation of a trained model
    help_ = "Load h5 model trained weights"
    parser.add_argument("--restore-weights",
                        help=help_)
    help_ = "Evaluate model"
    parser.add_argument("--evaluate",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Image file for evaluation"
    parser.add_argument("--image-file",
                        default=None,
                        help=help_)
    help_ = "Class probability threshold (>= is an object)"
    parser.add_argument("--class-threshold",
                        default=0.8,
                        type=float,
                        help=help_)
    help_ = "NMS IoU threshold"
    parser.add_argument("--iou-threshold",
                        default=0.2,
                        type=float,
                        help=help_)
    help_ = "Use soft NMS or not"
    parser.add_argument("--soft-nms",
                        default=False,
                        action='store_true', 
                        help=help_)

    args = parser.parse_args()
    ssd = SSD(args)

    if args.summary:
        ssd.print_summary()

    if args.restore_weights:
        ssd.restore_weights()
        if args.evaluate:
            if args.image_file is None:
                ssd.evaluate_test()
            else:
                ssd.evaluate(image_file=args.image_file)
            
    if args.train:
        ssd.train()
