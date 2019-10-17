"""Build, train and evaluate a MINE Model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras.layers import Input, Dense, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

import numpy as np
import os
import argparse
import vgg

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.contingency import margins


from data_generator import DataGenerator
from utils import unsupervised_labels, center_crop

def mi(model, joint_x1, joint_x2, marginal_x1, marginal_x2):
    t = model.predict(joint_x1, joint_x2)
    et = K.exp(model.predcict(marginal_x1, marginal_x2))
    mi_lb = K.mean(t) - K.log(K.mean(et))
    return mi_lb, t, et


def sample(joint=True,
           mean=[0, 0],
           cov=[[1, 0.9], [0.9, 1]],
           n_data=1000000):
    xy = np.random.multivariate_normal(mean=mean,
                                       cov=cov,
                                       size=n_data)
    if joint:
        return xy 
    y = np.random.multivariate_normal(mean=mean,
                                      cov=cov,
                                      size=n_data)
    x = xy[:,0].reshape(-1,1)
    y = y[:,1].reshape(-1,1)
   
    xy = np.concatenate([x, y], axis=1)
    return xy


def compute_mi(cov_xy=0.9, n_bins=100):
    cov=[[1, cov_xy], [cov_xy, 1]]
    data = sample(cov=cov)
    joint, edge = np.histogramdd(data, bins=n_bins)
    joint /= joint.sum()
    eps = np.finfo(float).eps
    joint[joint<eps] = eps
    x, y = margins(joint)
    xy = x*y
    xy[xy<eps] = eps
    mi = joint*np.log(joint/xy)
    mi = mi.sum()
    print("Computed MI: %0.6f" % mi)
    return mi


class SimpleMINE():
    def __init__(self,
                 args):
        self.args = args
        self._model = None
        self.build_model()


    # build a simple MINE model
    def build_model(self):
        inputs1 = Input(shape=(1))
        inputs2 = Input(shape=(1))
        x1 = Dense(16)(inputs1)
        x2 = Dense(16)(inputs2)
        x = Add()([x1, x2])
        x = Activation('relu')(x)
        outputs = Dense(1)(x)
        inputs = [inputs1, inputs2]
        self._model = Model(inputs, outputs, name='MINE')
        optimizer = Adam(lr=0.01)
        self._model.compile(optimizer=optimizer, loss=self.loss)
        self._model.summary()


    # MI loss 
    def loss(self, y_true, y_pred):
        size = self.args.batch_size
        # lower half is pred for joint dist
        pred_xy = y_pred[0: size, :]
        # upper half is pred for marginal dist
        pred_x_y = y_pred[size: y_pred.shape[0], :]
        loss = K.mean(pred_xy) \
               - K.log(K.mean(K.exp(pred_x_y)))
        return -loss


    # Train MINE to estimate MI between X and Y of a 2D Gaussian
    def train(self):
        plot_loss = []
        cov=[[1, self.args.cov_xy], [self.args.cov_xy, 1]]
        loss = 0.
        for epoch in range(self.args.epochs):
            xy = sample(n_data=self.args.batch_size,
                        cov=cov)
            x1 = xy[:,0].reshape(-1,1)
            y1 = xy[:,1].reshape(-1,1)
            xy = sample(joint=False,
                        n_data=self.args.batch_size,
                        cov=cov)
            x2 = xy[:,0].reshape(-1,1)
            y2 = xy[:,1].reshape(-1,1)
    
            x =  np.concatenate((x1, x2))
            y =  np.concatenate((y1, y2))
            loss_item = self._model.train_on_batch([x, y],
                                                   np.zeros(x.shape))
            loss += loss_item
            plot_loss.append(loss_item)
            if (epoch + 1) % 100 == 0:
                print("Epoch %d MINE MI: %0.6f" % ((epoch+1), -loss/100))
                loss = 0.



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MI on 2D Gaussian')
    parser.add_argument('--cov_xy',
                        type=float,
                        default=0.5,
                        help='Gaussian off diagonal element')
    parser.add_argument('--save-dir',
                       default="weights",
                       help='Folder for storing model weights (h5)')
    parser.add_argument('--save-weights',
                       default=None,
                       help='Folder for storing model weights (h5)')
    parser.add_argument('--dataset',
                       default=mnist,
                       help='Dataset to use')
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='Number of epochs to train')
    parser.add_argument('--batch-size',
                        type=int,
                        default=10000,
                        metavar='N',
                        help='Train batch size')
    args = parser.parse_args()
    print("Covariace off diagonal:", args.cov_xy)
    simple_mine = SimpleMINE(args)
    simple_mine.train()
    compute_mi(cov_xy=args.cov_xy)
