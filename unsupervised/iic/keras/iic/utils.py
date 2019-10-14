import os
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment


def unsupervised_labels(y, y_hat, num_classes, num_clusters):
    """
    :param y: true label
    :param y_hat: concentration parameter
    :param num_classes: number of classes (determined by data)
    :param num_clusters: number of clusters (determined by model)
    :return: classification error rate
    """
    assert num_classes == num_clusters

    # initialize count matrix
    cnt_mtx = np.zeros([num_classes, num_classes])

    # fill in matrix
    for i in range(len(y)):
        cnt_mtx[int(y_hat[i]), int(y[i])] += 1

    # find optimal permutation
    row_ind, col_ind = linear_sum_assignment(-cnt_mtx)

    # compute error
    error = 1 - cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()


    accuracy = (1.0 - error) * 100.
    #print('Classification accuracy= {:.2f}%'.format(accuracy))

    return accuracy
