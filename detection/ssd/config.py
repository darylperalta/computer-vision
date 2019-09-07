"""Project config

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

params = {
        'dataset' : 'real_heads',
        'data_path' : 'dataset/real_heads',
        'train_labels' : 'labels_train.csv',
        'test_labels' : 'labels_test.csv',
        'epoch_offset': 30,
        'aspect_ratios': [1, 2, 0.5],
        'gt_label_iou_thresh' : 0.6,
        'class_thresh' : 0.6,
        'iou_thresh' : 0.2,
        'is_soft_nms' : True,
        'n_classes' : 1,
        'classes' : ["background", "Head"],
        }
