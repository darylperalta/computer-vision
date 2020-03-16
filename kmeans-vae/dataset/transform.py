'''
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision.transforms as transforms

def mnist_transform():
    c = [24 - i for i in range(0, 10, 2)]
    totensor = transforms.ToTensor()
    crop_center = transforms.CenterCrop(c[0])
    resize = transforms.Resize(c[0])
    norm = transforms.Normalize((0.1307,), (0.3015,))
    test_transform = transforms.Compose([crop_center, totensor, norm])
    crop = transforms.RandomChoice([transforms.RandomCrop(c[1]),
                                   transforms.RandomCrop(c[2]),
                                   transforms.RandomCrop(c[3]),
                                   transforms.RandomCrop(c[4]),
                                   transforms.CenterCrop(c[1]),
                                   transforms.CenterCrop(c[2]),
                                   transforms.CenterCrop(c[3]),
                                   transforms.CenterCrop(c[4])])
    affine = transforms.RandomAffine(20,
                                     shear=(10, 10, 10, 10),
                                     translate=(0.1, 0.1))
    affine_center = transforms.Compose([affine, crop_center])
    crop_resize = transforms.Compose([crop, resize])
    crop_or_affine = transforms.RandomChoice([crop_resize, affine_center])
    train_transform = transforms.Compose([crop_or_affine, totensor, norm])

    return train_transform, test_transform

def mnist_transform_no_augment():
    #norm = transforms.Normalize((0.1307,), (0.3015,))
    totensor = transforms.ToTensor()
    return totensor, totensor
    #crop = transforms.CenterCrop(24)
    #transform = transforms.Compose([crop, totensor])
    #return transform, transform

def generic_transform():
    totensor = transforms.ToTensor()
    return totensor, totensor

def crop_transform(crop_size=28):
    totensor = transforms.ToTensor()
    crop = transforms.CenterCrop(crop_size)
    transform = transforms.Compose([crop, totensor])
    return transform

# Performance in % for pool=3,4 n_units=128,256
# noaugment = [59.3, 53.56, 61.17, 69.19, 64.24, 52.04] = 59.92
# shrink=[70.62, 57.56, 55.46, 69.8, 68.62, 60.83] = 63.81
# crop_resize =[68.71, 64.12, 61.1, 52.65, 52.57, 53.56] = 58.78
# aspect=[71.09, 60.57, 64.41, 64.36, 61.4, 64.86] = 64.44
# choice=[aspect, shrink]= [ 61.12, 65.25, 66.33, 68.93, 59.16, 69.61] = 65.1
# random=[aspect, shrink]= [58.50,59.68, 60.37,70.24,68.55,61.77] = 63.2
def fashionmnist_transform():
    totensor = transforms.ToTensor()
    crop_center = transforms.CenterCrop(28)
    resize = transforms.Resize(28)
    norm = transforms.Normalize((0.2860,), (0.3205,))
    test_transform = transforms.Compose([totensor])
    flip = transforms.RandomHorizontalFlip()
    crop = transforms.RandomChoice([
                                   transforms.RandomCrop(26),
                                   transforms.RandomCrop(24),
                                   transforms.RandomCrop(22),
                                   transforms.RandomCrop(20),
                                   transforms.RandomCrop(18),
                                   transforms.CenterCrop(26),
                                   transforms.CenterCrop(24),
                                   transforms.CenterCrop(22),
                                   transforms.CenterCrop(20),
                                   transforms.CenterCrop(18),
                                   ])
    affine = transforms.RandomAffine(10,
                                     shear=(4, 4, 4, 4),
                                     translate=(0.1, 0.1))
    affine_center = transforms.Compose([affine, flip, crop_center])
    crop_resize = transforms.Compose([crop, flip,  resize])
    shrink = transforms.RandomChoice([
                                     transforms.Compose([transforms.Resize(10),transforms.Pad(9)]),
                                     transforms.Compose([transforms.Resize(12),transforms.Pad(8)]),
                                     transforms.Compose([transforms.Resize(14),transforms.Pad(7)]),
                                     transforms.Compose([transforms.Resize(16),transforms.Pad(6)]),
                                     transforms.Compose([transforms.Resize(18),transforms.Pad(5)]),
                                     transforms.Compose([transforms.Resize(20),transforms.Pad(4)]),
                                     transforms.Compose([transforms.Resize(22),transforms.Pad(3)]),
                                     transforms.Compose([transforms.Resize(24),transforms.Pad(2)]),
                                     ])

    aspect = transforms.RandomChoice([
                    transforms.Compose([transforms.RandomResizedCrop(10, scale=(0.8, 1.0), ratio=(0.5, 1.5)),transforms.Pad(9)]),
                    transforms.Compose([transforms.RandomResizedCrop(12, scale=(0.8, 1.0), ratio=(0.5, 1.5)),transforms.Pad(8)]),
                    transforms.Compose([transforms.RandomResizedCrop(14, scale=(0.8, 1.0), ratio=(0.5, 1.5)),transforms.Pad(7)]),
                    transforms.Compose([transforms.RandomResizedCrop(16, scale=(0.8, 1.0), ratio=(0.5, 1.5)),transforms.Pad(6)]),
                    transforms.Compose([transforms.RandomResizedCrop(18, scale=(0.8, 1.0), ratio=(0.5, 1.5)),transforms.Pad(5)]),
                    transforms.Compose([transforms.RandomResizedCrop(20, scale=(0.8, 1.0), ratio=(0.5, 1.5)),transforms.Pad(4)]),
                    transforms.Compose([transforms.RandomResizedCrop(22, scale=(0.8, 1.0), ratio=(0.5, 1.5)),transforms.Pad(3)]),
                    transforms.Compose([transforms.RandomResizedCrop(24, scale=(0.8, 1.0), ratio=(0.5, 1.5)),transforms.Pad(2)]),
                    ])

    distort = transforms.RandomChoice([aspect, shrink, affine_center])

    train_transform = transforms.Compose([distort, totensor])

    return train_transform, test_transform


def fashionmnist_transform_():

    norm = transforms.Normalize((0.2860,), (0.3205,))
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])
    return train_transform, test_transform

def cifar10_transform_():
    c = [32 - i for i in range(0, 10, 2)]
    totensor = transforms.ToTensor()
    crop_center = transforms.CenterCrop(c[0])
    resize = transforms.Resize(c[0])
    # [0.4914, 0.4822, 0.4465])  [0.2023, 0.1994, 0.2010]
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    test_transform = transforms.Compose([totensor])

    crop = transforms.RandomChoice([transforms.RandomCrop(c[1]),
                                   transforms.RandomCrop(c[2]),
                                   transforms.RandomCrop(c[3]),
                                   transforms.RandomCrop(c[4]),
                                   transforms.CenterCrop(c[1]),
                                   transforms.CenterCrop(c[2]),
                                   transforms.CenterCrop(c[3]),
                                   transforms.CenterCrop(c[4])])
    affine = transforms.RandomAffine(8,
                                     shear=(8, 8, 8, 8),
                                     translate=(0.1, 0.1))
    affine_center = transforms.Compose([affine, crop_center])
    crop_resize = transforms.Compose([crop, resize])
    r_crop = transforms.RandomCrop(32, padding=4)
    r_flip = transforms.RandomHorizontalFlip()
    crop_or_affine = transforms.RandomChoice([crop_resize, affine_center, r_crop, r_flip])

    train_transform = transforms.Compose([get_color_distortion(s=.5), crop_or_affine, totensor])
    #train_transform = transforms.Compose([crop_or_affine, totensor])

    return train_transform, test_transform


def cifar10_transform():
    t0 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(s=0.5),
        transforms.ToTensor(),
    ])
    t1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(s=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_transform = transforms.RandomChoice([t0, t1])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return test_transform, test_transform


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort
