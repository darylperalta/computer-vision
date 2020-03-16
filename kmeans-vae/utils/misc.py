'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import argparse
from scipy.optimize import linear_sum_assignment

# linear assignment algorithm
def unsupervised_labels(y, yp, n_classes, n_clusters):
    assert n_classes == n_clusters

    # initialize count matrix
    C = np.zeros([n_clusters, n_classes])

    # populate count matrix
    for i in range(len(y)):
        C[int(yp[i]), int(y[i])] += 1

    # optimal permutation using Hungarian Algo
    # the higher the count, the lower the cost
    # so we use -C for linear assignment
    row, col = linear_sum_assignment(-C)

    # compute accuracy
    accuracy = C[row, col].sum() / C.sum()

    return accuracy * 100


def get_device(verbose=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #if torch.cuda.device_count() > 1:
    #    print("Available GPUs:", torch.cuda.device_count())
    #    # model = nn.DataParallel(model)
    if verbose:
        print("Device:", device)
    return device


def init_weights(model, std=0.01):
    if type(model) == nn.Linear:
        nn.init.normal_(model.weight, 0, std)
        model.bias.data.zero_()
    if type(model) == nn.Conv2d:
        nn.init.kaiming_normal_(model.weight)
        model.bias.data.zero_()



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    x_train = dataset(root='./data',
                      train=True,
                      download=True,
                      transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(x_train, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std...')
    for inputs, targets in dataloader:
        channels = inputs.size()[1]
        for i in range(channels):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(x_train))
    std.div_(len(x_train))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def get_args():
    parser = argparse.ArgumentParser(description='MIMax')
    parser.add_argument('--single',
                        default=False,
                        action='store_true',
                        help='Use single branch model (supervised)')
    parser.add_argument('--sgd',
                        default=False,
                        action='store_true',
                        help='Use optimizer')
    parser.add_argument('--supervised',
                        default=False,
                        action='store_true',
                        help='Use double branch model (supervised)')
    parser.add_argument('--div-loss',
                        default="l1",
                        help='MI divergence loss')
    parser.add_argument('--alpha',
                        type=float,
                        default=2,
                        metavar='N',
                        help='Divergence loss alpha weight')
    parser.add_argument('--n-heads',
                        type=int,
                        default=2,
                        metavar='N',
                        help='Number of heads')
    parser.add_argument('--overcluster',
                        type=int,
                        default=0,
                        metavar='N',
                        help='If overcluster, 10x n_classes')
    parser.add_argument('--channels',
                        type=int,
                        default=1,
                        metavar='N',
                        help='Number of channels')
    parser.add_argument('--weight-std',
                        type=float,
                        default=0.5,
                        metavar='N',
                        help='Linear layer initial weights std')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-3,
                        metavar='N',
                        help='Linear layer initial weights std')
    parser.add_argument('--batch-size',
                        type=int,
                        default=512,
                        metavar='N',
                        help='Batch size for training')
    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        metavar='N',
                        help='Number of epochs to train')
    parser.add_argument('--lr',
                        type=float,
                        default=4e-4,
                        metavar='N',
                        help='Learning rate')
    parser.add_argument('--no-augment',
                        default=False,
                        action='store_true',
                        help='Do not use data augmentation')
    parser.add_argument('--vae-latent-dim',
                        type=int,
                        default=0,
                        help='VAE latent dim (enabled when >0)')
    parser.add_argument('--vae-weights',
                        default=None,
                        help='VAE weights')
    parser.add_argument('--kmeans',
                        default=None,
                        help='KMeans pickle file')
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Train model')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='Eval model')
    parser.add_argument('--save-dir',
                        default="weights",
                        help='Folder of model files')
    parser.add_argument('--save-weights',
                        default="classifier.pt",
                        help='Save current model weights on this file (pt)')
    parser.add_argument('--restore-weights',
                        default="classifier.pt",
                        help='Load saved model weights from this file (pt)')
    parser.add_argument('--summary',
                        default=False,
                        action='store_true',
                        help='Print model summary')
    parser.add_argument('--dataset',
                        default="mnist",
                        metavar='N',
                        help='Dataset for training an unsupervised classifier')

    args = parser.parse_args()
    return args
