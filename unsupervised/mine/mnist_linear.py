
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import mine

from dataset import mnist


class Model(nn.Module):
    def __init__(self, latent_dim=10, hidden_units=128, encoder=None):
        super(Model, self).__init__()
        self.linear = torch.nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.Linear(hidden_units, 10),
            nn.LogSoftmax(dim=1)
            )
        self.encoder = encoder

    def forward(self, x):
        x = self.linear(x)
        return x


def train(args,
          encoder,
          model,
          device,
          train_loader,
          optimizer,
          epoch):
    encoder.eval()
    model.train()
    log_interval = len(train_loader) // 10
    done_data = 0
    for i, data in enumerate(train_loader):
        x, y = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            latent = encoder(x)
        y_pred = model(latent)
        # y_pred = model(x)

        loss = F.nll_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        done_data += len(data[0])
        if (i + 1) % log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch,
                  done_data,
                  len(train_loader.dataset),
                  100. * done_data / len(train_loader.dataset),
                  loss.item()))


def test(args, encoder, model, device, test_loader):
    model.eval()
    encoder.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data[0].to(device), data[1].to(device)
            latent = encoder(x)
            y_pred = model(latent)
            # y_pred = model(x)
            test_loss += F.nll_loss(y_pred, y).item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss,
          correct,
          len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 4} if use_cuda else {}

    transform = transforms.Compose([transforms.ToTensor()])
    x_train = datasets.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transform)

    x_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform)

    print("Train dataset size:", len(x_train))
    print("Test dataset size", len(x_test))

    DataLoader = torch.utils.data.DataLoader
    train_loader = DataLoader(x_train,
                              shuffle=True,
                              batch_size=args.batch_size,
                              **kwargs)

    test_loader = DataLoader(x_test,
                             shuffle=True,
                             batch_size=args.batch_size,
                             **kwargs)

    latent_dim = 10

    device = torch.device("cuda" if use_cuda else "cpu")
    encoder = mine.Encoder(latent_dim=latent_dim).to(device)
    encoder.load_state_dict(torch.load("weights/mnist_encoder.pt"))
    model = Model(encoder=encoder, latent_dim=latent_dim).to(device)
    if torch.cuda.device_count() > 1:
        print("Available GPUs:", torch.cuda.device_count())
        # model = nn.DataParallel(model)
    print(encoder)
    print(model)
    print(device)
    optimizer = optim.Adam(model.parameters())

    start_time = datetime.datetime.now()
    for epoch in range(1, args.epochs + 1):
        train(args, encoder, model, device, train_loader, optimizer, epoch)
    elapsed_time = datetime.datetime.now() - start_time
    print("Elapsed time (train): %s" % elapsed_time)
    test(args, encoder, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
