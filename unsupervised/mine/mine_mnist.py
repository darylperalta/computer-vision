
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import mine

from dataset import mnist


def train(args,
          model,
          device,
          joint_data,
          marginal1_data,
          marginal2_data,
          optimizer,
          epoch,
          ma_et = 1.):
    model.train()
    log_interval = len(joint_data.dataset) // joint_data.batch_size
    log_interval //= 5
    x_train = zip(joint_data, marginal1_data, marginal2_data)
    datalen = 0
    for i, data in enumerate(x_train):
        # data[0] is pair of 2 images + 1 label
        # data[1]/[2] is 1 image + 1 label
        xy, x, y = data[0], data[1], data[2]
        jx1 = xy[0][0].to(device)
        jx2 = xy[0][1].to(device)

        # shuffle x2
        #idx = torch.randperm(x2.nelement())
        #x2 = x2.view(-1)[idx].view(x2.size())
        #exit(0)

        mx1 = x[0].to(device)
        mx2 = y[0].to(device)

        mi_lb, t, et = mine.mi(model, jx1, jx2, mx1, mx2)
        loss = -mi_lb
        #loss = torch.mean(pred_xy) \
        #       - torch.log(torch.mean(torch.exp(pred_x_y)))
        #loss = -loss #maximize

        #ma_rate = 0.1
        #ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
        #loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(t)*torch.mean(et))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        datalen += len(x[0])
        if (i + 1) % log_interval == 0 or datalen == len(joint_data.dataset):
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch,
                  datalen,
                  len(joint_data.dataset),
                  100. * (i + 1) / len(joint_data),
                  loss.item()))

    return mi_lb, ma_et


def test(args, model, device, test_loader):
    model.eval()
    backbone = model.backbone
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = backbone(inputs).cpu().numpy()
            labels = labels.cpu().numpy()
            print(outputs, labels)


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
                        default=1024,
                        metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs',
                        type=int,
                        default=40,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    parser.add_argument('--encoder-weights',
                        default=None,
                        help='Encoder parameters')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)

    # kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    kwargs = {'num_workers': 4} if use_cuda else {}

    transform1 = transforms.Compose([transforms.ToTensor()])
    affine = transforms.RandomAffine(5, shear=(10, 10, 10, 10), translate=(0.2,0.2))
    transform2 = transforms.Compose([affine, transforms.ToTensor()])

    joint = mnist.SiameseMNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform1,
                               siamese_transform=transform2)

    DataLoader = torch.utils.data.DataLoader
    joint_data = DataLoader(joint,
                            shuffle=True,
                            batch_size=args.batch_size,
                            **kwargs)

    marginal1 = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform1)
    marginal1_data = DataLoader(marginal1,
                                shuffle=True,
                                batch_size=args.batch_size,
                                **kwargs)

    marginal2 = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform2)
    marginal2_data = DataLoader(marginal2,
                                shuffle=True,
                                batch_size=args.batch_size,
                                **kwargs)


    x_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform1)

    test_loader = DataLoader(x_test,
                             shuffle=True,
                             batch_size=8,
                             **kwargs)


    for i, data in enumerate(joint_data):
        break
        print(len(data))
        img = data[0][0][0]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        img = data[0][1][0]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        if i == 2:
            break
       
    for i, data in enumerate(marginal1_data):
        break
        print(len(data))
        img = data[0][0]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        if i == 2:
            break
       
    for i, data in enumerate(marginal2_data):
        break
        print(len(data))
        img = data[0][0]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        if i == 2:
            break
       
    print("Train dataset size:", len(joint))

    device = torch.device("cuda" if use_cuda else "cpu")
    model = mine.Model().to(device)
    if torch.cuda.device_count() > 1:
        print("Available GPUs:", torch.cuda.device_count())
        # model = nn.DataParallel(model)
    print(model)
    print(device)
    optimizer = optim.Adam(model.parameters())

    start_time = datetime.datetime.now()
    ma_et = 1.
    for epoch in tqdm(range(args.epochs)):
        mi_lb, ma_et = train(args,
              model,
              device,
              joint_data,
              marginal1_data,
              marginal2_data,
              optimizer,
              epoch)
        print(mi_lb.detach().cpu().numpy())
    elapsed_time = datetime.datetime.now() - start_time
    print("Elapsed time (train): %s" % elapsed_time)
    if (args.save_model):
        os.makedirs("weights", exist_ok=True) 
        path = os.path.join("weights", "mnist_encoder.pt")
        torch.save(model.backbone.state_dict(), path)

    # test(args, model, device, test_loader)

if __name__ == '__main__':
    main()
