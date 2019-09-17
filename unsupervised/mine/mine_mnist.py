import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from dataset import mnist

class Encoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Encoder, self).__init__()
        self.backbone = torch.nn.Sequential(
            # (channel, filters, kernel_size)
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.Softmax()
            )

    def forward(self, x):
        x = self.backbone(x)
        return x


class Mine(nn.Module):
    def __init__(self, latent_dim=10, hidden_units=64):
        super(Mine, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_units)
        self.fc2 = nn.Linear(latent_dim, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)

    def forward(self, x, y):
        x = F.relu(self.fc1(x) + self.fc2(y))
        x = self.fc3(x)
        return x


class Model(nn.Module):
    def __init__(self, latent_dim=10, hidden_units=256):
        super(Model, self).__init__()
        self._backbone = Encoder(latent_dim=latent_dim)
        self.mine = Mine(latent_dim=latent_dim, hidden_units=hidden_units)

    def forward(self, x, y):
        x = self._backbone(x)
        y = self._backbone(y)
        x = self.mine(x, y)
        return x

    @property
    def backbone(self):
        return self._backbone


def train(args,
          model,
          device,
          joint_data,
          marginal1_data,
          marginal2_data,
          optimizer,
          epoch):
    model.train()
    plot_loss = []
    log_interval = len(joint_data.dataset) // joint_data.batch_size
    log_interval //= 5
    x_train = zip(joint_data, marginal1_data, marginal2_data)
    datalen = 0
    for i, data in enumerate(x_train):
        # data[0] is pair of 2 images + 1 label
        # data[1]/[2] is 1 image + 1 label
        xy, x, y = data[0], data[1], data[2]
        #a = np.array(inputs[1])
        #print(a.shape)
        #print(inputs[1])
        #print(len(x))
        #print(len(y))
        #exit(0)
        x1 = xy[0][0].to(device)
        x2 = xy[0][1].to(device)
        pred_xy = model(x1, x2)

        x1 = x[0].to(device)
        x2 = y[0].to(device)
        pred_x_y = model(x1, x2)

        loss = torch.mean(pred_xy) \
               - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -loss #maximize
        plot_loss.append(loss.data.cpu().numpy())
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



def test(args, model, device, test_loader):
    model.eval()
    backbone = model.backbone
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = backbone(inputs).cpu().numpy()
            # outputs = np.argmax(outputs, axis=1)
            labels = labels.cpu().numpy()
            print(outputs, labels)
            # test_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            #pred = outputs.argmax(dim=1, keepdim=True)
            #correct += pred.eq(labels.view_as(pred)).sum().item()


    exit(0)
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
    model = Model().to(device)
    if torch.cuda.device_count() > 1:
        print("Available GPUs:", torch.cuda.device_count())
        # model = nn.DataParallel(model)
    print(model)
    print(device)
    optimizer = optim.Adam(model.parameters())

    start_time = datetime.datetime.now()
    for epoch in tqdm(range(args.epochs)):
        train(args,
              model,
              device,
              joint_data,
              marginal1_data,
              marginal2_data,
              optimizer,
              epoch)
    elapsed_time = datetime.datetime.now() - start_time
    print("Elapsed time (train): %s" % elapsed_time)
    if (args.save_model):
        os.makedirs("weights", exist_ok=True) 
        path = os.path.join("weights", "mnist_encoder.pt")
        torch.save(model.backbone.state_dict(), path)

    test(args, model, device, test_loader)

if __name__ == '__main__':
    main()
