import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import datetime
import argparse
import matplotlib.pyplot as plt

from dataset import mnist

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        # (channel, filters, kernel_size)
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        # (28,28), (14, 14), (7,7)
        self.fc1 = nn.Linear(64 * 7 * 7, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    log_interval = len(train_loader) // 10
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = F.nll_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        if (i + 1) % log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch,
                  i * len(data),
                  len(train_loader.dataset),
                  100. * i / len(train_loader),
                  loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            test_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

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
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    kwargs = {'num_workers': 4} if use_cuda else {}

    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.1307,), (0.3081,))])
    transform1 = transforms.Compose([transforms.ToTensor()])
    affine = transforms.RandomAffine(5, shear=(10, 10, 10, 10), translate=(0.2,0.2))
    transform2 = transforms.Compose([affine, transforms.ToTensor()])

    x_train = mnist.SiameseMNIST(root='./data',
                                train=True,
                                download=True,
                                transform=transform1,
                                siamese_transform=transform2)

    DataLoader = torch.utils.data.DataLoader
    train_loader = DataLoader(x_train,
                              shuffle=True,
                              batch_size=args.batch_size,
                              **kwargs)

    for i, data in enumerate(train_loader):
        #print(data.shape)
        img = data[0][0][0]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        img = data[0][1][0]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        if i == 4:
            break
        #img = data1[0][0]
        #plt.imshow(img.squeeze(), cmap='gray')
        #plt.show()
       
    exit(0)


    x_train1 = datasets.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transform1)

    x_train2 = datasets.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transform2)

    x_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform1)

    print("Train dataset size:", len(x_train1))
    print("Test dataset size", len(x_test))


    train_loader1 = DataLoader(x_train1,
                               shuffle=False,
                               batch_size=args.batch_size,
                               **kwargs)

    train_loader2 = DataLoader(x_train2,
                               shuffle=False,
                               batch_size=args.batch_size,
                               **kwargs)

    for index, (data1, data2) in enumerate(zip(train_loader1, train_loader2)):
        img = data1[0][0]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        img = data2[0][0]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()

        break


    test_loader = DataLoader(x_test,
                             shuffle=True,
                             batch_size=args.batch_size,
                             **kwargs)


    device = torch.device("cuda" if use_cuda else "cpu")
    encoder = Encoder().to(device)
    if torch.cuda.device_count() > 1:
        print("Available GPUs:", torch.cuda.device_count())
        # model = nn.DataParallel(model)
    print(encoder)
    print(device)
    exit(0)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    start_time = datetime.datetime.now()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
    elapsed_time = datetime.datetime.now() - start_time
    print("Elapsed time (train): %s" % elapsed_time)
    test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
