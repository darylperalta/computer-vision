"""
Train:
python3 vae.py --train --epochs=60 --latent-dim=16 --dataset=mnist
or
python3 vae.py --train --epochs=60 --latent-dim=16 --dataset=fashionmnist

Eval:
python3 vae.py --restore-weights=mnist-vae-16-beta-1.pt --latent-dim=16 --kmeans=mnist-kmeans-16-beta-1.pt --dataset=nmnist
or
python3 vae.py --restore-weights=fashionmnist-vae-32-beta-1.pt --latent-dim=32 --kmeans=fashionmnist-kmeans-32-beta-1.pt --dataset=fashionmnist

"""



from __future__ import print_function
import argparse
import torch
import os
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dataset.transform import crop_transform
from torch.utils.data import DataLoader

from loss import AverageMeter
from utils.ui import progress_bar
from utils.misc import get_device, unsupervised_labels

import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs',
                        type=int, 
                        default=20, 
                        metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--seed', 
                        type=int, 
                        default=1, 
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--crop-size',
                        type=int,
                        default=28, 
                        metavar='N',
                        help='Crop size')
    parser.add_argument('--latent-dim',
                        type=int,
                        default=32, 
                        metavar='N',
                        help='Latent dim')
    parser.add_argument('--beta',
                        type=float, 
                        default=1, 
                        metavar='N',
                        help='Beta in BetaVAE')
    parser.add_argument('--save-dir',
                        default="weights",
                        help='Folder of model files')
    parser.add_argument('--restore-weights',
                        default=None,
                        help='Restore model weights on this file (pt)')
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Train model')
    parser.add_argument('--tsne',
                        default=False,
                        action='store_true',
                        help='Use TSNE')
    parser.add_argument('--kmeans',
                        default=None,
                        help='KMeans pickle file')
    parser.add_argument('--dataset',
                        default=None,
                        help='Dataset to use')
    return parser.parse_args()


def get_dataloader(args, dataset):
    device = get_device()
    cuda = "cuda" in str(device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             **kwargs)
    return data_loader


class Encoder(nn.Module):
    def __init__(self, latent_dim=16, crop_size=28):
        super(Encoder, self).__init__()
       
        mid_dim = 128 
        down_dim = crop_size // 4
        # in=24, out=12
        self.conv2d1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)

        # in=12, out=6
        self.conv2d2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        self.feature = nn.Linear(down_dim*down_dim*64, mid_dim)
        self.mu = nn.Linear(mid_dim, latent_dim)
        self.logvar = nn.Linear(mid_dim, latent_dim)


    def forward(self, x):
        x = F.relu(self.conv2d1(x), inplace=True)
        x = F.relu(self.conv2d2(x), inplace=True)
        x = self.flatten(x)
        x = F.relu(self.feature(x), inplace=True)
        return self.mu(x), self.logvar(x)
        

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, crop_size=28):
        super(Decoder, self).__init__()

        self.down_dim = crop_size // 4
       
        self.sample = nn.Linear(latent_dim, self.down_dim*self.down_dim*64)
        self.conv2d1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=0)
        self.conv2d3 = nn.ConvTranspose2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.relu(self.sample(x), inplace=True)
        x = x.view(-1, 64, self.down_dim, self.down_dim)
        x = F.relu(self.conv2d1(x), inplace=True)
        x = F.relu(self.conv2d2(x), inplace=True)
        x = self.sigmoid(self.conv2d3(x))
        return x
        

class VAE(nn.Module):
    def __init__(self, latent_dim=16, crop_size=28):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dim=latent_dim, crop_size=crop_size)
        self.decoder = Decoder(latent_dim=latent_dim, crop_size=crop_size)
        self.init_weights()


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


    def init_weights(self, std=0.1):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0, std)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.bias.data.zero_()
            elif type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(args, xp, x, mu, logvar, beta=1.0):
    x = x.view(-1, args.crop_size**2)
    xp = xp.view(-1, args.crop_size**2)
    recon = F.binary_cross_entropy(xp, x, reduction='sum')
    #MSE = F.mse_loss(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta*kl, recon, kl


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    recon_losses = AverageMeter()
    kl_losses = AverageMeter()
    device = get_device()
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        xp, mu, logvar = model(x)
        loss, recon, kl = loss_function(args, xp, x, mu, logvar, args.beta)
        loss.backward()
        recon_losses.update(recon.float().item()/x.size(0))
        kl_losses.update(kl.float().item()/x.size(0))
        optimizer.step()
        progress_bar(i,
                     len(train_loader), 
                     'Epoch: %d | Recon Loss: %.6f | KL Loss: %.6f' % (epoch, recon_losses.avg, kl_losses.avg))


def test(args, model, test_loader, epoch=0):
    model.eval()
    recon_losses = AverageMeter()
    kl_losses = AverageMeter()
    device = get_device()
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            mu, logvar = model.encoder(x)

            noise = torch.randn_like(mu) * 2*logvar.exp().pow(0.5)
            noised_mu = mu + noise
            z = model.reparameterize(noised_mu, logvar)
            noised_pos = model.decoder(z)

            noise = torch.randn_like(mu) * 2*logvar.exp().pow(0.5)
            noised_mu = mu + noise
            z = model.reparameterize(noised_mu, logvar)
            noised_neg = model.decoder(z)

            xp, mu, logvar = model(x)
            loss, recon, kl = loss_function(args, xp, x, mu, logvar, args.beta)
            recon_losses.update(recon.float().item()/x.size(0))
            kl_losses.update(kl.float().item()/x.size(0))

            progress_bar(i,
                         len(test_loader), 
                         'Epoch: %d | Recon Loss: %.6f | KL Loss: %.6f' % (epoch, recon_losses.avg, kl_losses.avg))

            if i == 0:
                n = min(x.size(0), 64)
                folder = "results"
                os.makedirs(folder, exist_ok=True)
                img = torch.cat([x[:n],
                                xp.view(args.batch_size, 1, args.crop_size, args.crop_size)[:n]])
                filename = folder + '/recon.png'
                save_image(img.cpu(), filename, nrow=n)
                img = torch.cat([x[:n],
                                noised_pos.view(args.batch_size, 1, args.crop_size, args.crop_size)[:n]])
                filename = folder + '/noised_pos.png'
                save_image(img.cpu(), filename, nrow=n)
                img = torch.cat([x[:n],
                                noised_neg.view(args.batch_size, 1, args.crop_size, args.crop_size)[:n]])
                filename = folder + '/noised_neg.png'
                save_image(img.cpu(), filename, nrow=n)


def tsne(args, model, data_loader, tsne=False):
    model.eval()
    device = get_device()
    mus = np.array([])
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            mu, logvar = model.encoder(x)
            if len(mus) == 0:
                mus = mu.cpu()
            else:
                mus = np.concatenate((mus, mu.cpu()), axis=0)
            progress_bar(i, len(data_loader))

    if tsne:
        mus_tsne = TSNE(n_components=2).fit_transform(mus)
        print(mus_tsne.shape)
        return mus, mus_tsne

    print(mus.shape)
    return mus, _


def plot_tsne(mus_tsne, filename=None, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mus_tsne)
    x = mus_tsne[:,0]
    y = mus_tsne[:,1]
    data = pd.DataFrame()
    data['x'] = x
    data['y'] = y
    data['label'] = kmeans.labels_
    #current_palette = sns.color_palette()
    #sns.palplot(current_palette)
    ax = sns.scatterplot(
                    x="x", y="y",
                    hue="label",
                    data=data,
                    palette=sns.color_palette("hls", n_clusters),
                    alpha=0.3
                    )
    fig = ax.get_figure()
    fig.savefig('results/tsne.png')


def to_categorical(y, n_clusters=10):
    return torch.eye(n_clusters, dtype=torch.long)[y]

def plot_centroid(args, model, data_loader, mus, filename=None, n_clusters=10):
    if filename is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mus)
        filename = args.dataset + "-kmeans-" + str(args.latent_dim) + "-beta-" + str(args.beta) + ".pt" 
        path = os.path.join(args.save_dir, filename)
        print("Saving kmeans: %s\n" % (path))
        pickle.dump(kmeans, open(path, "wb"))
    else:
        #kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        path = os.path.join(args.save_dir, filename)
        print("Loading kmeans: %s\n" % (path))
        kmeans = pickle.load(open(path, "rb"))
    centers = kmeans.cluster_centers_
    model.eval()
    device = get_device()
    with torch.no_grad():
        ytrue = torch.tensor(np.array([]), dtype=torch.long)
        ypred = torch.tensor(np.array([]), dtype=torch.long)
        ytrue = ytrue.to(device)
        ypred = ypred.to(device)
        for i, (x, target) in enumerate(data_loader):
            x = x.to(device)
            target = target.to(device)

            mu, logvar = model.encoder(x)
            labels = kmeans.predict(mu.cpu())

            #y = to_categorical(torch.from_numpy(labels).to(device, dtype=torch.long), n_clusters=n_clusters)
            y = torch.from_numpy(labels).to(device, dtype=torch.long)
            ytrue = torch.cat((ytrue, target), axis=0)
            ypred = torch.cat((ypred, y.to(device)), axis=0)
            accuracy = unsupervised_labels(ytrue.cpu(), ypred.cpu(), n_clusters, n_clusters)
            method = "Unsupervised (VAE)"
            progress_bar(i,
                         len(data_loader), 
                         '%s | Acc: %0.2f%%'
                         % (method, accuracy))

            if i==0:
                centroid_mu = centers[labels]
                centroid_mu = torch.tensor(centroid_mu).to(device)
                z = model.reparameterize(centroid_mu, logvar)
                centroid = model.decoder(z)

                n = min(x.size(0), 64)
                folder = "results"
                os.makedirs(folder, exist_ok=True)

                xp, mu, logvar = model(x)
                img = torch.cat([x[:n],
                                xp.view(args.batch_size, 1, args.crop_size, args.crop_size)[:n]])
                filename = folder + '/recon.png'
                save_image(img.cpu(), filename, nrow=n)

                img = torch.cat([x[:n],
                                centroid.view(args.batch_size, 1, args.crop_size, args.crop_size)[:n]])
                filename = folder + '/centroid.png'
                save_image(img.cpu(), filename, nrow=n)


def vae():
    args = get_args()
    device = get_device()
    #torch.manual_seed(args.seed)
    model = VAE(args.latent_dim, args.crop_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    if args.dataset=="mnist":
        dataset = datasets.MNIST('./data', 
                                 train=True, 
                                 download=True,
                                 transform=crop_transform(args.crop_size))
        train_loader = get_dataloader(args, dataset)

        dataset = datasets.MNIST('./data', 
                                 train=False, 
                                 download=True,
                                 transform=crop_transform(args.crop_size))
        test_loader = get_dataloader(args, dataset)
    elif args.dataset == "fashionmnist":
        dataset = datasets.FashionMNIST('./data', 
                                        train=True, 
                                        download=True,
                                        transform=crop_transform(args.crop_size))
        train_loader = get_dataloader(args, dataset)

        dataset = datasets.FashionMNIST('./data', 
                                        train=False, 
                                        download=True,
                                        transform=crop_transform(args.crop_size))
        test_loader = get_dataloader(args, dataset)
    else:
        exit(0)
    print("Dataset: ", args.dataset)

    folder = args.save_dir
    os.makedirs(folder, exist_ok=True)

    compute_kmeans = False
    if args.train:
        for epoch in range(1, args.epochs + 1):
            print("Epoch %d" % (epoch))
            train(args, model, train_loader, optimizer, epoch)
            
        test(args, model, test_loader, epoch)
        filename = args.dataset + "-vae-" + str(args.latent_dim) + "-beta-" + str(args.beta) + ".pt" 
        path = os.path.join(folder, filename)
        print("Saving weights: %s\n" % (path))
        torch.save(model.state_dict(), path)
        compute_kmeans = True



    if args.restore_weights is not None:
        path = os.path.join(folder, args.restore_weights)
        print("Loading weights... ", path)
        model.load_state_dict(torch.load(path))
        test(args, model, test_loader)
        compute_kmeans = True

    if compute_kmeans:
        mus, mus_tsne = tsne(args, model, train_loader, tsne=args.tsne)
        if args.tsne:
            plot_tsne(mus_tsne)
        if args.kmeans is None:
            plot_centroid(args, model, train_loader, mus, args.kmeans)
        else:
            plot_centroid(args, model, test_loader, mus, args.kmeans)


if __name__ == "__main__":
    vae()
