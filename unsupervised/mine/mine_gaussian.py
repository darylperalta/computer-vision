import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats.contingency import margins

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


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
    print("Computed MI:", mi)
    return mi


class Mine1(nn.Module):
    def __init__(self, hidden_units=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, hidden_units)
        self.fc2 = nn.Linear(1, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2  


class Mine2(nn.Module):
    def __init__(self, input_size=2, hidden_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output = F.elu(self.fc1(x))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output


def train_mine(cov_xy=0.9, mine_model=2, device='cpu'):
    if mine_model==2:
        model = Mine2().to(device)
    else:
        model = Mine2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    plot_loss = []
    n_samples = 10000
    n_epoch = 1000
    cov=[[1, cov_xy], [cov_xy, 1]]
    for epoch in tqdm(range(n_epoch)):
        xy = sample(n_data=n_samples, cov=cov)
        x1 = xy[:,0].reshape(-1,1)
        y1 = xy[:,1].reshape(-1,1)
        xy = sample(joint=False, n_data=n_samples, cov=cov)
        x2 = xy[:,0].reshape(-1,1)
        y2 = xy[:,1].reshape(-1,1)
    
        x1 = torch.from_numpy(x1).to(device)
        y1 = torch.from_numpy(y1).to(device)
        x2 = torch.from_numpy(x2).to(device)
        y2 = torch.from_numpy(y2).to(device)
        x1 = Variable(x1.type(torch.FloatTensor),
                              requires_grad = True)
        y1 = Variable(y1.type(torch.FloatTensor),
                              requires_grad = True)
        x2 = Variable(x2.type(torch.FloatTensor),
                              requires_grad = True)
        y2 = Variable(y2.type(torch.FloatTensor),
                              requires_grad = True)
    
        if True:
            xy = torch.cat((x1, y1), 1)
            pred_xy = model(xy)
            xy = torch.cat((x2, y2), 1)
            pred_x_y = model(xy)
        else:
            pred_xy = model(x1, y1)
            pred_x_y = model(x2, y2)

        loss = torch.mean(pred_xy) \
               - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -loss #maximize
        plot_loss.append(loss.data.numpy())
        model.zero_grad()
        loss.backward()
        optimizer.step()

    x = np.arange(len(plot_loss))
    y = np.array(plot_loss).reshape(-1,)
    sns.set()
    sns.scatterplot(x=x, y=-y)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MI on 2D Gaussian')
    parser.add_argument('--cov_xy',
                        type=float,
                        default=0.5,
                        help='gaussian off diagonal element')
    parser.add_argument('--mine_model',
                        type=int,
                        default=1,
                        help='gaussian off diagonal element')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)
    cov_xy = args.cov_xy
    print("Covariace off diagonal:", cov_xy)
    compute_mi(cov_xy=cov_xy)
    train_mine(cov_xy=cov_xy,
               mine_model=args.mine_model,
               device=device)
