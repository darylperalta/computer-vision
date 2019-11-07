'''Demonstrates MAML by learning to predict a Gaussian dist mean

python3 maml.py

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import argparse


class SimpleMAML(nn.Module):
    def __init__(self, args, device):
        super(SimpleMAML, self).__init__()
        self.args = args
        self.device = device
        hidden_units = 256
        self.fc1 = nn.Linear(self.args.n_samples, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.fc1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)
        self.fc3.bias.data.fill_(0.01)
        self.sample_means()
        self.meta_optim = torch.optim.Adam(self.parameters(), lr=self.args.meta_lr)
        self.update_optim = torch.optim.Adam(self.parameters(), lr=self.args.update_lr)


    def sample_means(self):
        # n_tasks 1D Gaussians form distribution of tasks
        # for meta-learning
        # train means are from 0.5 to 1.0
        self.means = np.random.uniform(0.5, 1, self.args.n_tasks)
        # held out 1D Gaussian is for meta testing
        # test mean is outside train means
        self.held_out = np.random.uniform(0.0, 0.2, 1)


    def forward(self, x):
        # 3-layer MLP for mean prediction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


    def sample_input(self, mean, batch_size, n_samples):
        # sample input data fr gaussian dist
        samples = np.random.normal(mean, size=(batch_size, n_samples))
        samples = np.reshape(samples, (batch_size, n_samples))
        samples = torch.from_numpy(samples)
        samples = samples.type(torch.FloatTensor)
        samples = samples.to(self.device)
        return samples


    def sample_target(self, mean, batch_size):
        # sample of output data (mean)
        mean = np.repeat(mean, batch_size)
        mean = np.reshape(mean, (batch_size, -1))
        mean = torch.from_numpy(mean)
        mean = mean.type(torch.FloatTensor)
        mean = mean.to(self.device)
        return mean


    def train(self, test=False):
        # number of task samples fr distribution of tasks
        n_task_samples = self.args.n_tasks #// 2 
        # number of data points as model input
        n_samples = self.args.n_samples
        # initial theta before update
        theta = []
        losses = []
        scalar_losses = []
        mse = nn.MSELoss()
        if test:
            data = self.held_out
            indexes = [0]
            #n_epochs = self.args.n_epochs
            n_epochs = 1
            batch_size = self.args.batch_size
        else:
            data = self.means
            n_epochs = self.args.n_epochs
            batch_size = self.args.batch_size

        for epoch in range(n_epochs):
            self.params = list(self.parameters())
            device = self.params[0].device
            # save the theta before computing phi
            theta = [p.clone().detach() for p in self.params]
            total_loss = 0
            phis = []
            if not test:
                # during meta-training, sample fr distribution of tasks
                indexes = np.random.randint(0, self.args.n_tasks, n_task_samples)
            # meta-learning
            for index in indexes:
                # train per sampled task
                for p, x in zip(self.params, theta):
                    p.data.copy_(x)
                mean = data[index]
                samples = self.sample_input(mean, batch_size, n_samples)
                y = self(samples)
                mean = self.sample_target(mean, batch_size)

                loss = mse(y, mean)
                #losses.append(loss)
                #scalar_losses.append(loss.item())

                # compute phi
                self.meta_optim.zero_grad()
                loss.backward()
                self.meta_optim.step()

                # save phi (theta prime)
                phi = [p.clone().detach() for p in list(self.parameters())]
                phis.append(phi)

                y = self(samples)
                if not test:
                    print(mean[0], y[0])

            # adaptation
            for phi, index in zip(phis, indexes):
                for p, x in zip(self.params, phi):
                    p.data.copy_(x)

                mean = data[index]
                samples = self.sample_input(mean, batch_size, n_samples)
                y = self(samples)
                mean = self.sample_target(mean, batch_size)
                loss = mse(y, mean)
                total_loss += loss

            # apply gradient on theta
            for p, x in zip(self.params, theta):
                p.data.copy_(x)
            loss = total_loss / self.args.n_tasks
            self.update_optim.zero_grad()
            loss.backward()
            self.update_optim.step()
            # observe prediction after meta-testing
            if test:
                y = self(samples)
                mean = mean[0].data.cpu()
                y = y[0].data.cpu()
                print("Meta-test:") 
                print("Ground truth: %0.6f, Prediction: %0.6f: " % (mean, y))
                #print("")
                #print("Meta-train tasks: ", self.means)


    # observe prediction before meta-testing
    def eval(self):
        with torch.no_grad():
            n_samples = self.args.n_samples
            mean = self.held_out
            samples = self.sample_input(mean, 1, n_samples)
            y = self(samples)
            y = y[0].data.cpu()
            pre_eval = "Without meta-testing: \n"
            pre_eval += "Ground truth: %0.6f, Prediction: %0.6f: " % (mean, y)
            return pre_eval
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on 1D Gaussian')
    parser.add_argument('--n-samples',
                        type=int,
                        default=2000,
                        help='Number of samples per task')
    parser.add_argument('--n-tasks',
                        type=int,
                        default=10,
                        help='Number of tasks (# of 1D Gaussians)')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='Batch size (k-shot = batch_size)')
    parser.add_argument('--update-lr',
                        type=float,
                        default=1e-3,
                        help='update learning rate')
    parser.add_argument('--meta-lr',
                        type=float,
                        default=5e-4,
                        help='meta learning rate')

    #torch.manual_seed(9007965514248702052)
    torch.manual_seed(104029687756708413)
    seed = torch.seed()
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Training on ", device)
    simple_maml = SimpleMAML(args, device).to(device)
    simple_maml.train()
    pre_eval = simple_maml.eval()
    simple_maml.train(test=True)
    print(pre_eval)
    print(seed)
