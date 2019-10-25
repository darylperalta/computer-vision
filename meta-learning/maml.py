'''Demonstrates MAML by learning to predict gaussian dist mean

python3 maml.py --n-samples=1000 --n-epochs=200 --batch-size=128

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
    def __init__(self, args):
        super(SimpleMAML, self).__init__()
        self.args = args
        self.n_tasks = self.args.n_tasks
        hidden_units = 128
        self.fc1 = nn.Linear(self.args.n_samples, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.sample_means()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)


    def sample_means(self):
        self.means = np.random.uniform(0.5, 1, self.args.n_tasks) #/ 100.
        self.held_out = np.random.uniform(0.0, 0.2, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


    def train(self, test=False):
        n_task_samples = self.args.n_tasks // 2 
        n_samples = self.args.n_samples
        initial_values = []
        losses = []
        scalar_losses = []
        mse = nn.MSELoss()
        lr = 1e-3
        if test:
            n_epochs = 1
            data = self.held_out
            indexes = [0]
            batch_size = 1
        else:
            n_epochs = self.args.n_epochs
            data = self.means
            batch_size = self.args.batch_size

        for epoch in range(n_epochs):
            self.params = list(self.parameters())
            device = self.params[0].device
            initial_values = [p.clone().detach() for p in self.params]
            total_loss = 0
            final_values = []
            if not test:
                indexes = np.random.randint(0, self.args.n_tasks, n_task_samples)
            for index in indexes:
                for p, x in zip(self.params, initial_values):
                    p.data.copy_(x)
                mean = data[index]
                samples = np.random.normal(mean, size=(batch_size, n_samples))
                samples = np.reshape(samples, (batch_size, n_samples))
                samples = torch.from_numpy(samples)
                samples = samples.type(torch.FloatTensor)
                samples = samples.to(device)
                y = self(samples)
                mean = np.repeat(mean, batch_size)
                mean = np.reshape(mean, (batch_size, -1))
                mean = torch.from_numpy(mean)
                mean = mean.type(torch.FloatTensor)
                mean = mean.to(device)

                loss = mse(y, mean)
                #losses.append(loss)
                #scalar_losses.append(loss.item())
                #updated = []
                #grads = torch.autograd.grad(loss, self.params,  create_graph=True, retain_graph=True)
                #for grad, param in zip(grads, self.params):
                #    x = param - lr * grad
                #    updated.append(x)
                
                #final_values.append(updated)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                updated = [p.clone().detach() for p in list(self.parameters())]
                final_values.append(updated)

                y = self(samples)
                print(mean[0], y[0])


            for updated, index in zip(final_values, indexes):
                for p, x in zip(self.params, updated):
                    p.data.copy_(x)

                mean = data[index]
                samples = np.random.normal(mean, size=(batch_size, n_samples))
                samples = np.reshape(samples, (batch_size, n_samples))
                samples = torch.from_numpy(samples)
                samples = samples.type(torch.FloatTensor)
                samples = samples.to(device)
                y = self(samples)
                mean = np.repeat(mean, batch_size)
                mean = np.reshape(mean, (batch_size, -1))
                mean = torch.from_numpy(mean)
                mean = mean.type(torch.FloatTensor)
                mean = mean.to(device)

                loss = mse(y, mean)
                total_loss += loss

            for p, x in zip(self.params, initial_values):
                p.data.copy_(x)
            loss = total_loss / self.args.n_tasks
            #print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if test:
                y = self(samples)
                print("Test: ", mean[0], y[0])


    def eval(self):
        with torch.no_grad():
            n_samples = self.args.n_samples
            mean = self.held_out
            samples = np.random.normal(mean, size=(1, n_samples))
            samples = np.random.normal(mean, size=(1, n_samples))
            samples = np.reshape(samples, (1, n_samples))
            samples = torch.from_numpy(samples)
            samples = samples.type(torch.FloatTensor)
            samples = samples.to(device)
            y = self(samples)
            print("Pre-eval: ", mean, y)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on 1D Gaussian')
    parser.add_argument('--n-samples',
                        type=int,
                        default=10,
                        help='Number of samples per tasks')
    parser.add_argument('--n-tasks',
                        type=int,
                        default=10,
                        help='Number of tasks')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        help='Batch size')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    simple_maml = SimpleMAML(args).to(device)
    simple_maml.train()
    simple_maml.eval()
    simple_maml.train(test=True)

