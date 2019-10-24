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


    def sample_means(self):
        self.means = np.random.uniform(0, 1, self.args.n_tasks) #/ 100.
        self.held_out = np.random.uniform(0, 1, 1)


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
        lr = 1e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        if test:
           n_epochs = 1
           data = self.held_out
           indexes = [i for i in range(1)]
        else:
           n_epochs = self.args.n_epochs
           data = self.means
           indexes = [i for i in range(self.args.n_tasks)]

        for epoch in range(n_epochs):
            self.params = list(self.parameters())
            device = self.params[0].device
            initial_values = [p.clone().detach() for p in self.params]
            total_loss = 0
            final_values = []
            #indexes = np.random.randint(0, self.args.n_tasks, self.args.n_tasks)
            for index in indexes:
                mean = data[index]
                samples = np.random.normal(mean, size=(self.args.batch_size, n_samples))
                samples = np.reshape(samples, (self.args.batch_size, n_samples))
                samples = torch.from_numpy(samples)
                samples = samples.type(torch.FloatTensor)
                samples = samples.to(device)
                y = self(samples)
                mean = np.repeat(mean, self.args.batch_size)
                mean = np.reshape(mean, (self.args.batch_size, -1))
                mean = torch.from_numpy(mean)
                mean = mean.type(torch.FloatTensor)
                mean = mean.to(device)

                loss = mse(y, mean)
                losses.append(loss)
                scalar_losses.append(loss.item())
                updated = []
                grads = torch.autograd.grad(loss, self.params,  create_graph=True, retain_graph=True)
                for grad, param in zip(grads, self.params):
                    x = param - lr * grad
                    updated.append(x)
                
                final_values.append(updated)

                print(mean[0], y[0])


            for updated, index in zip(final_values, indexes):
                for p, x in zip(self.params, updated):
                    p.data.copy_(x)

                mean = data[index]
                samples = np.random.normal(mean, size=(self.args.batch_size, n_samples))
                samples = np.reshape(samples, (self.args.batch_size, n_samples))
                samples = torch.from_numpy(samples)
                samples = samples.type(torch.FloatTensor)
                samples = samples.to(device)
                y = self(samples)
                mean = np.repeat(mean, self.args.batch_size)
                mean = np.reshape(mean, (self.args.batch_size, -1))
                mean = torch.from_numpy(mean)
                mean = mean.type(torch.FloatTensor)
                mean = mean.to(device)

                loss = mse(y, mean)
                total_loss += loss

            for p, x in zip(self.params, initial_values):
                p.data.copy_(x)
            loss = total_loss / self.args.n_tasks
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if test:
                print("Test: ", mean[0], y[0])


    def eval(self):
        n_samples = self.args.n_samples
        mean = 0.25
        samples = np.random.normal(mean, size=(1, n_samples))
        samples = np.random.normal(mean, size=(1, n_samples))
        samples = np.reshape(samples, (1, n_samples))
        samples = torch.from_numpy(samples)
        samples = samples.type(torch.FloatTensor)
        samples = samples.to(device)
        y = self(samples)
        print(y)
        



                
def train_mine(cov_xy=0.9, mine_model=2, device='cpu'):
    if mine_model==2:
        model = Mine2().to(device)
    else:
        model = Mine1().to(device)
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
        x1 = x1.type(torch.FloatTensor)
        y1 = y1.type(torch.FloatTensor)
        x2 = x2.type(torch.FloatTensor)
        y2 = y2.type(torch.FloatTensor)
    
        if mine_model==2:
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x = np.arange(len(plot_loss))
    y = np.array(plot_loss).reshape(-1,)
    sns.set()
    sns.scatterplot(x=x, y=-y)
    plt.show()


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
    simple_maml.train(test=True)

