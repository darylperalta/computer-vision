
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib.mine import mi, learn, ma
from lib import device

class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

def sample(data, batch_size=100, mode='joint'):
    if mode == 'joint':
        index = np.random.choice(range(data.shape[0]),
                                 size=batch_size,
                                 replace=False)
        sample = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]),
                                       size=batch_size,
                                       replace=False)
        marginal_index = np.random.choice(range(data.shape[0]),
                                          size=batch_size,
                                          replace=False)
        x = data[joint_index][:,0].reshape(-1,1)
        y = data[marginal_index][:,1].reshape(-1,1)
        sample = np.concatenate([x, y], axis=1)
    
    return sample


def train(data,
          model,
          optimizer, 
          batch_size=100, 
          iter_num=5000,
          log_freq=1000):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        joint = sample(data, batch_size=batch_size)
        marginal = sample(data,
                          batch_size=batch_size,
                          mode='marginal')
        batch = joint, marginal
        mi_lb, ma_et = learn(batch,
                             model,
                             optimizer,
                             ma_et)
        result.append(mi_lb.detach().cpu().numpy())
        if (i+1)%(log_freq)==0:
            print(result[-1])

    return result


x = np.random.multivariate_normal(mean=[0,0],
                                  cov=[[1,0], [0,1]],
                                  size = 300)

y = np.random.multivariate_normal(mean=[0,0],
                                  cov=[[1,0.8], [0.8,1]],
                                  size = 300)

sns.set()
#sns.scatterplot(x=x[:,0], y=x[:,1])
#sns.scatterplot(x=y[:,0], y=y[:,1])

joint_data = sample(y,
                    batch_size=100,
                    mode='joint')
sns.scatterplot(x=joint_data[:,0],
                y=joint_data[:,1],
                color='red')
#plt.show()

marginal_data = sample(y,
                       batch_size=100,
                       mode='marginal')
sns.scatterplot(x=marginal_data[:,0],
                y=marginal_data[:,1],
                color='red')



plt.clf()
model = Mine().to(device.get())
optimizer = optim.Adam(model.parameters(), lr=1e-3)
result = train(x, model, optimizer)
result = ma(result)
plt.plot(range(len(result)), result)
plt.show()


plt.clf()
model = Mine().to(device.get())
optimizer = optim.Adam(model.parameters(), lr=1e-3)
result = train(y, model, optimizer)
result = ma(result)
plt.plot(range(len(result)), result)
plt.show()
