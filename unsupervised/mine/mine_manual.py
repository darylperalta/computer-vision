import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats.contingency import margins

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n_bins = 100
covxy = .3

def sample(joint=True,
           mean=[0, 0],
           cov=[[1, covxy], [covxy, 1]],
           n_data=100000):
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

data = sample()
joint, edge = np.histogramdd(data, bins=n_bins)
joint /= joint.sum()
eps = np.finfo(float).eps
joint[joint<eps] = eps
x, y = margins(joint)
xy = x*y
xy[xy<eps] = eps
mi = joint*np.log(joint/xy)
print(mi)
print(joint)
print(xy)
mi = mi.sum()
print(mi)
# exit(0)

H=10
n_epoch = 1000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(1, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2  

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
plot_loss = []
for epoch in tqdm(range(n_epoch)):
    xy = sample()
    x1 = xy[:,0].reshape(-1,1)
    y1 = xy[:,1].reshape(-1,1)
    xy = sample(joint=False)
    x2 = xy[:,0].reshape(-1,1)
    y2 = xy[:,1].reshape(-1,1)
    
    x1 = Variable(torch.from_numpy(x1).type(torch.FloatTensor), requires_grad = True)
    y1 = Variable(torch.from_numpy(y1).type(torch.FloatTensor), requires_grad = True)
    x2 = Variable(torch.from_numpy(x2).type(torch.FloatTensor), requires_grad = True)
    y2 = Variable(torch.from_numpy(y2).type(torch.FloatTensor), requires_grad = True)
    
    pred_xy = model(x1, y1)
    pred_x_y = model(x2, y2)

    ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
    loss = - ret  # maximize
    plot_loss.append(loss.data.numpy())
    model.zero_grad()
    loss.backward()
    optimizer.step()

x = np.arange(len(plot_loss))
y = np.array(plot_loss).reshape(-1,)
#print(-plot_y)
#print(plot_y.shape)
#hv.Curve((plot_x, -plot_y)) * hv.Curve((plot_x, mi))
sns.set()
sns.scatterplot(x=x, y=-y)
plt.show()
