
import torch
import torch.optim as optim
import torch.autograd as autograd

import numpy as np

def mi(joint, marginal, model):
    t = model(joint)
    et = torch.exp(model(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn(batch,
          model,
          optimizer,
          ma_et,
          ma_rate=0.01,
          device='cpu'):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = torch.FloatTensor(joint)
    joint = torch.autograd.Variable(joint).to(device)
    marginal = torch.FloatTensor(marginal)
    marginal = torch.autograd.Variable(marginal).to(device)
    mi_lb, t, et = mi(joint, marginal, model)
    ma_et = (1 - ma_rate) * ma_et
    ma_et += (ma_rate * torch.mean(et))
                                
    # unbiasing use moving average
    loss = torch.mean(t)
    loss -= ((1 / ma_et.mean()).detach() * torch.mean(et))
    loss = -loss

    optimizer.zero_grad()
    autograd.backward(loss)
    optimizer.step()
    return mi_lb, ma_et


def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]
