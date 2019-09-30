
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def iic(z, zt, C=10):
    EPS = np.finfo(float).eps
    P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
    P = (P + P.t()) / 2.
    P = P / P.sum()
    P[(P < EPS).data] = EPS
    Pi = P.sum(dim=1).view(C, 1).expand(C, C)
    Pj = P.sum(dim=0).view(1, C).expand(C, C)
    Pi[(Pi < EPS).data] = EPS
    Pj[(Pj < EPS).data] = EPS
    return (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()

