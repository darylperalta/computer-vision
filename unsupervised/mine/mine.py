
import torch
import torch.nn as nn
import torch.nn.functional as F

def mi(joint, marginal, model):
    t = model(joint)
    et = torch.exp(model(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.backbone = torch.nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class Mine(nn.Module):
    def __init__(self, latent_dim, hidden_units):
        super(Mine, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_units)
        self.fc2 = nn.Linear(latent_dim, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)

    def forward(self, x, y):
        x = F.relu(self.fc1(x) + self.fc2(y))
        x = self.fc3(x)
        return x


class Model(nn.Module):
    def __init__(self, latent_dim=10, hidden_units=128):
        super(Model, self).__init__()
        self._backbone = Encoder(latent_dim=latent_dim)
        self.mine = Mine(latent_dim=latent_dim, hidden_units=hidden_units)

    def forward(self, x, y):
        x = self._backbone(x)
        y = self._backbone(y)
        x = self.mine(x, y)
        return x

    @property
    def backbone(self):
        return self._backbone

