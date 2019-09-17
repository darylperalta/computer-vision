
import torch
import torch.nn as nn
import torch.nn.functional as F

def mi(model, joint_x1, joint_x2, marginal_x1, marginal_x2):
    t = model(joint_x1, joint_x2)
    et = torch.exp(model(marginal_x1, marginal_x2))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.backbone = torch.nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, latent_dim),
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
    def __init__(self, latent_dim=16, hidden_units=256):
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

