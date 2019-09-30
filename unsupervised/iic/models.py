
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(model):
    if type(model) == nn.Linear:
        nn.init.kaiming_normal_(model.weight)
    if type(model) == nn.Conv2d:
        nn.init.kaiming_normal_(model.weight)
    

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
            nn.Softmax(dim=1)
        )
        self.backbone.apply(init_weights)

    def forward(self, x):
        x = self.backbone(x)
        return x


class Model(nn.Module):
    def __init__(self, latent_dim=10):
        super(Model, self).__init__()
        self._backbone = Encoder(latent_dim=latent_dim)

    def forward(self, x, y):
        x = self._backbone(x)
        y = self._backbone(y)
        return [x, y]

    @property
    def backbone(self):
        return self._backbone

