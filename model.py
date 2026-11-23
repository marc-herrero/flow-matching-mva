import torch
import torch.nn as nn


class VectorFieldMLP(nn.Module):
    def __init__(self, dim=1, width=128, depth=4):
        super().__init__()
        layers = [nn.Linear(dim + 1, width), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers.append(nn.Linear(width, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))
