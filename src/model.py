# MLP Architecture Definition
# model.py

import torch
import torch.nn as nn
from config import config

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: list = None,
        dropout: float = None,
    ):
        super().__init__()

        if input_dim is None:
            input_dim = config["input_dim"]
        if hidden_dims is None:
            hidden_dims = config["hidden_dims"]
        if dropout is None:
            dropout = config["dropout"]

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 1))

        self.network = nn.Sequential(*layers)

        print(self.network)

    def forward(self, x):
        return self.network(x).squeeze(1)

