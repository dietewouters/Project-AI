# MLP Architecture Definition
# model.py

import torch
import torch.nn as nn
from config import config

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = config["input_dim"],       # 2048 fingerprint bits + 1 noisy value
        hidden_dims: list = config["hidden_dims"],
        dropout: float = config["dropout"],
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer — single value, no activation for regression
        layers.append(nn.Linear(dims[-1], 1))

        self.network = nn.Sequential(*layers)

        print(self.network)

    def forward(self, x):
        return self.network(x).squeeze(1)  # shape (batch_size,)

