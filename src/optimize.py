# Hyperparameter search logic
import optuna
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import MLP
from train import train
from evaluate import evaluate
from data_loader import get_dataloaders

def create_optuna_objective(base_config, train_dataset, val_dataset, device):
    def objective(trial):
        cfg = copy.deepcopy(base_config)

