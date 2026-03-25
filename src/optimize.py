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

        # ---- Sample hyperparameters from search space ----
        # e.g. cfg["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        # e.g. cfg["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        # e.g. cfg["hidden_dims"] = ...

        # ---- Instantiate model with sampled config ----

        # ---- Build dataloaders from the provided datasets ----

        # ---- Run training ----

        # ---- Return validation loss ----

    return objective


def run_optimization(base_config, train_dataset, val_dataset, device, n_trials: int = 50) -> dict:
    # ---- Create the study ----

    # ---- Run the search ----

    # ---- Log and return best config ----
    pass


def log_results(study: optuna.Study) -> None:
    # ---- Print best trial ----

    # ---- Print best hyperparameters ----

    # ---- Optionally save all trials to CSV for inspection ----
    pass