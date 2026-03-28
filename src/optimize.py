# optimize.py

import copy
import optuna
import pandas as pd
from torch.utils.data import DataLoader

from model import MLP
from train import train


def create_optuna_objective(base_config, train_dataset, val_dataset, device):
    def objective(trial):
        cfg = copy.deepcopy(base_config)

        # ---- Sample hyperparameters from search space ----
        cfg["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        cfg["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])

        hidden_dims_options = [
            [128, 64],
            [256, 128],
            [512, 256],
            [512, 256, 128],
            [1024, 512, 256],
        ]
        cfg["hidden_dims"] = trial.suggest_categorical("hidden_dims", hidden_dims_options)

        # ---- Instantiate model with sampled config ----
        model = MLP(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        )

        # ---- Build dataloaders from the provided datasets ----
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False
        )

        loaders = {
            "train": train_loader,
            "val": val_loader
        }

        # ---- Run training ----
        _, history = train(model, loaders, cfg, device)

        # ---- Return validation loss ----
        return min(history["val_loss"])

    return objective


def run_optimization(base_config, train_dataset, val_dataset, device, n_trials: int = 20 ) -> dict:
    # ---- Create the study ----
    study = optuna.create_study(direction="minimize")

    # ---- Run the search ----
    objective = create_optuna_objective(base_config, train_dataset, val_dataset, device)
    study.optimize(objective, n_trials=n_trials)

    # ---- Log and return best config ----
    log_results(study)

    best_config = copy.deepcopy(base_config)
    best_config.update(study.best_trial.params)

    return best_config


def log_results(study: optuna.Study) -> None:
    # ---- Print best trial ----
    print("\n===== OPTUNA RESULTS =====")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.6f}")

    # ---- Print best hyperparameters ----
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")

    # ---- Optionally save all trials to CSV for inspection ----
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
    print("\nSaved all trial results to optuna_trials.csv")