import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_loader import get_dataloaders
from model import MLP
from config import config
from train import train
from optimize import run_optimization


def plot_history(history: dict):
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.show()


def main():
    FRACTION = 0.006
    NOISE = 0.01
    DATA_DIR = config["data_path"]
    INDICES_DIR = os.path.join(DATA_DIR, "indices")
    DATASET_DIR = os.path.join(DATA_DIR, "dataset")
    #FINGERPRINT_DIR = os.path.join(DATASET_DIR, "fingerprints")
    NOISE_DIR = os.path.join(DATASET_DIR, f"noise20_nitrogen")

    #TARGET_FILE = os.path.join(
    #    FINGERPRINT_DIR, f"groupadditivity_{FRACTION}_fingerprints.csv"
    #)

    TARGET_FILE = os.path.join(
        DATASET_DIR, f"groupadditivity_{FRACTION}.csv"
    )

    TRAIN_FILE = os.path.join(
        NOISE_DIR, f"groupadditivity_{FRACTION}_noise20_nitrogen.csv"
    )
    INDICES_FILE = os.path.join(INDICES_DIR, f"indices_{FRACTION}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    loaders = get_dataloaders(TARGET_FILE, TRAIN_FILE, INDICES_DIR)
    print("OK Dataloaders")

    
    train_dataset = loaders["train"].dataset
    val_dataset = loaders["val"].dataset
    test_dataset = loaders["test"].dataset

    # hyperparameter optimization
    #best_config = run_optimization(
    #    base_config=config,
    #    train_dataset=train_dataset,
    #    val_dataset=val_dataset,
    #    device=device,
    #    n_trials=10
    #)

    #print("\nBest config found:")
    #print(best_config)

    #
    final_loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False
        )
    }

    # final model
    model = MLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    )

    model, history = train(model, final_loaders, config, device)
    plot_history(history)


if __name__ == "__main__":
    main()