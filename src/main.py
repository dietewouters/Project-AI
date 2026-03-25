# Entry point - orchestrates everything
import os
import torch
from data_loader import get_dataloaders
from model import MLP
from config import config
from train import train
import matplotlib.pyplot as plt

def plot_history(history: dict):
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.show()

def main():
    FRACTION = 0.004
    NOISE = 0.01
    DATA_DIR = config['data_path']
    INDICES_DIR = os.path.join(DATA_DIR, 'indices')
    DATASET_DIR = os.path.join(DATA_DIR, 'dataset')

    NOISE_DIR = os.path.join(DATASET_DIR, f'noise{NOISE}')

    TARGET_FILE = os.path.join(DATASET_DIR, f'groupadditivity_{FRACTION}.csv')
    TRAIN_FILE = os.path.join(NOISE_DIR, f'groupadditivity_{FRACTION}_noise{NOISE}.csv')
    INDICES_FILE = os.path.join(INDICES_DIR, f'indices_{FRACTION}.csv')

    loaders = get_dataloaders(TARGET_FILE, TRAIN_FILE, INDICES_FILE)
    print("OK Dataloaders")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, history = train(MLP(), loaders, config, device)
    plot_history(history)

if __name__ == '__main__':
    main()