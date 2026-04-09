import os
import copy
import torch

from config import config
from model import MLP
from train import train
from noisy_only_loader import get_noisy_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(
        config["data_path"],
        "dataset",
        "fingerprints/groupadditivity_0.004_fingerprints.csv"
    )

    noisy_path = os.path.join(
        config["data_path"],
        "dataset",
        "noise0.4",
        "groupadditivity_0.004_noise0.4.csv"
    )

    # ---- dataloaders ----
    loaders = get_noisy_dataloaders(
        dataset_path=dataset_path,
        noisy_path=noisy_path,
    )

    # ---- config aanpassen ----
    model_config = copy.deepcopy(config)
    model_config["input_dim"] = 1   # 🔥 alleen noisy

    # ---- model ----
    model = MLP(
        input_dim=model_config["input_dim"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
    ).to(device)

    # ---- trainen ----
    model, history = train(
        model=model,
        loaders=loaders,
        config=model_config,
        device=device,
    )

    return model, history


if __name__ == "__main__":
    main()