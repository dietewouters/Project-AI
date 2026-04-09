import os
import copy
import torch

from config import config_VAE, config
from data_loader_vae import get_dataloaders
from vae_model import VAE
from train_vae import train_vae
from make_latens import encode_dataset
from latent_data_loader import make_latent_dataloaders

from model import MLP
from train import train


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(
        config_VAE["data_path"],
        "dataset",
        "fingerprints/groupadditivity_0.004_fingerprints.csv"
    )
    noisy_path = os.path.join(
        config_VAE["data_path"],
        "dataset",
        "noise0.4",
        "groupadditivity_0.004_noise0.4.csv"
    )

    # ---- 1. load fp/noisy/target data voor VAE ----
    loaders_vae = get_dataloaders(
        dataset_path=dataset_path,
        noisy_path=noisy_path,
    )

    train_loader_vae = loaders_vae["train"]
    val_loader_vae = loaders_vae["val"]

    # ---- 2. train VAE ----
    vae = VAE(
        fp_input_dim=config_VAE["fp_input_dim"],
        hidden_dims=config_VAE["hidden_dims"],
        latent_dim=config_VAE["latent_dim"],
    ).to(device)

    vae_history = train_vae(
        model=vae,
        train_loader=train_loader_vae,
        val_loader=val_loader_vae,
        config=config_VAE,
        device=device,
    )

    # ---- 3. encode datasets naar latent vectors ----
    z_train, noisy_train, y_train = encode_dataset(vae, train_loader_vae, device)
    z_val, noisy_val, y_val = encode_dataset(vae, val_loader_vae, device)

    # ---- 4. maak loaders voor oude MLP-train pipeline ----
    latent_loaders = make_latent_dataloaders(
        z_train=z_train,
        noisy_train=noisy_train,
        y_train=y_train,
        z_val=z_val,
        noisy_val=noisy_val,
        y_val=y_val,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # ---- 5. hergebruik oude config, maar pas input_dim aan ----
    mlp_config = copy.deepcopy(config)
    mlp_config["input_dim"] = config_VAE["latent_dim"] + 1

    # ---- 6. hergebruik oude MLP ----
    model = MLP(
        input_dim=mlp_config["input_dim"],
        hidden_dims=mlp_config["hidden_dims"],
        dropout=mlp_config["dropout"],
    ).to(device)

    # ---- 7. hergebruik oude train() ----
    model, mlp_history = train(
        model=model,
        loaders=latent_loaders,
        config=mlp_config,
        device=device,
    )

    return vae, model, vae_history, mlp_history


if __name__ == "__main__":
    vae, model, vae_history, mlp_history = main()