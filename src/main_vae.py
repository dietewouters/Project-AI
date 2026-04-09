import torch

from config import config_VAE
from vae_model import VAE
from train_vae import train_vae
from data_loader_vae import get_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(
        dataset_path=r"data/groupadditivity_h298/dataset/fingerprints/groupadditivity_0.004_fingerprints.csv",
        noisy_path=r"data/groupadditivity_h298/dataset/noise0.4/groupadditivity_0.004_noise0.4.csv",
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    vae = VAE(
        fp_input_dim=config_VAE["fp_input_dim"],
        hidden_dims=config_VAE["hidden_dims"],
        latent_dim=config_VAE["latent_dim"],
    ).to(device)

    history = train_vae(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_VAE,
        device=device,
    )

    return vae, history


if __name__ == "__main__":
    main()