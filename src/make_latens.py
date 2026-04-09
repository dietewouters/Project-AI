import torch


def encode_dataset(vae, loader, device):
    vae.eval()

    z_list = []
    noisy_list = []
    target_list = []

    with torch.no_grad():
        for fp, noisy, target in loader:
            fp = fp.to(device)

            z = vae.get_latent(fp)

            z_list.append(z.cpu())
            noisy_list.append(noisy.cpu())
            target_list.append(target.cpu())

    z_all = torch.cat(z_list, dim=0)
    noisy_all = torch.cat(noisy_list, dim=0)
    target_all = torch.cat(target_list, dim=0)

    return z_all, noisy_all, target_all