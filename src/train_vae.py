import torch
import torch.nn.functional as F



def vae_loss_function(recon_logits, x, mu, logvar, beta):
    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def reconstruction_accuracy(recon_logits, x):
    probs = torch.sigmoid(recon_logits)
    preds = (probs >= 0.5).float()
    correct = (preds == x).float().sum()
    total = x.numel()
    return correct.item(), total


def train_vae(model, train_loader, val_loader, config, device):
    if config["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    history = {
    "train_total": [],
    "val_total": [],
    "train_recon": [],
    "val_recon": [],
    "train_kl": [],
    "val_kl": [],
    "train_recon_per_bit": [],
    "val_recon_per_bit": [],
    "train_acc": [],
    "val_acc": [],
}

    fp_dim = config["fp_input_dim"]

    for epoch in range(config["epochs"]):
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_kl = 0.0

        train_correct = 0.0
        train_bits = 0

        for fp, _, _ in train_loader:
            fp = fp.to(device)

            optimizer.zero_grad()
            recon_logits, mu, logvar = model(fp)

            total_loss, recon_loss, kl_loss = vae_loss_function(
                recon_logits, fp, mu, logvar, beta=config["beta"]
            )

            total_loss.backward()
            optimizer.step()

            train_total += total_loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

            correct, total = reconstruction_accuracy(recon_logits, fp)
            train_correct += correct
            train_bits += total

        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_kl = 0.0

        val_correct = 0.0
        val_bits = 0

        with torch.no_grad():
            for fp, _, _ in val_loader:
                fp = fp.to(device)

                recon_logits, mu, logvar = model(fp)
                total_loss, recon_loss, kl_loss = vae_loss_function(
                    recon_logits, fp, mu, logvar, beta=config["beta"]
                )

                val_total += total_loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()

                correct, total = reconstruction_accuracy(recon_logits, fp)
                val_correct += correct
                val_bits += total

                n_train = len(train_loader.dataset)
                n_val = len(val_loader.dataset)

        # gemiddeld per sample
        train_total_ps = train_total / n_train
        val_total_ps = val_total / n_val

        train_recon_ps = train_recon / n_train
        val_recon_ps = val_recon / n_val

        train_kl_ps = train_kl / n_train
        val_kl_ps = val_kl / n_val

        # reconstructie ook per bit, extra interpreteerbaar
        train_recon_per_bit = train_recon / (n_train * fp_dim)
        val_recon_per_bit = val_recon / (n_val * fp_dim)

        history["train_total"].append(train_total_ps)
        history["val_total"].append(val_total_ps)
        history["train_recon"].append(train_recon_ps)
        history["val_recon"].append(val_recon_ps)
        history["train_kl"].append(train_kl_ps)
        history["val_kl"].append(val_kl_ps)
        history["train_recon_per_bit"].append(train_recon_per_bit)
        history["val_recon_per_bit"].append(val_recon_per_bit)
        
        train_acc = train_correct / train_bits
        val_acc = val_correct / val_bits

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
        f"[VAE] Epoch {epoch+1}/{config['epochs']} | "
        f"Train total: {train_total_ps:.4f} | Val total: {val_total_ps:.4f} | "
        f"Train recon: {train_recon_ps:.4f} | Val recon: {val_recon_ps:.4f} | "
        f"Train KL: {train_kl_ps:.4f} | Val KL: {val_kl_ps:.4f} | "
        f"Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}"
)

    return history