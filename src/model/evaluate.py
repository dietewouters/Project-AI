# Final evaluation on test set
import torch
import numpy as np


def evaluate(model, loader, criterion, device, delta=False, scaler=None) -> float:
    """
    Validation evaluation: Computes loss in scaled space (if scaler present).
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for fp, noisy, y_true in loader:
            if scaler:
                # Standardize inputs and targets together
                noisy_proc, y = scaler.transform(noisy, y_true)
            else:
                noisy_proc = noisy
                if delta:
                    y = (noisy.view(-1) - y_true.view(-1))
                else:
                    y = y_true

            # Prepare features
            y = y.to(device)
            X = torch.cat([fp, noisy_proc.view(-1, 1)], dim=1).to(device).float()
            predictions = model(X)
            loss = criterion(predictions.view(-1), y.view(-1))
                
            total_loss += loss.item() * len(fp)

    return total_loss / len(loader.dataset)

def test(model, loader, criterion, device, delta=False, scaler=None) -> float:
    """
    Final test evaluation: Reports loss in physical (unscaled) units.
    """
    actual_loader = loader['test'] if isinstance(loader, dict) and 'test' in loader else loader
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for fp, noisy, y_true in actual_loader:
            if scaler:
                # Transform only noisy input for the model pass
                noisy_proc = scaler.transform(noisy)
            else:
                noisy_proc = noisy

            # Prepare features
            y_true_eval = y_true.to(device)
            
            X = torch.cat([fp, noisy_proc.view(-1, 1)], dim=1).to(device).float()
            predictions = model(X)
            
            if scaler:
                # Use PropertyScaler to map prediction back to real physical units
                final_predictions = scaler.inverse_transform(noisy, predictions.view(-1))
                final_predictions = final_predictions.to(device)
            else:
                if delta:
                    # predictions is raw delta, so True = Noisy - Delta
                    final_predictions = noisy.to(device).view(-1) - predictions.view(-1)
                else:
                    final_predictions = predictions.view(-1)
                
            loss = criterion(final_predictions.view(-1), y_true_eval.view(-1))
            total_loss += loss.item() * len(X)

    final_loss = total_loss / len(actual_loader.dataset)
    print(f"Test Loss (Physical Units): {final_loss:.4f}")
    return final_loss