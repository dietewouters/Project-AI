# Final evaluation on test set
import torch

def evaluate(model, loader, criterion, device, delta=False) -> float:
    model.eval()
    total_loss = 0.0
    if delta:
        with torch.no_grad():
            for fp, noisy, _, y_delta in loader:
                X = torch.cat([fp, noisy], dim=1).to(device)
                y = y_delta.to(device)
                predictions = model(X)
                loss = criterion(predictions, y)
                total_loss += loss.item() * len(y)
    else:
        with torch.no_grad():
            for fp, noisy, y_true in loader:
                X = torch.cat([fp, noisy], dim=1).to(device)
                y = y_true.to(device)
                predictions = model(X)
                loss = criterion(predictions, y)
                total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)

def test(model, loader, criterion, device, delta=False) -> float:
    actual_loader = loader['test'] if isinstance(loader, dict) and 'test' in loader else loader
    model.eval()
    total_loss = 0.0
    if delta:
        with torch.no_grad():
            for fp, noisy, y_true, y_delta in actual_loader:
                X = torch.cat([fp, noisy], dim=1).to(device)
                y_true = y_true.to(device)
                noisy_tensor = noisy.to(device)
                predictions = model(X)
                final_predictions = predictions + noisy_tensor
                loss = criterion(final_predictions, y_true)
                total_loss += loss.item() * len(y_true)
    else:
        with torch.no_grad():
            for fp, noisy, y_true in actual_loader:
                X = torch.cat([fp, noisy], dim=1).to(device)
                y = y_true.to(device)
                predictions = model(X)
                loss = criterion(predictions, y)
                total_loss += loss.item() * len(y)

    final_loss = total_loss / len(actual_loader.dataset)
    print(f"Test Loss: {final_loss:.4f}")
    return final_loss