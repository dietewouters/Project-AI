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

def test(model, loader, criterion, device) -> float:
    total_loss = evaluate(model, loader['test'], criterion, device)
    print(f"Test Loss: {total_loss:.4f}")
    return total_loss