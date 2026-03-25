# Final evaluation on test set
import torch

def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            loss = criterion(predictions, y)
            total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)

def test(model, loader, criterion, device) -> float:
    total_loss = evaluate(model, loader['test'], criterion, device)
    print(f"Test Loss: {total_loss:.4f}")
    return total_loss