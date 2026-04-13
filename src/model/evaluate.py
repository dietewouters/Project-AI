# Final evaluation on test set
import torch

def evaluate(model, loader, criterion, device, delta=False) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for fp, noisy, y_true in loader:
            X = torch.cat([fp, noisy], dim=1).to(device)
            if delta:
                y = (noisy.squeeze(1) - y_true).to(device)
            else:
                y = y_true.to(device)
            predictions = model(X)
            loss = criterion(predictions.view(-1), y.view(-1))
            total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)

def test(model, loader, criterion, device, delta=False) -> float:
    actual_loader = loader['test'] if isinstance(loader, dict) and 'test' in loader else loader
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for fp, noisy, y_true in actual_loader:
            X = torch.cat([fp, noisy], dim=1).to(device)
            y_true = y_true.to(device)
            
            predictions = model(X)
            
            if delta:
                noisy_tensor = noisy.to(device)
                final_predictions = predictions + noisy_tensor.view(-1)
                loss = criterion(final_predictions.view(-1), y_true.view(-1))
            else:
                loss = criterion(predictions.view(-1), y_true.view(-1))
                
            total_loss += loss.item() * len(y_true)

    final_loss = total_loss / len(actual_loader.dataset)
    print(f"Test Loss: {final_loss:.4f}")
    return final_loss