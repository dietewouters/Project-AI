# Final evaluation on validation/test set
import torch


def compute_loss_on_original_scale(predictions, noisy, y_true, criterion, delta=False, target_scaler=None):
    predictions = predictions.view(-1)
    noisy = noisy.view(-1)
    y_true = y_true.view(-1)

    if delta:
        # model predicts delta = noisy - clean
        # => clean_pred = noisy - predicted_delta
        clean_pred = noisy - predictions
    else:
        clean_pred = predictions

    # unscale before computing loss
    if target_scaler is not None:
        clean_pred = target_scaler.inverse_transform_tensor(clean_pred)
        y_true = target_scaler.inverse_transform_tensor(y_true)

    loss = criterion(clean_pred, y_true)
    return loss


def evaluate(model, loader, criterion, device, delta=False, target_scaler=None) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for fp, noisy, y_true in loader:
            fp = fp.to(device)
            noisy = noisy.to(device)
            y_true = y_true.to(device)

            X = torch.cat([fp, noisy], dim=1)
            predictions = model(X)

            loss = compute_loss_on_original_scale(
                predictions=predictions,
                noisy=noisy,
                y_true=y_true,
                criterion=criterion,
                delta=delta,
                target_scaler=target_scaler
            )

            total_loss += loss.item() * len(y_true)

    return total_loss / len(loader.dataset)


def test(model, loader, criterion, device, delta=False, target_scaler=None) -> float:
    actual_loader = loader['test'] if isinstance(loader, dict) and 'test' in loader else loader

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for fp, noisy, y_true in actual_loader:
            fp = fp.to(device)
            noisy = noisy.to(device)
            y_true = y_true.to(device)

            X = torch.cat([fp, noisy], dim=1)
            predictions = model(X)

            loss = compute_loss_on_original_scale(
                predictions=predictions,
                noisy=noisy,
                y_true=y_true,
                criterion=criterion,
                delta=delta,
                target_scaler=target_scaler
            )

            total_loss += loss.item() * len(y_true)

    final_loss = total_loss / len(actual_loader.dataset)
    print(f"Test Loss: {final_loss:.4f}")
    return final_loss