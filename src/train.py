# Training loop + validation
import torch
import torch.nn as nn
from alembic.command import history

from model import MLP
from data_loader import get_dataloaders

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

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

def train(model, loaders, config, device, optimizer = None, criterion = None) -> tuple[MLP, dict]:
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    if criterion is None:
        criterion = nn.MSELoss()

    history = {"train_loss" : [], "val_loss" : []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, loaders['train'], optimizer, criterion, device)
        val_loss = evaluate(model, loaders['val'], criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch + 1}/{config['epochs']}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}")

        # ---- Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # ---- Early Stopping ----
        if patience_counter == config['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # ---- Restore best model ----
    model.load_state_dict(best_model_state)
    return model, history



def test(model, loaders, criterion, device) -> float:
    pass