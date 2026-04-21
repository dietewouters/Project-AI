# Training loop + validation
import torch
import torch.nn as nn
import numpy as np

from model.model import MLP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.evaluate import evaluate
from model.data_loader import get_dataloaders

def train_one_epoch(model, loader, optimizer, criterion, device, delta = False, scaler=None, progress=None, augment_noise=False, base_sigma=0.01) -> float:
    model.train()
    total_loss = 0.0
    
    batch_task = None
    if progress is not None:
        batch_task = progress.add_task("[magenta]Processing Batches...", total=len(loader))
        
    for fp, noisy, y_true in loader:
        noisy_batch = noisy

        # 2. Scaling & Transformation
        if scaler:
            # Use scaler for both input and target.
            # PropertyScaler.transform handles delta/standard/minmax internally.
            noisy_in, y = scaler.transform(noisy_batch, y_true)
        else:
            noisy_in = noisy_batch
            if delta:
                y = (noisy_batch.view(-1) - y_true.view(-1))
            else:
                y = y_true

        optimizer.zero_grad()
        
        X = torch.cat([fp, noisy_in.view(-1, 1)], dim=1).to(device)
        X = X.float() # Ensure float32 for model pass
        y = y.to(device)
        predictions = model(X)
        loss = criterion(predictions.view(-1), y.view(-1))
            
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X)
        if progress is not None and batch_task is not None:
            progress.update(batch_task, advance=1)
                
    if progress is not None and batch_task is not None:
        progress.remove_task(batch_task)
        
    return total_loss / len(loader.dataset)


from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import torch

console = Console()


def train(model, loaders, config, device, optimizer=None, criterion=None, delta=False, scaler=None) -> tuple:
    augment_noise = config.get("augment_noise", False)
    base_sigma = config.get("base_sigma", 0.01)
    model.to(device)

    if optimizer is None:
        wd = config.get('weight_decay', 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=wd)
        
    # Learning Rate Scheduler to handle noisy continuous inputs
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=max(1, config['patience'] // 2)
    )

    if criterion is None:
        criterion = nn.MSELoss()


    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    console.print(f"\n[bold cyan]Training Configuration[/bold cyan]")
    console.print(f"[yellow]Epochs:[/yellow] {config['epochs']}")
    console.print(f"[yellow]Learning Rate:[/yellow] {config['lr']}")
    console.print(f"[yellow]Patience:[/yellow] {config['patience']}\n")

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=config['epochs'])

        for epoch in range(config['epochs']):
            train_loss = train_one_epoch(
                model, loaders['train'], optimizer, criterion, device, 
                delta=delta, scaler=scaler, progress=progress,
                augment_noise=augment_noise, base_sigma=base_sigma
            )
            val_loss = evaluate(model, loaders['val'], criterion, device, delta=delta, scaler=scaler)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Step the learning rate scheduler based on validation loss
            scheduler.step(val_loss)

            # Color based on performance
            train_color = "green" if epoch == 0 or train_loss < history['train_loss'][-2] else "red"
            val_color = "green" if val_loss < best_val_loss else "red"

            status_msg = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                status_msg = "[bold green]✓ Best[/bold green]"
            else:
                patience_counter += 1
                status_msg = f"[yellow]⚠ Patience: {patience_counter}/{config['patience']}[/yellow]"

            progress.update(task, advance=1)

            progress.console.print(
                f"Epoch [bold cyan]{epoch + 1:3d}/{config['epochs']}[/bold cyan] | "
                f"Train Loss: [{train_color}]{train_loss:.6f}[/{train_color}] | "
                f"Val Loss: [{val_color}]{val_loss:.6f}[/{val_color}] | "
                f"{status_msg}"
            )


            # Early Stopping
            if patience_counter == config['patience']:
                console.print(f"\n[bold red]Early stopping at epoch {epoch + 1}[/bold red]\n")
                break

    # Restore best model
    model.load_state_dict(best_model_state)

    console.print(f"[bold green]✓ Training Complete[/bold green]")
    console.print(f"[cyan]Best Val Loss: {best_val_loss:.6f}[/cyan]\n")

    return model, history
