# Training loop + validation
import torch
import torch.nn as nn

from model.model import MLP
from model.evaluate import evaluate
from model.data_loader import get_dataloaders

def train_one_epoch(model, loader, optimizer, criterion, device, delta = False, progress=None) -> float:
    model.train()
    total_loss = 0.0
    
    batch_task = None
    if progress is not None:
        batch_task = progress.add_task("[magenta]Processing Batches...", total=len(loader))
        
    if delta:
        for fp, noisy, _, y_delta in loader:
            X = torch.cat([fp, noisy], dim=1).to(device)
            y = y_delta.to(device)

            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            if progress is not None and batch_task is not None:
                progress.update(batch_task, advance=1)
    else:
        for fp, noisy, y_true in loader:
            X = torch.cat([fp, noisy], dim=1).to(device)
            y = y_true.to(device)

            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
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


def train(model, loaders, config, device, optimizer=None, criterion=None, delta=False) -> tuple:
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
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
            train_loss = train_one_epoch(model, loaders['train'], optimizer, criterion, device, delta=delta, progress=progress)
            val_loss = evaluate(model, loaders['val'], criterion, device, delta=delta)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Color based on performance
            train_color = "green" if epoch == 0 or train_loss < history['train_loss'][-2] else "red"
            val_color = "green" if val_loss < best_val_loss else "red"

            progress.update(task, advance=1)

            console.print(
                f"Epoch [bold cyan]{epoch + 1:3d}/{config['epochs']}[/bold cyan] | "
                f"Train Loss: [{train_color}]{train_loss:.6f}[/{train_color}] | "
                f"Val Loss: [{val_color}]{val_loss:.6f}[/{val_color}]",
                end=""
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                console.print(" [bold green]✓ Best[/bold green]")
            else:
                patience_counter += 1
                console.print(f" [yellow]⚠ Patience: {patience_counter}/{config['patience']}[/yellow]")

            # Early Stopping
            if patience_counter == config['patience']:
                console.print(f"\n[bold red]Early stopping at epoch {epoch + 1}[/bold red]\n")
                break

    # Restore best model
    model.load_state_dict(best_model_state)

    console.print(f"[bold green]✓ Training Complete[/bold green]")
    console.print(f"[cyan]Best Val Loss: {best_val_loss:.6f}[/cyan]\n")

    return model, history