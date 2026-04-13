# Training loop + validation
import torch
import torch.nn as nn

from model.evaluate import evaluate

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

console = Console()


def compute_loss_on_original_scale(predictions, noisy, y_true, criterion, delta=False, target_scaler=None):
    """
    predictions: model output
    noisy: noisy input tensor
    y_true: clean target tensor
    delta:
        False -> model predicts clean target directly
        True  -> model predicts delta = noisy - y_true
    target_scaler:
        None -> no scaling used
        scaler object with inverse_transform_tensor(...) otherwise
    """

    predictions = predictions.view(-1)
    noisy = noisy.view(-1)
    y_true = y_true.view(-1)

    if delta:
        # Model predicts delta = noisy - clean
        # => clean_pred = noisy - predicted_delta
        clean_pred = noisy - predictions
    else:
        # Model predicts clean directly
        clean_pred = predictions

    # If scaling was used, unscale both predicted clean and true clean before loss
    if target_scaler is not None:
        clean_pred = target_scaler.inverse_transform_tensor(clean_pred)
        y_true = target_scaler.inverse_transform_tensor(y_true)

    loss = criterion(clean_pred, y_true)
    return loss


def train_one_epoch(model, loader, optimizer, criterion, device, delta=False, progress=None, target_scaler=None) -> float:
    model.train()
    total_loss = 0.0

    batch_task = None
    if progress is not None:
        batch_task = progress.add_task("[magenta]Processing Batches...", total=len(loader))

    for fp, noisy, y_true in loader:
        fp = fp.to(device)
        noisy = noisy.to(device)
        y_true = y_true.to(device)

        X = torch.cat([fp, noisy], dim=1)

        optimizer.zero_grad()
        predictions = model(X)

        loss = compute_loss_on_original_scale(
            predictions=predictions,
            noisy=noisy,
            y_true=y_true,
            criterion=criterion,
            delta=delta,
            target_scaler=target_scaler
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_true)

        if progress is not None and batch_task is not None:
            progress.update(batch_task, advance=1)

    if progress is not None and batch_task is not None:
        progress.remove_task(batch_task)

    return total_loss / len(loader.dataset)


def train(model, loaders, config, device, optimizer=None, criterion=None, delta=False, target_scaler=None) -> tuple:
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
    console.print(f"[yellow]Patience:[/yellow] {config['patience']}")
    console.print(f"[yellow]Delta Mode:[/yellow] {delta}")
    console.print(f"[yellow]Scaling Active:[/yellow] {target_scaler is not None}\n")

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
                model,
                loaders['train'],
                optimizer,
                criterion,
                device,
                delta=delta,
                progress=progress,
                target_scaler=target_scaler
            )

            val_loss = evaluate(
                model,
                loaders['val'],
                criterion,
                device,
                delta=delta,
                target_scaler=target_scaler
            )

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            train_color = "green" if epoch == 0 or train_loss < history['train_loss'][-2] else "red"
            val_color = "green" if val_loss < best_val_loss else "red"

            progress.update(task, advance=1)

            console.print(
                f"Epoch [bold cyan]{epoch + 1:3d}/{config['epochs']}[/bold cyan] | "
                f"Train Loss: [{train_color}]{train_loss:.6f}[/{train_color}] | "
                f"Val Loss: [{val_color}]{val_loss:.6f}[/{val_color}]",
                end=""
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                console.print(" [bold green]✓ Best[/bold green]")
            else:
                patience_counter += 1
                console.print(f" [yellow]⚠ Patience: {patience_counter}/{config['patience']}[/yellow]")

            if patience_counter == config['patience']:
                console.print(f"\n[bold red]Early stopping at epoch {epoch + 1}[/bold red]\n")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    console.print(f"[bold green]✓ Training Complete[/bold green]")
    console.print(f"[cyan]Best Val Loss: {best_val_loss:.6f}[/cyan]\n")

    return model, history