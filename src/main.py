import os
import glob
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
import questionary
from model.data_loader import get_dataloaders
from model.model import MLP, MLPDelta
from config import config
from model.train import train
from model.data_loader_scaled import get_dataloaders_scaled
# from optimize import run_optimization

console = Console()

def plot_history(history: dict):
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["train_eval_loss"], label="Train Loss at end of epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.show()

def plot_history2(history: dict):
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["train_eval_loss"], label="Train Loss at end of epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.show()

def get_file_len(filepath):
    # Fast line counter skipping header
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return sum(1 for _ in f) - 1
    return 0

import re

def get_file_len(filepath):
    # Fast line counter skipping header
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return sum(1 for _ in f) - 1
    return 0

def get_fraction_from_filename(filename):
    match = re.search(r'_([0-9]+\.[0-9]+)', filename)
    if match:
        return float(match.group(1))
    # Check if integer 1 e.g. groupadditivity_1
    match_int = re.search(r'_1[_\.]', filename)
    if match_int:
        return 1.0
    return 0.0

def interactive_setup(data_dir: str, splits=None):
    console.print("\n[bold magenta]--- Dataset Setup Wizard ---[/bold magenta]")
    
    dataset_dir = os.path.join(data_dir, "dataset")
    indices_dir = os.path.join(data_dir, "indices")
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    if splits is None:
        splits = ["train", "val"]
    
    available_targets = glob.glob(os.path.join(dataset_dir, "*.csv"))
    available_targets = [os.path.basename(f) for f in available_targets if "noise" not in os.path.basename(f)]
    
    if not available_targets:
        available_targets = ["groupadditivity_1.csv", "groupadditivity_secondarytest.csv", "groupadditivity_test.csv"]
        
    loaders_config = {}
    
    for split in splits:
        console.print(f"\n[bold cyan]Configuring {split.upper()} Dataset[/bold cyan]")
        
        # Smart defaults based on split
        def_target = available_targets[0]
        if split == "train" and "groupadditivity_1.csv" in available_targets: def_target = "groupadditivity_1.csv"
        if split == "val" and "groupadditivity_secondarytest.csv" in available_targets: def_target = "groupadditivity_secondarytest.csv"
        if split == "test" and "groupadditivity_test.csv" in available_targets: def_target = "groupadditivity_test.csv"
        
        # Menu selection with arrow keys
        target_name = questionary.select(
            f"Select Target Master Dictionary for {split.upper()}:",
            choices=available_targets,
            default=def_target
        ).ask()
        
        target_path = os.path.join(dataset_dir, target_name)
        
        parts = target_name.split('_')
        indices_filename = "indices_" + parts[-1] if len(parts) > 1 else target_name
        indices_path = os.path.join(indices_dir, indices_filename)
        
        max_mols = get_file_len(target_path)
        if max_mols <= 0: max_mols = 1000 # Test default
        
        total_molecules = IntPrompt.ask(f"How many total molecules to use? (Max: {max_mols})", default=max_mols)
        master_fraction_ratio = float(total_molecules) / max_mols if max_mols > 0 else 1.0
        
        mode = questionary.select("Select noise generation mode", choices=["precomputed", "on-the-fly"], default="precomputed").ask()
        
        if mode == "precomputed":
            try:
                noise_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and "noise" in d.lower()]
            except FileNotFoundError:
                noise_dirs = []
                
            if not noise_dirs:
                noise_dirs = ["noise0.01", "noise0.02", "noise20_nitrogen"]
                
            active_noise_dirs = questionary.checkbox(
                "Select which noise levels to include in this set (Space to check, Enter to submit):",
                choices=noise_dirs
            ).ask()
            
            if not active_noise_dirs:
                active_noise_dirs = []
                    
            dataset_blocks = []
            if active_noise_dirs:
                fully_random = questionary.confirm("Distribute perfectly equally automatically?", default=False).ask()
                remaining = 1.0
                
                for i, d in enumerate(active_noise_dirs):
                    if fully_random:
                        pct = 1.0 / len(active_noise_dirs)
                    else:
                        if i == len(active_noise_dirs) - 1:
                            pct = remaining
                            console.print(f"[green]Auto-assigning remaining {pct*100:.1f}% to '{d}'.[/green]")
                        else:
                            pct_str = Prompt.ask(f"Fraction for '{d}'? (0.0 to {remaining:.2f})", default=str(round(remaining / (len(active_noise_dirs)-i), 2)))
                            pct = float(pct_str)
                            if pct > remaining:
                                pct = remaining
                            remaining -= pct
                            
                    # Required absolute fraction for this noise
                    req_fraction = master_fraction_ratio * pct
                    console.print(f"Targeting fraction ~{req_fraction:.3f} for noise '{d}'")
                    
                    noise_dir_path = os.path.join(dataset_dir, d)
                    try:
                        avail_nf = [f for f in os.listdir(noise_dir_path) if "groupadditivity_" in f and f.endswith(".csv")]
                    except FileNotFoundError:
                        avail_nf = []
                        
                    # Filter files so we only pick noise blocks that actually match our target dictionary!
                    if "secondarytest" in target_name:
                        avail_nf = [f for f in avail_nf if "secondarytest" in f]
                    elif "test" in target_name:
                        avail_nf = [f for f in avail_nf if "test" in f and "secondarytest" not in f]
                    else:
                        avail_nf = [f for f in avail_nf if "test" not in f]
                        
                    # Filter and sort files by fraction (Descending ensures we use the largest files first, minimizing total file count)
                    file_fractions = []
                    for nf in avail_nf:
                        f_val = get_fraction_from_filename(nf)
                        
                        # Test and secondarytest sets are often whole, not fractionally stripped. Force include them.
                        if "test" in target_name and f_val == 0.0:
                            f_val = 1.0 
                            
                        if f_val > 0.0:
                            file_fractions.append((nf, f_val))
                            
                    if not file_fractions:
                        console.print(f"  [red]Warning: No appropriately matched noise files found in '{d}' for '{target_name}'![/red]")
                        
                    file_fractions.sort(key=lambda x: x[1], reverse=True)
                    
                    current_sum = 0.0
                    for nf, f_val in file_fractions:
                        if current_sum >= req_fraction:
                            break
                        current_sum += f_val
                        
                        # Use the string representation from filename or default formatting to find clean/index matches
                        if "secondarytest" in nf:
                            s_f = "secondarytest"
                        elif "test" in nf:
                            s_f = "test"
                        else:
                            # Replicate the f_val string safely from the matched regex
                            matched_str_grp = re.search(r'_([0-9]+\.[0-9]+)', nf)
                            if matched_str_grp:
                                s_f = matched_str_grp.group(1)
                            else:
                                s_f = "1"
                                
                        clean_fn = f"groupadditivity_{s_f}.csv"
                        idx_fn = f"indices_{s_f}.csv"
                        
                        dataset_blocks.append({
                            "clean_path": os.path.join(dataset_dir, clean_fn),
                            "noisy_path": os.path.join(noise_dir_path, nf),
                            "indices_path": os.path.join(indices_dir, idx_fn)
                        })
                        console.print(f"  [cyan]+ Picked File:[/cyan] {nf} (fraction {f_val})")
            
            loaders_config[split] = {
                "mode": "precomputed",
                "target_path": target_path,
                "indices_path": indices_path,
                "total_molecules": total_molecules,
                "blocks": dataset_blocks
            }
            
        elif mode == "on-the-fly":
            noise_std = Prompt.ask("Enter Gaussian noise standard deviation", default="0.01")
            loaders_config[split] = {
                "mode": "on-the-fly",
                "target_path": target_path,
                "indices_path": indices_path,
                "total_molecules": total_molecules,
                "noise_std": float(noise_std)
            }
            
    if splits == ["train", "val"]:
        if Confirm.ask("\n[bold yellow]Do you want to configure a Test dataset as well?[/bold yellow]", default=False):
            test_config = interactive_setup(data_dir, splits=["test"])
            loaders_config.update(test_config)
            
    return loaders_config

def test_model(model, data_dir, loaders_config=None, test_loader=None, device=None, delta=False, base_name=None, target_scaler=None):
    """
    Dedicated function for testing a trained model.
    Pass a loaders_config with 'test' defined, or it will launch the setup wizard.
    """
    from model.evaluate import test
    import torch.nn as nn
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if test_loader is None:
        if loaders_config is None or "test" not in loaders_config:
            console.print("\n[bold yellow]No test configuration passed. Launching Wizard...[/bold yellow]")
            loaders_config = interactive_setup(data_dir, splits=["test"])

        console.print("\n[bold cyan]Loading Test Dataset...[/bold cyan]")

        scaling_type = None
        if "test" in loaders_config:
            scaling_type = loaders_config["test"].get("scaling", "none")

        if scaling_type == "none":
            loaders = get_dataloaders(loaders_config)
            train_names = set(loaders['train'].dataset.names)

        else:
            loaders, target_scaler = get_dataloaders_scaled(loaders_config)

        if "test" not in loaders:
            console.print("[red]Test DataLoader could not be built. Aborting test.[/red]")
            return None

        test_loader = loaders["test"]
        
    criterion = nn.MSELoss() # Update this if your test function relies on another default
    
    console.print("\n[bold cyan]--- Running Test Evaluation ---[/bold cyan]")
    test_loss = test(
    model,
    test_loader,
    criterion,
    device,
    delta=delta,
    target_scaler=target_scaler
)
    console.print(f"[bold green]Final Test Loss:[/bold green] {test_loss:.6f}\n")
    
    if not base_name:
        import time
        base_name = f"manual_test_{int(time.time())}"
        
    tests_dir = os.path.join("results", "tests")
    os.makedirs(tests_dir, exist_ok=True)
    test_path = os.path.join(tests_dir, f"{base_name}_test.json")
    import json
    with open(test_path, 'w') as f:
        json.dump({"test_loss": test_loss}, f, indent=4)
    console.print(f"[bold green]✔ Test Score saved safely to:[/bold green] {test_path}\n")
        
    return test_loss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud_bundle", type=str, default=None, help="Path to a pre-packaged Train/Val .pt Kaggle Dataset Bundle")
    parser.add_argument("--test_bundle", type=str, default=None, help="Path to a pre-packaged Test .pt Kaggle Dataset Bundle")
    args = parser.parse_args()
    
    DATA_DIR = config["data_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_scaler = None
    
    # ---------------- HEADLESS KAGGLE BOOT ---------------- #
    if args.cloud_bundle or args.test_bundle:
        console.print(f"\n[bold magenta]--- KAGGLE SERVER ENVIRONMENT DETECTED ---[/bold magenta]")
        console.print(f"[cyan]Bypassing explicit wizard dependencies.[/cyan]")
        
        from model.data_loader import get_cloud_dataloaders
        loaders = {}
        delta_choice = False
        
        if args.cloud_bundle:
            l1 = get_cloud_dataloaders(args.cloud_bundle)
            loaders.update(l1)
            
        if args.test_bundle:
            l2 = get_cloud_dataloaders(args.test_bundle)
            loaders.update(l2)
            
        loaders_config = {"test": True} if "test" in loaders else {} # Mock configuration presence
        
        console.print(f"[bold green]✔ Headless Extraction Complete. Initiating Native Boot.[/bold green]\n")
        
        # If no Train/Val is provided, just jump completely to test_model directly against weights!
        if args.test_bundle and not args.cloud_bundle:
             console.print("[yellow]Only Test Bundle Provided! (Please ensure you implement specific weight loading if you intended to do so later!)[/yellow]")
        
    # ---------------- INTERACTIVE LOCAL BOOT ---------------- #
    # ---------------- INTERACTIVE LOCAL BOOT ---------------- #
    else:
        console.print("\n[bold magenta]--- General Execution Mode ---[/bold magenta]")
        import questionary
        
        exec_mode = questionary.select(
            "What would you like to do?",
            choices=["Train a New Model", "Test an Existing Model"]
        ).ask()
        
        if exec_mode == "Test an Existing Model":
            model_files = glob.glob(os.path.join("results", "models", "*.pth"))
            if not model_files:
                console.print("[red]No trained models found in results/models/ ![/red]")
                return
            chosen_model_path = questionary.select("Select a trained model to evaluate:", choices=model_files).ask()
            
            delta_choice = questionary.confirm("Are we performing Delta Evaluation (predicting noise differences)?", default=False).ask()
            
            console.print("\n[bold magenta]--- Test Dataset Config ---[/bold magenta]")
            use_ready_loader = questionary.confirm("Provide testing dataset from an already ready bundle (.pt)?", default=False).ask()
            
            loaders = {}
            loaders_config = {}
            
            if use_ready_loader:
                bundle_files = glob.glob(os.path.join("results", "cloud_datasets", "*.pt"))
                if not bundle_files:
                    console.print("[yellow]No '.pt' bundles found. Falling back to manual setup.[/yellow]")
                    use_ready_loader = False
                else:
                    chosen_bundle = questionary.select("Choose a dataset bundle:", choices=bundle_files).ask()
                    from model.data_loader import get_cloud_dataloaders
                    l1 = get_cloud_dataloaders(chosen_bundle)
                    loaders.update(l1)
                    
            if not use_ready_loader:
                loaders_config = interactive_setup(DATA_DIR, splits=["test"])
                console.print("\n[bold cyan]Loading Test Dataset...[/bold cyan]")
                loaders = get_dataloaders(loaders_config)
                
            # If the user chose a training bundle to evaluate training loss, we capture the best available loader
            test_loader = loaders.get("test") or loaders.get("val") or loaders.get("train")
            
            if test_loader is None:
                console.print("[red]Test DataLoader could not be generated. Aborting.[/red]")
                return

            import json
            base = os.path.basename(chosen_model_path).replace(".pth", "")
            base_config = os.path.join("results", "configs", f"{base}_config.json")
            if os.path.exists(base_config):
                with open(base_config, 'r') as f:
                    run_config = json.load(f)
            else:
                run_config = config
                
            if delta_choice:
                model = MLPDelta(
                    input_dim=run_config["input_dim"],
                    hidden_dims=run_config["hidden_dims"],
                    dropout=run_config["dropout"],
                ).to(device)
            else:
                model = MLP(
                    input_dim=run_config["input_dim"],
                    hidden_dims=run_config["hidden_dims"],
                    dropout=run_config["dropout"],
                ).to(device)
                
            model.load_state_dict(torch.load(chosen_model_path, map_location=device, weights_only=True))
            
            test_model(
                model, 
                DATA_DIR, 
                loaders_config=loaders_config, 
                test_loader=loaders.get("test"), 
                device=device, 
                delta=delta_choice,
                base_name=base_name,
                target_scaler=target_scaler
            )
            return

        # ==========================================
        # TRAIN A NEW MODEL
        # ==========================================
        console.print("\n[bold magenta]--- New Model Training Config ---[/bold magenta]")
        
        delta_choice = questionary.confirm("Are we performing Delta Training (predicting noise differences)?", default=False).ask()
        
        use_ready_loader = questionary.confirm("Train locally with an already ready data loader bundle (.pt)?", default=False).ask()
        
        scaling_type = questionary.select(
        "Which scaling do you want to use for enthalpy values?",
        choices=[
            "none",
            "standard",
            "minmax"
            ]
        ).ask()
        
        loaders = {}
        
        if use_ready_loader:
            bundle_files = glob.glob(os.path.join("results", "cloud_datasets", "*.pt"))
            if not bundle_files:
                console.print("[yellow]No '.pt' bundles found in 'results/cloud_datasets/'. Falling back to manual setup.[/yellow]")
                use_ready_loader = False
            else:
                chosen_bundle = questionary.select(
                    "Choose a dataset bundle:",
                    choices=bundle_files
                ).ask()
                
                from model.data_loader import get_cloud_dataloaders
                l1 = get_cloud_dataloaders(chosen_bundle)
                loaders.update(l1)
                loaders_config = {"test": True} if "test" in loaders else {}
                target_scaler = None
                
        if not use_ready_loader:
    
            # Run the interactive setup wizard for train & val
            loaders_config = interactive_setup(DATA_DIR)

            console.print("\n[bold cyan]Loading Datasets...[/bold cyan]")

            if scaling_type == "none":
                loaders = get_dataloaders(loaders_config)
                target_scaler = None
                train_names = set(loaders['train'].dataset.names)
                val_names = set(loaders['val'].dataset.names)

                overlap = train_names.intersection(val_names)

                print(f"\n[DEBUG] Overlap train-val molecules: {len(overlap)}")
            else:
                for split in ["train", "val", "test"]:
                    if split in loaders_config:
                        loaders_config[split]["scaling"] = scaling_type

                loaders, target_scaler = get_dataloaders_scaled(loaders_config)

            print("OK Dataloaders")
            
            action_choice = questionary.select(
                "Dataset Generation Complete! What would you like to do next?",
                choices=["Train Locally", "Export to Cloud (.pt)", "Both"],
                default="Train Locally"
            ).ask()
            
            if action_choice in ["Export to Cloud (.pt)", "Both"]:
                import time
                timestamp = int(time.time())
                from model.data_loader import export_to_cloud_bundle
                console.print(f"\n[cyan]Packaging Native PyTorch Tensors for Cloud Deployment...[/cyan]")
                
                train_val_loaders = {k: v for k, v in loaders.items() if k in ["train", "val"]}
                test_loaders = {k: v for k, v in loaders.items() if k == "test"}
                
                if train_val_loaders:
                    custom_train_name = questionary.text("Enter a name for the Train/Val dataset bundle (leave blank for timestamp):").ask()
                    bundle_tr_name = custom_train_name if custom_train_name else f"kaggle_train_val_{timestamp}"
                    export_path_train = os.path.join("results", "cloud_datasets", f"{bundle_tr_name}.pt")
                    export_to_cloud_bundle(train_val_loaders, export_path_train)
                    console.print(f"[bold green]✔ Train/Val Dataset successfully sealed at:[/bold green] {export_path_train}")
                    
                if test_loaders:
                    custom_test_name = questionary.text("Enter a name for the Test dataset bundle (leave blank for timestamp):").ask()
                    bundle_te_name = custom_test_name if custom_test_name else f"kaggle_test_{timestamp}"
                    export_path_test = os.path.join("results", "cloud_datasets", f"{bundle_te_name}.pt")
                    export_to_cloud_bundle(test_loaders, export_path_test)
                    console.print(f"[bold green]✔ Test Dataset successfully sealed at:[/bold green] {export_path_test}")
                
                if action_choice == "Export to Cloud (.pt)":
                    console.print("\n[yellow]Skipping Local Training. Exiting.[/yellow]")
                    return

    # Final model setup
    if delta_choice:
        model = MLPDelta(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        ).to(device)
    else:
        model = MLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        ).to(device)

    # Train model if training splits are present
    if "train" in loaders and "val" in loaders:
        model, history = train(
            model,
            loaders,
            config,
            device,
            delta=delta_choice,
            target_scaler=target_scaler
        )
        plot_history(history)
        plot_history2(history)
        
        # Save Model and History automatically
        import time, json
        delta_str = "delta" if delta_choice else "nodelta"
        timestamp = int(time.time())
        default_base_name = f"MLP_{delta_str}_{timestamp}"
        
        custom_base_name = questionary.text(f"Enter a base name for the model results (leave blank for: {default_base_name}):").ask()
        base_name = custom_base_name if custom_base_name else default_base_name
        
        config["delta"] = delta_choice
        
        models_dir = os.path.join("results", "models")
        histories_dir = os.path.join("results", "histories")
        configs_dir = os.path.join("results", "configs")
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(histories_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"{base_name}.pth")
        history_path = os.path.join(histories_dir, f"{base_name}_history.json")
        config_path = os.path.join(configs_dir, f"{base_name}_config.json")
        plots_dir = os.path.join("results", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"{base_name}_curve.png")
        
        torch.save(model.state_dict(), model_path)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        plt.title(f"Training Convergence - {base_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
            
        console.print(f"\n[bold green]✔ Model saved safely to:[/bold green] {model_path}")
        console.print(f"[bold green]✔ History saved safely to:[/bold green] {history_path}")
        console.print(f"[bold green]✔ Config saved safely to:[/bold green] {config_path}")
        console.print(f"[bold green]✔ Curve Plotted safely to:[/bold green] {plot_path}\n")
        
    # Execute Test Phase if explicitly requested
    if "test" in loaders_config or "test" in loaders:
        test_model(
            model, 
            DATA_DIR, 
            loaders_config=loaders_config, 
            test_loader=loaders.get("test"), 
            device=device, 
            delta=delta_choice if 'delta_choice' in locals() else False,
            base_name=base_name if 'base_name' in locals() else None,
            target_scaler=target_scaler
        )

if __name__ == "__main__":
    main()