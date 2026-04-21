import os
import glob
import itertools
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import questionary
import time
import json
import re
import argparse
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt

from model.data_loader import get_dataloaders, get_cloud_dataloaders, export_to_cloud_bundle, ARSDataset
from model.scaler import PropertyScaler
from model.model import MLP, MLPDelta, MLPFiLM, MLPARS, DeltaGRNMLP, GRNFiLMMLP, GRNARSMLP
from config import config
from model.train import train
from model.evaluate import test
from utils.kaggle_gen import generate_kaggle_bundler, generate_kaggle_runner, generate_kaggle_plotter

console = Console()

# ==========================================
# HELPER UTILITIES
# ==========================================

def ask_with_default(msg, def_val):
    console.print(f"\n[bold white]?[/bold white] {msg} [yellow]({def_val})[/yellow]: ", end="")
    try:
        val = input().strip()
    except (EOFError, KeyboardInterrupt):
        return def_val
    return val if val else def_val

def get_file_len(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return sum(1 for _ in f) - 1
    return 0

def get_fraction_from_filename(filename):
    if "secondarytest" in filename: return 1.0
    if "test" in filename and "secondarytest" not in filename: return 1.0
    match = re.search(r'_([0-9]+\.[0-9]+)', filename)
    if match: return float(match.group(1))
    if re.search(r'_1[_\.]', filename): return 1.0
    return 0.0

def plot_history(history: dict, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss (Physical)")
    plt.plot(history["val_loss"], label="Val Loss (Physical)")
    if "train_eval_loss" in history:
        plt.plot(history["train_eval_loss"], label="Train Eval (Physical)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Physical Units)")
    plt.title("Training History")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ==========================================
# MODULAR WIZARDS
# ==========================================

def dataset_builder_wizard(data_dir: str, splits=None):
    """ Interactive setup for creating NEW data loaders from raw CSVs """
    console.print("\n[bold magenta]--- Dataset Setup Wizard ---[/bold magenta]")
    dataset_dir = os.path.join(data_dir, "dataset")
    indices_dir = os.path.join(data_dir, "indices")
    
    if splits is None: splits = ["train", "val"]
    
    available_targets = [os.path.basename(f) for f in glob.glob(os.path.join(dataset_dir, "*.csv")) if "noise" not in f]
    if not available_targets:
        available_targets = ["groupadditivity_1.csv", "groupadditivity_secondarytest.csv", "groupadditivity_test.csv"]
        
    loaders_config = {}
    for split in splits:
        console.print(f"\n[bold cyan]Configuring {split.upper()} Dataset[/bold cyan]")
        def_target = available_targets[0]
        if split == "train" and "groupadditivity_1.csv" in available_targets: def_target = "groupadditivity_1.csv"
        if split == "val" and "groupadditivity_secondarytest.csv" in available_targets: def_target = "groupadditivity_secondarytest.csv"
        if split == "test" and "groupadditivity_test.csv" in available_targets: def_target = "groupadditivity_test.csv"
        
        target_name = questionary.select(f"Select Target Master Dictionary for {split.upper()}:", choices=available_targets, default=def_target).ask()
        target_path = os.path.join(dataset_dir, target_name)
        indices_path = os.path.join(indices_dir, "indices_" + target_name.split('_')[-1] if '_' in target_name else "indices_" + target_name)
        
        max_mols = get_file_len(target_path) or 1000
        total_molecules = IntPrompt.ask(f"How many total molecules to use? (Max: {max_mols})", default=max_mols)
        master_fraction_ratio = float(total_molecules) / max_mols if max_mols > 0 else 1.0
        
        gen_mode = questionary.select("Select noise generation mode", choices=["precomputed", "on-the-fly"], default="precomputed").ask()
        
        if gen_mode == "precomputed":
            noise_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and "noise" in d.lower()]
            active_noise = questionary.checkbox("Select noise levels to include:", choices=noise_dirs).ask() or []
            
            dataset_blocks = []
            if active_noise:
                fully_random = questionary.confirm("Distribute perfectly equally automatically?", default=False).ask()
                remaining = 1.0
                
                for i, d in enumerate(active_noise):
                    if fully_random:
                        pct = 1.0 / len(active_noise)
                    else:
                        if i == len(active_noise) - 1:
                            pct = remaining
                            console.print(f"[green]Auto-assigning remaining {pct*100:.1f}% to '{d}'.[/green]")
                        else:
                            pct_str = Prompt.ask(f"Fraction for '{d}'? (0.0 to {remaining:.2f})", default=str(round(remaining / (len(active_noise)-i), 2)))
                            pct = float(pct_str)
                            if pct > remaining: pct = remaining
                            remaining -= pct

                    req_fraction = master_fraction_ratio
                    console.print(f"Targeting whole required set for noise '{d}' (~{req_fraction:.3f})")
                    
                    noise_dir_path = os.path.join(dataset_dir, d)
                    avail_nf = [f for f in os.listdir(noise_dir_path) if f.endswith(".csv")]
                    if "secondarytest" in target_name: avail_nf = [f for f in avail_nf if "secondarytest" in f]
                    elif "test" in target_name: avail_nf = [f for f in avail_nf if "test" in f and "secondarytest" not in f]
                    else: avail_nf = [f for f in avail_nf if "test" not in f]
                    
                    file_fractions = [(f, get_fraction_from_filename(f)) for f in avail_nf]
                    file_fractions = [f for f in file_fractions if f[1] > 0]
                    
                    if not file_fractions:
                        console.print(f"  [red]Warning: No appropriately matched noise files found in '{d}'![/red]")
                        continue

                    best_subset = []
                    min_diff = float('inf')
                    limit_k = min(len(file_fractions) + 1, 4)
                    for k in range(1, limit_k):
                        found_k_improvement = False
                        for combo in itertools.combinations(file_fractions, k):
                            combo_sum = sum(c[1] for c in combo)
                            diff = abs(combo_sum - req_fraction)
                            if diff < min_diff * 0.95:
                                min_diff = diff
                                best_subset = combo
                                found_k_improvement = True
                        if not found_k_improvement and k > 1: break
                    
                    if best_subset:
                        for nf, f_val in best_subset:
                            clean_fn = f"groupadditivity_{target_name.split('_')[-1]}" if '_' in target_name else target_name
                            dataset_blocks.append({
                                "clean_path": os.path.join(dataset_dir, clean_fn),
                                "noisy_path": os.path.join(noise_dir_path, nf),
                                "indices_path": indices_path
                            })
                            console.print(f"  [cyan]+ Picked File:[/cyan] {nf} (fraction {f_val})")
            
            loaders_config[split] = {"mode": "precomputed", "target_path": target_path, "indices_path": indices_path, "total_molecules": total_molecules, "blocks": dataset_blocks}
        else:
            std = float(Prompt.ask("Enter Gaussian noise standard deviation", default="0.01"))
            loaders_config[split] = {"mode": "on-the-fly", "target_path": target_path, "indices_path": indices_path, "total_molecules": total_molecules, "noise_std": std}
            
    return loaders_config

def get_data_wizard(data_dir: str, splits=None):
    """ Unified Data Loader Wizard: Bundle vs Manual """
    if splits is None: splits = ["train", "val"]
    
    msg = f"Provide {', '.join(splits)} dataset from an already ready bundle (.pt)?"
    use_bundle = questionary.confirm(msg, default=False).ask()
    
    loaders = {}
    bundle_scaler = None
    loaders_config = {}

    if use_bundle:
        bundle_files = glob.glob(os.path.join("results", "cloud_datasets", "*.pt"))
        if not bundle_files:
            console.print("[yellow]No '.pt' bundles found. Falling back to manual setup.[/yellow]")
            use_bundle = False
        else:
            chosen = questionary.select("Choose a dataset bundle:", choices=bundle_files).ask()
            loaders.update(get_cloud_dataloaders(chosen))
            bundle = torch.load(chosen, weights_only=False)
            if "metadata" in bundle:
                bundle_scaler = PropertyScaler.from_dict(bundle["metadata"])
            loaders_config = {s: True for s in loaders.keys()}

    if not use_bundle:
        loaders_config = dataset_builder_wizard(data_dir, splits=splits)
        # Handle scaling selection here if it's meant to be used with get_dataloaders_scaled
    
    return loaders, bundle_scaler, loaders_config

def get_config_wizard(loaders, delta_default=False, existing_scaler=None):
    """ Choose Architecture, Scaling and Learning Objectives """
    console.print("\n[bold magenta]--- Model & Scaling Configuration ---[/bold magenta]")
    # 2. Activation Choice
    activation_type = questionary.select(
        "Select Activation Function (prevents vanishing signal):",
        choices=["GELU", "ELU", "ReLU"],
        default="GELU"
    ).ask()

    # 3. Model Strategy Configuration
    strategy = questionary.select(
        "Select Model Strategy:",
        choices=[
            questionary.Choice("Hybrid GRN-FiLM (Modulated)", value="grn-film"),
            questionary.Choice("Hybrid GRN-ARS (Adaptive Routing)", value="grn-ars"),
            questionary.Choice("Gated Residual Network (GRN Delta)", value="grn"),
            questionary.Choice("Adaptive Residual Scaling (Dual-Head ARS)", value="ars"),
            questionary.Choice("FiLM Delta Learning (Feature-wise Modulation)", value="film"),
            questionary.Choice("Standard Delta Learning (Residual)", value="delta"),
            questionary.Choice("Standard Regression (No Delta)", value="none")
        ],
        default="grn-ars"
    ).ask()

    if strategy == "grn-film":
        model_type = "GRNFiLMMLP"
        delta_choice = True
    elif strategy == "grn-ars":
        model_type = "GRNARSMLP"
        delta_choice = True
    elif strategy == "grn":
        model_type = "DeltaGRNMLP"
        delta_choice = True
    elif strategy == "ars":
        model_type = "MLPARS"
        delta_choice = True
    elif strategy == "delta":
        model_type = "MLPDelta"
        delta_choice = True
    elif strategy == "film":
        model_type = "MLPFiLM"
        delta_choice = True
    else:
        model_type = "MLP"
        delta_choice = False

    # 4. Scaling Strategy
    choices = [
        questionary.Choice("Delta Scaling (Signal Amplification)", value="delta"),
        questionary.Choice("Standard Scaling (Independent)", value="standard"),
        questionary.Choice("MinMax Scaling (Bound to [0,1])", value="minmax"),
        questionary.Choice("None (Raw Physical Units)", value="none")
    ]
    if existing_scaler:
        choices.insert(0, questionary.Choice(f"Use Bundle's Scaler ({existing_scaler.mode})", value="original"))

    scaling_mode = questionary.select(
        "Select Scaling Strategy:",
        choices=choices,
        default="delta" if delta_choice else "standard"
    ).ask()
    
    return scaling_mode, delta_choice, model_type, activation_type

# ==========================================
# EXECUTION MODES
# ==========================================

def test_flow(device, data_dir, model=None, model_path=None, delta=False, scaler=None):
    """ Reusable Test Execution Branch """
    if model is None and model_path:
        base = os.path.basename(model_path).replace(".pth", "")
        conf_path = os.path.join("results", "configs", f"{base}_config.json")
        run_config = config
        if os.path.exists(conf_path):
            with open(conf_path, 'r') as f: run_config = json.load(f)
        
        m_type = run_config.get("model_type", "MLP")
        h_dims = run_config.get("hidden_dims", [256, 64])
        drop = run_config.get("dropout", 0.4)
        act = run_config.get("activation_type", "GELU")
        if m_type == "GRNFiLMMLP":
            model = GRNFiLMMLP(run_config["input_dim"], h_dims, drop, act)
        elif m_type == "GRNARSMLP":
            model = GRNARSMLP(run_config["input_dim"], h_dims, drop, act)
        elif m_type == "DeltaGRNMLP":
            model = DeltaGRNMLP(run_config["input_dim"], h_dims, drop, act)
        elif m_type == "MLPARS":
            model = MLPARS(run_config["input_dim"], h_dims, drop, act)
        elif m_type == "MLPFiLM":
            model = MLPFiLM(run_config["input_dim"], h_dims, drop, act)
        elif m_type == "MLPDelta" or delta:
            model = MLPDelta(run_config["input_dim"], h_dims, drop, act)
        else:
            model = MLP(run_config["input_dim"], h_dims, drop, act)
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)

    # 1. Get Data
    console.print("\n[bold magenta]--- Test Dataset Setup ---[/bold magenta]")
    loaders, bundle_scaler, loaders_config = get_data_wizard(data_dir, splits=["test"])
    
    # Handling manual scaling if not from bundle
    if not loaders and loaders_config:
        scaling_mode, delta, model_type, activation_type = get_config_wizard(None, delta_default=delta, existing_scaler=bundle_scaler)
        if scaling_mode != "none":
             for split in loaders_config: loaders_config[split]["scaling"] = scaling_mode
             loaders, scaler = get_dataloaders_scaled(loaders_config)
        else:
             loaders = get_dataloaders(loaders_config)
             scaler = None

    test_loader = loaders.get("test") or loaders.get("val") or (list(loaders.values())[0] if loaders else None)
    if not test_loader: return console.print("[red]No test loader available.[/red]")

    # 2. Results
    console.print("\n[bold cyan]--- Running Evaluation ---[/bold cyan]")
    criterion = nn.MSELoss()
    loss = test(model, test_loader, criterion, device, delta=delta, scaler=scaler)
    
    base_name = f"manual_test_{int(time.time())}"
    os.makedirs(os.path.join("results", "tests"), exist_ok=True)
    test_path = os.path.join("results", "tests", f"{base_name}_test.json")
    with open(test_path, 'w') as f:
        json.dump({"test_loss": loss, "delta": delta, "scaler": "none" if scaler is None else "active"}, f, indent=4)
    
    return loss

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud_bundle", type=str, default=None)
    parser.add_argument("--test_bundle", type=str, default=None)
    args = parser.parse_args()
    
    DATA_DIR = config["data_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.cloud_bundle or args.test_bundle:
        return console.print("[yellow]Headless mode not fully implemented in this resolved version.[/yellow]")

    console.print("\n[bold magenta]=== MOLECULAR PROPERTY PREDICTION PIPELINE ===[/bold magenta]")
    mode = questionary.select("What would you like to do?", choices=["Train a New Model", "Test an Existing Model", "Kaggle Tools", "Exit"]).ask()
    
    if mode == "Exit" or mode is None: return

    if mode == "Test an Existing Model":
        model_files = glob.glob(os.path.join("results", "models", "*.pth"))
        if not model_files: return console.print("[red]No models found in results/models/![/red]")
        path = questionary.select("Select model:", choices=model_files).ask()
        delta = questionary.confirm("Was this model trained with Delta Learning?", default=False).ask()
        test_flow(device, DATA_DIR, model_path=path, delta=delta)

    elif mode == "Kaggle Tools":
        exec_kaggle_mode(DATA_DIR, config)

    elif mode == "Train a New Model":
        # 1. Wizard Configuration
        loaders, bundle_scaler, loaders_config = get_data_wizard(DATA_DIR, splits=["train", "val"])
        scaling_mode, delta, model_type, activation_type = get_config_wizard(loaders, existing_scaler=bundle_scaler)
        
        # 2. Final Data Loading (if manual)
        target_scaler = bundle_scaler
        if not loaders and loaders_config:
            if scaling_mode != "none":
                for split in loaders_config: loaders_config[split]["scaling"] = scaling_mode
                loaders, target_scaler = get_dataloaders_scaled(loaders_config)
            else:
                loaders = get_dataloaders(loaders_config)
                target_scaler = None

        # 3. Model Init
        config["model_type"] = model_type
        config["activation_type"] = activation_type
        
        if model_type == "GRNFiLMMLP":
            model = GRNFiLMMLP(config["input_dim"], config["hidden_dims"], config["dropout"], activation_type).to(device)
        elif model_type == "GRNARSMLP":
            model = GRNARSMLP(config["input_dim"], config["hidden_dims"], config["dropout"], activation_type).to(device)
        elif model_type == "DeltaGRNMLP":
            model = DeltaGRNMLP(config["input_dim"], config["hidden_dims"], config["dropout"], activation_type).to(device)
        elif model_type == "MLPARS":
            model = MLPARS(config["input_dim"], config["hidden_dims"], config["dropout"], activation_type).to(device)
        elif model_type == "MLPFiLM":
            model = MLPFiLM(config["input_dim"], config["hidden_dims"], config["dropout"], activation_type).to(device)
        elif model_type == "MLPDelta":
            model = MLPDelta(config["input_dim"], config["hidden_dims"], config["dropout"], activation_type).to(device)
        else:
            model = MLP(config["input_dim"], config["hidden_dims"], config["dropout"], activation_type).to(device)

        # 4. Action
        action = questionary.select("Configuration ready. Next step?", choices=["Train Locally", "Export Dataset (.pt)", "Both", "Cancel"]).ask()
        if action == "Cancel" or action is None: return

        if action in ["Train Locally", "Both"]:
            model, history = train(model, loaders, config, device, delta=delta, scaler=target_scaler)
            
            timestamp = int(time.time())
            base_name = ask_with_default("Base name for results", f"MLP_{'delta' if delta else 'reg'}_{timestamp}")
            
            os.makedirs(os.path.join("results", "models"), exist_ok=True)
            os.makedirs(os.path.join("results", "configs"), exist_ok=True)
            os.makedirs(os.path.join("results", "histories"), exist_ok=True)
            os.makedirs(os.path.join("results", "plots"), exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join("results", "models", f"{base_name}.pth"))
            with open(os.path.join("results", "configs", f"{base_name}_config.json"), 'w') as f:
                json.dump(config, f, indent=4)
            with open(os.path.join("results", "histories", f"{base_name}_history.json"), 'w') as f:
                json.dump(history, f, indent=4)
            
            plot_history(history, save_path=os.path.join("results", "plots", f"{base_name}_curve.png"))
            console.print(f"\n[bold green]✔ Results saved for:[/bold green] {base_name}")

            if questionary.confirm("Run Test Evaluation now?", default=True).ask():
                test_flow(device, DATA_DIR, model=model, delta=delta, scaler=target_scaler)

        if action in ["Export Dataset (.pt)", "Both"]:
            name = ask_with_default("Name for .pt bundle:", f"bundle_{int(time.time())}")
            export_path = os.path.join("results", "cloud_datasets", f"{name}.pt")
            export_to_cloud_bundle(loaders, export_path)

def exec_kaggle_mode(data_dir, config):
    kaggle_choice = questionary.select(
        "Which Kaggle script do you want to generate?",
        choices=["Dataset BUNDLER", "Training RUNNER", "Standalone PLOTTER"]
    ).ask()
    if not kaggle_choice: return

    username = ask_with_default("Kaggle Username", "username")
    
    if "BUNDLER" in kaggle_choice:
        loaders_config = dataset_builder_wizard(data_dir)
        script_content = generate_kaggle_bundler(loaders_config, username=username)
        path = os.path.join("results", "kaggle_scripts", f"kaggle_bundle_{int(time.time())}.py")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f: f.write(script_content)
        console.print(f"[green]✔ Generated at {path}[/green]")
    elif "RUNNER" in kaggle_choice:
        console.print("\n[bold cyan]--- Kaggle Runner Configuration ---[/bold cyan]")
        train_val_slug = ask_with_default("Train/Val Dataset Slug", "structural-noise-dataloaders")
        test_slug = ask_with_default("Test Dataset Slug", "structural-noise-dataloaders")
        script_slug = ask_with_default("Source Code Slug", "src-model")
        
        train_val_name = ask_with_default("Train/Val Bundle Name (.pt)", "scaffold_train_val.pt")
        test_name = ask_with_default("Test Bundle Name (.pt)", "scaffold_test.pt")
        
        strategy = questionary.select(
            "Select Model Strategy for Kaggle:",
            choices=[
                questionary.Choice("Hybrid GRN-FiLM", value="grn-film"),
                questionary.Choice("Hybrid GRN-ARS", value="grn-ars"),
                questionary.Choice("Gated Residual Network (GRN Delta)", value="grn"),
                questionary.Choice("Adaptive Residual Scaling", value="ars"),
                questionary.Choice("FiLM Delta Learning", value="film"),
                questionary.Choice("Standard Delta Learning", value="delta"),
                questionary.Choice("Standard Regression", value="none")
            ],
            default="grn-ars"
        ).ask()

        if strategy == "grn-film":
            model_type = "GRNFiLMMLP"
            delta = True
        elif strategy == "grn-ars":
            model_type = "GRNARSMLP"
            delta = True
        elif strategy == "grn":
            model_type = "DeltaGRNMLP"
            delta = True
        elif strategy == "ars":
            model_type = "MLPARS"
            delta = True
        elif strategy == "film":
            model_type = "MLPFiLM"
            delta = True
        elif strategy == "delta":
            model_type = "MLPDelta"
            delta = True
        else:
            model_type = "MLP"
            delta = False

        ars_choice = True if strategy == "ars" else False
        
        scaling_mode = questionary.select(
            "Select Scaling Strategy:",
            choices=[
                questionary.Choice("Delta Scaling", value="delta"),
                questionary.Choice("Standard Scaling", value="standard"),
                questionary.Choice("MinMax Scaling", value="minmax"),
                questionary.Choice("None", value="none")
            ],
            default="delta" if delta else "standard"
        ).ask()

        act_choice = questionary.select(
            "Select Activation Function:",
            choices=["GELU", "ELU", "ReLU", "SiLU"],
            default="GELU"
        ).ask()

        # Final Settings
        base_name = f"kaggle_run_{int(time.time())}"
        

        script_content = generate_kaggle_runner(
            {}, config, delta, ars_choice=ars_choice, model_type=model_type,
            scaling_mode=scaling_mode, username=username,
            train_val_slug=train_val_slug, test_slug=test_slug, script_slug=script_slug,
            train_val_name=train_val_name, test_name=test_name,
            base_name=base_name, activation_type=act_choice
        )
        
        path = os.path.join("results", "kaggle_scripts", f"kaggle_runner_{int(time.time())}.py")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f: f.write(script_content)
        console.print(f"[green]✔ Generated at {path}[/green]")
        
    else: # Standalone PLOTTER
        script_content = generate_kaggle_plotter(username=username)
        path = os.path.join("results", "kaggle_scripts", f"kaggle_plotter_{int(time.time())}.py")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f: f.write(script_content)
        console.print(f"[green]✔ Generated at {path}[/green]")

if __name__ == "__main__":
    main()