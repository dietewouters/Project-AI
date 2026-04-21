import os
import json
import pprint

def generate_kaggle_bundler(loaders_config, username="username", dataset_name="project-data", script_name="src-model", train_val_name="kaggle_train_val.pt", test_name="kaggle_test.pt"):
    """
    Generates a Python script for Kaggle that:
    1. Organizes raw data folders into a unified structure.
    2. Packages selected datasets into .pt bundles for training.
    """
    import copy
    # Sanitize config for Kaggle: convert absolute paths to basenames
    sc = copy.deepcopy(loaders_config)
    for split, conf in sc.items():
        if isinstance(conf, dict):
            if "target_path" in conf:
                conf["target_path"] = os.path.basename(conf["target_path"])
            if "indices_path" in conf:
                conf["indices_path"] = os.path.basename(conf["indices_path"])
            if "blocks" in conf:
                for block in conf["blocks"]:
                    for key in ["clean_path", "noisy_path", "indices_path"]:
                        if key in block:
                            p = block[key]
                            if "dataset/" in p:
                                block[key] = p.split("dataset/")[-1]
                            elif "indices/" in p:
                                block[key] = p.split("indices/")[-1]
                            else:
                                block[key] = os.path.basename(p)

    config_final = pprint.pformat(sc, indent=4)

    template = f'''# =================================================================
# KAGGLE DATASET BUNDLER SCRIPT
# Purpose: Merge separate raw data folders and export .pt bundles.
# =================================================================
import os
import sys
import torch
import shutil
import time

# --- DYNAMIC CONFIGURATION (Set by Wizard) ---
KAGGLE_USERNAME = "{username}"
DATASET_NAME = "{dataset_name}"
SCRIPT_DATASET_NAME = "{script_name}"

# Output Bundle Names
TRAIN_VAL_BUNDLE = "{train_val_name}"
TEST_BUNDLE = "{test_name}"

# Paths
BASE_KAGGLE_PATH = f"/kaggle/input/datasets/{{KAGGLE_USERNAME}}/{{DATASET_NAME}}"
IF_NOT_DATASETS = f"/kaggle/input/{{DATASET_NAME}}"
MODEL_SLUG_PATH = f"/kaggle/input/dataset/{{KAGGLE_USERNAME}}/{{SCRIPT_DATASET_NAME}}"

# Determine true DATA_BASE
DATA_BASE_ROOT = BASE_KAGGLE_PATH if os.path.exists(BASE_KAGGLE_PATH) else IF_NOT_DATASETS
# Use the 'data' subfolder if it exists (highly likely based on working scripts)
DATA_BASE = os.path.join(DATA_BASE_ROOT, "data") if os.path.exists(os.path.join(DATA_BASE_ROOT, "data")) else DATA_BASE_ROOT
print(f"[+] Using DATA_BASE: {{DATA_BASE}}")

# 1. ORGANIZE DIRECTORY STRUCTURE (SYMLINKS)
DATA_ROOT = "/kaggle/working/data"
PROJECT_DATA = os.path.join(DATA_ROOT, "groupadditivity")
os.makedirs(os.path.join(PROJECT_DATA, "dataset", "noise0.01"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DATA, "indices"), exist_ok=True)
os.makedirs(os.path.join(DATA_ROOT, "processed_features"), exist_ok=True)

# 2. PATH INDEXING & DIAGNOSTICS
print("\\n" + "="*50)
print("KAGGLE DIAGNOSTICS PROBE")
print("="*50)
print(f"Current Working Directory: {{os.getcwd()}}")
print(f"Python sys.path: {{sys.path}}")

SOURCE_ROOT = None
# Search specifically for the folder containing 'model/' and 'config.py'
print("[+] Searching for primary source root...")

# Prioritize the specified script dataset path
if os.path.exists(MODEL_SLUG_PATH):
    sys.path.append(MODEL_SLUG_PATH)
    print(f"✅ Found source code in model slug: {{MODEL_SLUG_PATH}}")
    SOURCE_ROOT = MODEL_SLUG_PATH

if not SOURCE_ROOT:
    for root_node, dirs, files in os.walk('/kaggle/input'):
        if 'model' in dirs and 'config.py' in files:
            SOURCE_ROOT = root_node
            if SOURCE_ROOT not in sys.path:
                sys.path.insert(0, SOURCE_ROOT)
            print("✅ Success! Source root discovered at: " + str(SOURCE_ROOT))
            break

if not SOURCE_ROOT:
    print("⚠️ Warning: Could not find 'model/' package automatically.")
print("="*50 + "\\n")

from model.data_loader import get_dataloaders, export_to_cloud_bundle

# 3. CONFIGURE & LOAD DATA
# Configuration (Only using the Wizard choices)
raw_config = {config_final}

# Reconstruct full Kaggle paths
loaders_config = {{}}
for split, conf in raw_config.items():
    c = conf.copy()
    # Deep Search: Check for groupadditivity or groupadditivity_h298 nesting
    def find_p(base, sub, fn):
        for nested in ["groupadditivity", "groupadditivity_h298"]:
            p = os.path.join(base, nested, sub, fn)
            if os.path.exists(p): return p
        return os.path.join(base, sub, fn)

    if "target_path" in c:
        c["target_path"] = find_p(DATA_BASE, "dataset", c["target_path"])
    if "indices_path" in c:
        c["indices_path"] = find_p(DATA_BASE, "indices", c["indices_path"])
    if "blocks" in c:
        new_blocks = []
        for b in c["blocks"]:
            nb = b.copy()
            if "clean_path" in nb: nb["clean_path"] = find_p(DATA_BASE, "dataset", nb["clean_path"])
            if "noisy_path" in nb: nb["noisy_path"] = find_p(DATA_BASE, "dataset", nb["noisy_path"])
            if "indices_path" in nb: nb["indices_path"] = find_p(DATA_BASE, "indices", nb["indices_path"])
            new_blocks.append(nb)
        c["blocks"] = new_blocks
    loaders_config[split] = c

print("\\n[+] Parsing raw files and generating sparse signatures...")
# Features also might be nested
FEAT_BASE = None
for nested in ["groupadditivity", "groupadditivity_h298"]:
    p = os.path.join(DATA_BASE, nested, "processed_features")
    if os.path.exists(p):
        FEAT_BASE = p
        break
if not FEAT_BASE:
    FEAT_BASE = os.path.join(DATA_BASE, "processed_features")

loaders = get_dataloaders(loaders_config, features_dir=FEAT_BASE)

# 4. EXPORT TO BUNDLES
BUNDLE_DIR = "/kaggle/working/results/cloud_datasets"
os.makedirs(BUNDLE_DIR, exist_ok=True)

# Split 1: Train/Val Bundle
train_val_loaders = {{k: v for k, v in loaders.items() if k in ['train', 'val']}}
if train_val_loaders:
    export_path_tv = os.path.join(BUNDLE_DIR, TRAIN_VAL_BUNDLE)
    print(f"[+] Sealing Train/Val Tensors into bundle: {{export_path_tv}}...")
    export_to_cloud_bundle(train_val_loaders, export_path_tv)

# Split 2: Test Bundle
test_loaders = {{k: v for k, v in loaders.items() if k == 'test'}}
if test_loaders:
    export_path_test = os.path.join(BUNDLE_DIR, TEST_BUNDLE)
    print(f"[+] Sealing Test Tensors into bundle: {{export_path_test}}...")
    export_to_cloud_bundle(test_loaders, export_path_test)

print(f"\\n[✔] COMPLETE! Files created in {{BUNDLE_DIR}}:")
if train_val_loaders: print(f" - Train/Val: {{TRAIN_VAL_BUNDLE}}")
if test_loaders:      print(f" - Test:      {{TEST_BUNDLE}}")
'''
    return template

def generate_kaggle_runner(loaders_config, global_config, delta_choice, ars_choice=False, model_type="MLPDelta", scaling_mode="delta", username="username", train_val_slug="project-data", test_slug="project-data", script_slug="src-model", train_val_name="kaggle_train_val.pt", test_name="kaggle_test.pt", do_test=True, base_name="model_run", activation_type="GELU"):
    """
    Generates a streamlined Python script for Kaggle training.
    """
    import pprint
    config_final = pprint.pformat(global_config, indent=4)
    
    template = f'''# =================================================================
# KAGGLE MODEL TRAINING SCRIPT (STREAMLINED)
# =================================================================
import os
import sys
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import numpy as np

# --- DYNAMIC CONFIGURATION ---
KAGGLE_USERNAME = "{username}"
TRAIN_VAL_SLUG = "{train_val_slug}"
TEST_SLUG = "{test_slug}"
SCRIPT_SLUG = "{script_slug}"
TRAIN_VAL_BUNDLE = "{train_val_name}"
TEST_BUNDLE = "{test_name}"
BASE_NAME = "{base_name}"
DELTA_CHOICE = {delta_choice}
MODEL_TYPE = "{model_type}"

SCALING_MODE = "{scaling_mode}"

# Paths
TRAIN_VAL_PATH = f"/kaggle/input/datasets/{{KAGGLE_USERNAME}}/{{TRAIN_VAL_SLUG}}/results/cloud_datasets/{{TRAIN_VAL_BUNDLE}}"
if not os.path.exists(TRAIN_VAL_PATH):
    TRAIN_VAL_PATH = f"/kaggle/input/{{TRAIN_VAL_SLUG}}/results/cloud_datasets/{{TRAIN_VAL_BUNDLE}}"

TEST_PATH = f"/kaggle/input/datasets/{{KAGGLE_USERNAME}}/{{TEST_SLUG}}/results/cloud_datasets/{{TEST_BUNDLE}}"
if not os.path.exists(TEST_PATH):
    TEST_PATH = f"/kaggle/input/{{TEST_SLUG}}/results/cloud_datasets/{{TEST_BUNDLE}}"

SOURCE_PATH = f"/kaggle/input/datasets/{{KAGGLE_USERNAME}}/{{SCRIPT_SLUG}}"
if not os.path.exists(SOURCE_PATH):
    SOURCE_PATH = f"/kaggle/input/{{SCRIPT_SLUG}}"

# 1. SETUP ENVIRONMENT
if SOURCE_PATH not in sys.path: sys.path.append(SOURCE_PATH)
# Deep search for the 'src' folder containing 'model/'
found_src = False
for root, dirs, files in os.walk(SOURCE_PATH):
    if 'model' in dirs and 'config.py' in files:
        if root not in sys.path: sys.path.insert(0, root)
        found_src = True
        break
if not found_src:
    print(f"[!] Warning: Could not find 'model' folder in {{SOURCE_PATH}}. Imports may fail.")

# 2. IMPORTS
import torch
import torch.nn as nn
import numpy as np
import json
import os
from model.model import MLP, MLPDelta, MLPFiLM, MLPARS, DeltaGRNMLP, GRNFiLMMLP, GRNARSMLP
from model.train import train_one_epoch
from model.evaluate import evaluate
from model.scaler import PropertyScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- DATA LOADING WRAPPER (Kaggle Specific) ---
class SafeDataLoader:
    def __init__(self, dataloader): self.dataloader = dataloader; self.dataset = dataloader.dataset
    def __iter__(self):
        for fp, noisy, y_true in self.dataloader:
            if fp.dtype == torch.uint8:
                # Bit-unpack 1024-bit fingerprints on the fly
                fp = torch.from_numpy(np.unpackbits(fp.cpu().numpy(), axis=-1).astype(np.float32)).to(device)
            yield fp, noisy, y_true
    def __len__(self): return len(self.dataloader)

# --- INITIALIZATION ---
TRAIN_VAL_PATH = f"/kaggle/input/{train_val_slug}/{train_val_name}"
print(f"[+] Loading bundle: {{TRAIN_VAL_PATH}}...")

from model.data_loader import get_cloud_dataloaders
raw_loaders = get_cloud_dataloaders(TRAIN_VAL_PATH)
loaders = {{k: SafeDataLoader(v) for k, v in raw_loaders.items()}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL_TYPE == "GRNFiLMMLP":
    model = GRNFiLMMLP(1025, activation_type="{activation_type}")
elif MODEL_TYPE == "GRNARSMLP":
    model = GRNARSMLP(1025, activation_type="{activation_type}")
elif MODEL_TYPE == "DeltaGRNMLP":
    model = DeltaGRNMLP(1025, activation_type="{activation_type}")
elif MODEL_TYPE == "MLPARS":
    model = MLPARS(1025, activation_type="{activation_type}")
elif MODEL_TYPE == "MLPFiLM":
    model = MLPFiLM(1025, activation_type="{activation_type}")
elif MODEL_TYPE == "MLPDelta":
    model = MLPDelta(1025, activation_type="{activation_type}")
else:
    model = MLP(1025, activation_type="{activation_type}")
model.to(device)

scaler = PropertyScaler(mode="{scaling_mode}")
if 'train' in loaders:
    scaler.fit(loaders['train'].dataset.noisy, loaders['train'].dataset.y_true)

# --- TRAINING LOOP ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
criterion = nn.MSELoss()
history = {{"train_loss": [], "val_loss": []}}

print(f"[+] Starting {{MODEL_TYPE}} training...")
for epoch in range(15):
    t_loss = train_one_epoch(model, loaders['train'], optimizer, criterion, device, delta=DELTA_CHOICE, scaler=scaler)
    v_loss = evaluate(model, loaders['val'], criterion, device, delta=DELTA_CHOICE, scaler=scaler)
    scheduler.step(v_loss)
    history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
    print(f"Epoch {{epoch+1:02d}} | Train: {{t_loss:.6f}} | Val: {{v_loss:.6f}}")

# --- SAVE RESULTS ---
res_dir = "/kaggle/working/results"
for d in ["models", "histories"]: os.makedirs(os.path.join(res_dir, d), exist_ok=True)
torch.save(model.state_dict(), os.path.join(res_dir, "models", f"{base_name}.pt"))
with open(os.path.join(res_dir, "histories", f"{base_name}_history.json"), "w") as f: json.dump(history, f)
print(f"✅ Results saved to {{res_dir}}")
'''
    return template

def generate_kaggle_plotter(username="username"):
    """
    Generates a standalone Python script for Kaggle that parses history files
    and generates custom plots based on user arguments.
    """
    template = f'''# =================================================================
# KAGGLE POST-PROCESSING: PLOTTING TOOL
# Purpose: Generate custom plots from saved training histories.
# Usage: !python plotter.py --history results/histories/run1_history.json --metric val --name model_a_val
# =================================================================
import os
import json
import argparse
import matplotlib.pyplot as plt

def plot_history(history, metric, output_path, base_name):
    plt.figure(figsize=(10,6))
    
    if metric == 'both':
        if 'train_loss' in history:
            plt.plot(history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2, marker='o', markersize=4, alpha=0.8)
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2, marker='s', markersize=4, alpha=0.8)
        if 'train_eval_loss' in history:
            plt.plot(history['train_eval_loss'], label='Train Eval (Original Scale)', color='#2ca02c', linestyle='--', alpha=0.6)
        title = f"Training Convergence - {{base_name}}"
    elif metric == 'train':
        plt.plot(history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2.5)
        title = f"Training Loss - {{base_name}}"
    elif metric == 'val':
        plt.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2.5)
        title = f"Validation Loss - {{base_name}}"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add info box
    best_val = min(history['val_loss']) if 'val_loss' in history and history['val_loss'] else "N/A"
    textstr = f"Best Val Loss: {{best_val}}" if isinstance(best_val, str) else f"Best Val Loss: {{best_val:.6f}}"
    plt.gcf().text(0.15, 0.02, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved to: {{output_path}}")

def main():
    parser = argparse.ArgumentParser(description="Kaggle Plotting Utility")
    parser.add_argument("--history", type=str, help="Path to the history JSON file")
    parser.add_argument("--metric", type=str, choices=['train', 'val', 'both'], default='both', help="Which metric to plot")
    parser.add_argument("--name", type=str, default=None, help="Custom name for the plot file")
    args = parser.parse_args()

    # 1. Discovery logic
    HISTORIES_DIR = "/kaggle/working/results/histories"
    PLOTS_DIR = "/kaggle/working/results/plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not args.history:
        # Auto-discover if none provided
        if os.path.exists(HISTORIES_DIR):
            files = [f for f in os.listdir(HISTORIES_DIR) if f.endswith(".json")]
            if not files:
                print(f"❌ No history files found in {{HISTORIES_DIR}}")
                return
            print("Available histories:")
            for i, f in enumerate(files):
                print(f"  [{{i}}] {{f}}")
            choice = input("\\nSelect index to plot (or 'all'): ").strip()
            if choice == 'all':
                to_plot = [os.path.join(HISTORIES_DIR, f) for f in files]
            else:
                try:
                    to_plot = [os.path.join(HISTORIES_DIR, files[int(choice)])]
                except:
                    print("Invalid choice.")
                    return
        else:
            print(f"❌ Directory {{HISTORIES_DIR}} not found.")
            return
    else:
        to_plot = [args.history]

    # 2. Execution
    for h_path in to_plot:
        if not os.path.exists(h_path):
            print(f"⚠️ File not found: {{h_path}}")
            continue
            
        with open(h_path, 'r') as f:
            history = json.load(f)
        
        base_name_val = os.path.basename(h_path).replace("_history.json", "").replace(".json", "")
        out_name = args.name if args.name else f"{{base_name_val}}_{{args.metric}}_curve.png"
        if not out_name.endswith(".png"): out_name += ".png"
        
        plot_path = os.path.join(PLOTS_DIR, out_name)
        plot_history(history, args.metric, plot_path, base_name_val)

if __name__ == "__main__":
    main()
'''
    return template
