import os
import json

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

    config_final = json.dumps(sc, indent=4)

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


def generate_kaggle_runner(loaders_config, global_config, delta_choice, username="username", train_val_slug="project-data", test_slug="project-data", script_slug="src-model", train_val_name="kaggle_train_val.pt", test_name="kaggle_test.pt", do_test=True, base_name="model_run"):
    """
    Generates a Python script for Kaggle that:
    1. Indexes source code.
    2. Loads pre-made .pt bundles.
    3. Initializes the model and runs training.
    4. Saves ALL results (model, config, history, plots) to /kaggle/working properly.
    """
    
    # Template for the Kaggle Training Runner
    template = f'''# =================================================================
# KAGGLE MODEL TRAINING SCRIPT
# Purpose: Train and evaluate the model using pre-made .pt bundles.
# =================================================================
import os
import sys
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

# --- DYNAMIC CONFIGURATION (Set by Wizard) ---
KAGGLE_USERNAME = "{username}"
TRAIN_VAL_SLUG = "{train_val_slug}"
TEST_SLUG = "{test_slug}"
SCRIPT_SLUG = "{script_slug}"

# Bundle Filenames
TRAIN_VAL_BUNDLE = "{train_val_name}"
TEST_BUNDLE = "{test_name}"

# Execution Settings
DO_TEST = {do_test}
BASE_NAME = "{base_name}"

# Paths
TRAIN_VAL_PATH = f"/kaggle/input/datasets/{{KAGGLE_USERNAME}}/{{TRAIN_VAL_SLUG}}/{{TRAIN_VAL_BUNDLE}}"
if not os.path.exists(TRAIN_VAL_PATH):
    TRAIN_VAL_PATH = f"/kaggle/input/{{TRAIN_VAL_SLUG}}/{{TRAIN_VAL_BUNDLE}}"

TEST_PATH = f"/kaggle/input/datasets/{{KAGGLE_USERNAME}}/{{TEST_SLUG}}/{{TEST_BUNDLE}}"
if not os.path.exists(TEST_PATH):
    TEST_PATH = f"/kaggle/input/{{TEST_SLUG}}/{{TEST_BUNDLE}}"

SOURCE_PATH = f"/kaggle/input/datasets/{{KAGGLE_USERNAME}}/{{SCRIPT_SLUG}}"
if not os.path.exists(SOURCE_PATH):
    SOURCE_PATH = f"/kaggle/input/{{SCRIPT_SLUG}}"

# 1. PATH INDEXING & DIAGNOSTICS
print("\\n" + "="*50)
print("KAGGLE DIAGNOSTICS PROBE")
print("="*50)
import os, sys # Re-importing inside just in case of any weird scope issues
print("Current Working Directory: " + os.getcwd())
print("Python sys.path: " + str(sys.path))
if os.path.exists('/kaggle/input'):
    print("Kaggle Inputs: " + str(os.listdir('/kaggle/input')))

SOURCE_ROOT = None
TRAIN_VAL_ROOT = os.path.dirname(TRAIN_VAL_PATH)
TEST_ROOT = os.path.dirname(TEST_PATH)

print("[+] Discovering package roots...")
for p in [SOURCE_PATH, TRAIN_VAL_ROOT, TEST_ROOT]:
    if os.path.exists(p):
        if p not in sys.path: sys.path.append(p)
        # Recursive check to handle zip-nesting
        for root_node, dirs, files in os.walk(p):
            if 'model' in dirs and 'config.py' in files:
                SOURCE_ROOT = root_node
                if SOURCE_ROOT not in sys.path:
                    sys.path.insert(0, SOURCE_ROOT)
                print("✅ Source root discovered at: " + str(SOURCE_ROOT))
                break
if not SOURCE_ROOT:
    print("⚠️ Warning: Could not find 'model/' package automatically.")
print("="*50 + "\\n")

def plot_history(history, plot_path):
    plt.figure(figsize=(10,6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title(f"Training Convergence - {{BASE_NAME}}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss curve saved to: {{plot_path}}")

# 2. LOAD SOURCE & PARAMS
from model.model import MLP, MLPDelta
from model.train import train
from model.data_loader import get_cloud_dataloaders
from model.evaluate import evaluate
from config import config # Use the config file from your source dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {{device}}")

# Override local path with Kaggle path
config['data_path'] = "/kaggle/input"
delta_choice = {delta_choice}

# 3. LOAD DATA BUNDLES
print("\\n[+] Unpacking PyTorch Tensors from bundles...")
loaders = {{}}

if os.path.exists(TRAIN_VAL_PATH):
    print(f"  [>] Loading Train/Val: {{TRAIN_VAL_PATH}}")
    loaders.update(get_cloud_dataloaders(TRAIN_VAL_PATH))
else:
    print(f"  [!] Warning: Train/Val bundle not found at {{TRAIN_VAL_PATH}}")

if DO_TEST:
    if os.path.exists(TEST_PATH):
        print(f"  [>] Loading Test: {{TEST_PATH}}")
        loaders.update(get_cloud_dataloaders(TEST_PATH))
    else:
        print(f"  [!] Warning: Test bundle not found at {{TEST_PATH}}")

# 4. INITIALIZE MODEL
print("\\n[+] Initializing Model...")
input_dim = config.get("input_dim", 1025)
hidden_dims = config.get("hidden_dims", [512, 256, 128, 64])
dropout = config.get("dropout", 0.2)
activation = config.get("activation_function", "GELU")

if delta_choice:
    model = MLPDelta(input_dim, hidden_dims, dropout)
    print("  [>] Architecture: MLPDelta (Residual Correction)")
else:
    model = MLP(input_dim, hidden_dims, dropout)
    print("  [>] Architecture: Standard MLP")

# 5. EXECUTE TRAINING
print("\\n[+] Starting Training Loop...")
model, history = train(model, loaders, config, device, delta=delta_choice)

# 6. FINAL EVALUATION
if DO_TEST and 'test' in loaders:
    print("\\n[+] Running Final Evaluation on Test Set...")
    criterion = nn.MSELoss()
    test_loss = evaluate(model, loaders['test'], criterion, device, delta=delta_choice)
    print(f"  [✔] Final Test Loss: {{test_loss:.6f}}")

# 7. SAVE COMPREHENSIVE RESULTS
RESULTS_BASE = "/kaggle/working/results"
models_dir = os.path.join(RESULTS_BASE, "models")
histories_dir = os.path.join(RESULTS_BASE, "histories")
configs_dir = os.path.join(RESULTS_BASE, "configs")
plots_dir = os.path.join(RESULTS_BASE, "plots")

for d in [models_dir, histories_dir, configs_dir, plots_dir]:
    os.makedirs(d, exist_ok=True)

model_path = os.path.join(models_dir, f"{{BASE_NAME}}.pt")
history_path = os.path.join(histories_dir, f"{{BASE_NAME}}_history.json")
config_path = os.path.join(configs_dir, f"{{BASE_NAME}}_config.json")
plot_path = os.path.join(plots_dir, f"{{BASE_NAME}}_curve.png")

print("\\n[+] Persisting results to /kaggle/working/results/...")
torch.save(model.state_dict(), model_path)
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
    
plot_history(history, plot_path)

print(f"\\n[✔] COMPLETE! All artifacts saved with base name: {{BASE_NAME}}")
'''
    return template
