import torch
import numpy as np
import os
from tqdm import tqdm

def repair_bundle(input_path, output_path):
    print(f"\n[+] Loading bit-packed bundle: {input_path}")
    # Load with mmap to be safe
    bundle = torch.load(input_path, weights_only=False)
    
    repaired_bundle = {}
    
    for split, tensor_dict in bundle.items():
        if split == "metadata":
            repaired_bundle[split] = tensor_dict
            continue
            
        print(f"  [>] Processing split: {split.upper()}")
        fp = tensor_dict['fp']
        noisy = tensor_dict['noisy']
        y_true = tensor_dict['y_true']
        
        # RECURSIVE UNPACKING: Detect double or single packing
        current_fp = fp
        while current_fp.shape[-1] < 1024:
            dim = current_fp.shape[-1]
            if dim in [16, 128]:
                print(f"    [!] Packing detected (shape {dim}). Unpacking 8x...")
                fp_np = current_fp.numpy().astype(np.uint8)
                unpacked_np = np.unpackbits(fp_np, axis=-1).astype(np.float32)
                current_fp = torch.from_numpy(unpacked_np)
            else:
                print(f"    [!] Error: Unexpected shape {dim}. Cannot unpack further.")
                break
        
        fp_final = current_fp.to(torch.float16)
        print(f"    [✔] Final shape: {fp_final.shape}")
            
        repaired_bundle[split] = {
            'fp': fp_final,
            'noisy': noisy.to(torch.float32),
            'y_true': y_true.to(torch.float32)
        }
        
    print(f"  [+] Saving repaired bundle to: {output_path}")
    torch.save(repaired_bundle, output_path)
    print(f"  [✔] Done!")

if __name__ == "__main__":
    BUNDLE_DIR = "results/cloud_datasets"
    
    # Files to repair
    targets = [
        "scaffold_train_val.pt",
        "scaffold_test.pt"
    ]
    
    for filename in targets:
        in_p = os.path.join(BUNDLE_DIR, filename)
        if os.path.exists(in_p):
            # Create a backup just in case
            backup_p = in_p + ".packed.bak"
            if not os.path.exists(backup_p):
                print(f"[!] Creating backup: {backup_p}")
                os.rename(in_p, backup_p)
                repair_bundle(backup_p, in_p)
            else:
                print(f"[!] Backup already exists, using it as source...")
                repair_bundle(backup_p, in_p)
        else:
            print(f"[!] File not found: {in_p}")
