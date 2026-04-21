import os
import torch
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    # --- DEFAULT PATH DETECTION ---
    IS_KAGGLE = os.path.exists('/kaggle/input')
    DEFAULT_BASE = "/kaggle/working" if IS_KAGGLE else os.path.dirname(os.path.abspath(__file__))

    # --- CLI ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Merge quarterly bit-packed bundles into final sets.")
    parser.add_argument("--parts_dir", type=str, 
                        default=os.path.join(DEFAULT_BASE, "dataset", "pt_parts"),
                        help="Folder containing bundle_q*.pt")
    parser.add_argument("--out_dir", type=str, 
                        default=os.path.join(DEFAULT_BASE, "results", "cloud_datasets"),
                        help="Where to save the final .pt files")
    
    args = parser.parse_args()

    # --- EXECUTION ---
    print("🚀 STAGE 5: Merging Quarterly Bundles...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load all parts
    bundles = []
    for i in range(4):
        pt_path = os.path.join(args.parts_dir, f"bundle_q{i}.pt")
        if os.path.exists(pt_path):
            print(f"   Loading {pt_path}...")
            bundles.append(torch.load(pt_path))
        else:
            print(f"❌ Error: {pt_path} missing!")
            
    if len(bundles) < 4:
        print("❌ Error: Not all 4 parts are available. Run Stage 4 for each quarter first.")
        exit(1)

    # 2. Prepare final bundles
    final_bundle_train_val = {}
    final_bundle_test = {}
    
    # Extract metadata from the first part (they should all be identical)
    if "metadata" in bundles[0]:
        final_bundle_train_val["metadata"] = bundles[0]["metadata"]
        final_bundle_test["metadata"] = bundles[0]["metadata"]

    # 3. Concatenate Splits
    for split_type in ["train", "val", "test"]:
        print(f"   Merging {split_type.upper()} tensors...")
        fps = []
        noisys = []
        y_trues = []
        
        for b in bundles:
            if split_type in b:
                fps.append(b[split_type]["fp"])
                noisys.append(b[split_type]["noisy"])
                y_trues.append(b[split_type]["y_true"])
        
        if fps:
            target_bundle = final_bundle_test if split_type == "test" else final_bundle_train_val
            target_bundle[split_type] = {
                "fp": torch.cat(fps, dim=0),
                "noisy": torch.cat(noisys, dim=0),
                "y_true": torch.cat(y_trues, dim=0)
            }
            print(f"      Total {split_type.upper()} molecules: {len(target_bundle[split_type]['fp'])}")

    # 4. Save Final Files
    out_tv = os.path.join(args.out_dir, "scaffold_train_val.pt")
    out_test = os.path.join(args.out_dir, "scaffold_test.pt")
    
    print(f"\n💾 Saving final bundles to {args.out_dir}...")
    torch.save(final_bundle_train_val, out_tv)
    torch.save(final_bundle_test, out_test)
    
    print(f"✅ STAGE 5 COMPLETE!")
    print(f"   Final Train/Val File: {os.path.getsize(out_tv) / (1024**2):.1f} MB")
    print(f"   Final Test File:      {os.path.getsize(out_test) / (1024**2):.1f} MB")
