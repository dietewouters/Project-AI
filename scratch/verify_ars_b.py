import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, "/Users/Christian/Desktop/University/Master 1/Semester 2/Project AI /Project-AI/src")

from model.model import ARSDeltaMLP
from model.scaler import PropertyScaler

def verify_ars_logic():
    print("=== ARS Option B Verification ===")
    
    # 1. Setup Scaler
    scaler = PropertyScaler(mode="delta")
    # Simulation: Enthalpy around -500, error around 10
    noisy = torch.tensor([-500.0, -510.0, -490.0]).float()
    clean = torch.tensor([-510.0, -500.0, -505.0]).float() # delta = V - Y = [10, -10, 15]
    scaler.fit(noisy, clean)
    
    # 2. Setup Model
    model = ARSDeltaMLP(1024, [128], 0.1)
    model.eval()
    
    # 3. Mock forward pass
    # Standardize V input for model
    v_in = (noisy - scaler.mean_h) / scaler.std_delta
    fp = torch.zeros((3, 1024))
    
    with torch.no_grad():
        # Alpha ~ 1.0 (since bias is 5.0)
        # Delta ~ 0.0 (since weights are 0.0)
        residue_pred = model(fp, v_in) # Returns alpha * delta
        
    print(f"Alpha Initialized: {torch.sigmoid(model.alpha_head.bias).item():.4f}")
    print(f"Residue Pred (initial, should be ~0): {residue_pred}")
    
    # 4. Inverse Transform Check
    # If pred=0, physical_y should be noisy
    y_phys = scaler.inverse_transform(noisy, residue_pred)
    print(f"Noisy: {noisy}")
    print(f"Clean (target): {clean}")
    print(f"Predicted Physical (alpha~1, delta~0): {y_phys}")
    
    # 5. Mock a "Good" Prediction
    # Let's say alpha=1 and delta is the true residual
    _, y_resid_scaled = scaler.transform(noisy, clean)
    print(f"True Scaled Residue: {y_resid_scaled}")
    
    # Physical unscaling
    y_final = scaler.inverse_transform(noisy, y_resid_scaled)
    print(f"Unscaled True Residue (should match Clean): {y_final}")
    
    is_correct = torch.allclose(y_final, clean, atol=1e-4)
    print(f"Math Consistency Check: {'PASSED' if is_correct else 'FAILED'}")

if __name__ == "__main__":
    verify_ars_logic()
