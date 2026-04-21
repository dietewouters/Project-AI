import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class PropertyScaler:
    """ 
    Handles centering of the background property and standardization 
    of the residue (delta) or absolute values.
    """
    def __init__(self, mode="delta"):
        self.mode = mode # "none", "standard", "delta", "minmax"
        self.mean_h = 0.0
        self.std_h = 1.0
        self.mean_y = 0.0
        self.std_y = 1.0
        self.std_delta = 1.0
        self.is_fitted = False
        
        # sklearn scalers for standard/minmax modes
        self.scaler_h = None
        self.scaler_y = None

    def fit(self, noisy, clean):
        """
        Fits the scaler to the provided noisy and clean tensors.
        noisy: Tensor of noisy input properties
        clean: Tensor of clean target properties
        """
        # Handle both Tensors and Numpy arrays
        if torch.is_tensor(noisy):
            noisy_flat = noisy.view(-1).cpu().numpy()
        else:
            noisy_flat = noisy.flatten()
            
        if torch.is_tensor(clean):
            clean_flat = clean.view(-1).cpu().numpy()
        else:
            clean_flat = clean.flatten()
        
        if self.mode == "delta":
            # 1. Calculate the mean of input values (noisy)
            self.mean_h = float(np.mean(noisy_flat))
            
            # 2. Calculate the delta (noisy - clean)
            delta = noisy_flat - clean_flat
            
            # 3. Calculate the std_delta
            self.std_delta = float(np.std(delta))
            if self.std_delta == 0: self.std_delta = 1.0
            
            # 4. Scale what is necessary (done in transform)
            print(f"\n[Scaler] Delta Scaling Metrics Fitted:")
            print(f"  [>] Mean (Noisy Input): {self.mean_h:.4f}")
            print(f"  [>] Std (Delta):        {self.std_delta:.4f}")
            
        elif self.mode == "standard":
            self.scaler_h = StandardScaler()
            self.scaler_y = StandardScaler()
            self.scaler_h.fit(noisy_flat.reshape(-1, 1))
            self.scaler_y.fit(clean_flat.reshape(-1, 1))
            print(f"\n[Scaler] Sklearn StandardScaler Fitted")
            
        elif self.mode == "minmax":
            self.scaler_h = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            self.scaler_h.fit(noisy_flat.reshape(-1, 1))
            self.scaler_y.fit(clean_flat.reshape(-1, 1))
            print(f"\n[Scaler] Sklearn MinMaxScaler Fitted")
            
        elif self.mode == "none":
            print(f"\n[Scaler] No Scaling Applied")
            
        self.is_fitted = True

    def transform(self, noisy, clean=None):
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform!")
            
        if torch.is_tensor(noisy):
            n_flat = noisy.view(-1).cpu().numpy().reshape(-1, 1)
        else:
            n_flat = noisy.reshape(-1, 1)
        
        if self.mode == "none":
            n_scaled = n_flat
            if clean is not None:
                if torch.is_tensor(clean):
                    y_scaled = clean.view(-1).cpu().numpy()
                else:
                    y_scaled = clean.flatten()
                return torch.tensor(n_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32)
            return torch.tensor(n_scaled, dtype=torch.float32)
            
        elif self.mode in ["standard", "minmax"]:
            n_scaled = self.scaler_h.transform(n_flat)
            if clean is not None:
                if torch.is_tensor(clean):
                    c_np = clean.view(-1).cpu().numpy().reshape(-1, 1)
                else:
                    c_np = clean.reshape(-1, 1)
                y_scaled = self.scaler_y.transform(c_np)
                return torch.tensor(n_scaled, dtype=torch.float32), torch.tensor(y_scaled.flatten(), dtype=torch.float32)
            return torch.tensor(n_scaled, dtype=torch.float32)
            
        else: # "delta" mode
            n_scaled = n_flat - self.mean_h
            if clean is not None:
                if torch.is_tensor(clean):
                    c_np = clean.view(-1).cpu().numpy()
                else:
                    c_np = clean.flatten()
                y_scaled = (n_flat.flatten() - c_np) / self.std_delta
                return torch.tensor(n_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32)
            return torch.tensor(n_scaled, dtype=torch.float32)

    def inverse_transform(self, noisy, pred_out):
        if torch.is_tensor(noisy):
            n_flat = noisy.view(-1).cpu().numpy().reshape(-1, 1)
        else:
            n_flat = noisy.reshape(-1, 1)
            
        if torch.is_tensor(pred_out):
            p_flat = pred_out.view(-1).cpu().numpy().reshape(-1, 1)
        else:
            p_flat = pred_out.reshape(-1, 1)
        
        if self.mode == "none":
            return torch.tensor(p_flat.flatten(), dtype=torch.float32)
            
        elif self.mode in ["standard", "minmax"]:
            p_inv = self.scaler_y.inverse_transform(p_flat)
            return torch.tensor(p_inv.flatten(), dtype=torch.float32)
            
        else: # "delta" mode
            y_inv = n_flat.flatten() - (p_flat.flatten() * self.std_delta)
            return torch.tensor(y_inv, dtype=torch.float32)

    def to_dict(self):
        d = {"mode": self.mode, "is_fitted": self.is_fitted}
        if self.mode == "delta":
            d.update({
                "mean_h": self.mean_h,
                "std_delta": self.std_delta
            })
        elif self.mode == "standard" and self.is_fitted:
            d.update({
                "h_mean": self.scaler_h.mean_.tolist(),
                "h_var": self.scaler_h.var_.tolist(),
                "y_mean": self.scaler_y.mean_.tolist(),
                "y_var": self.scaler_y.var_.tolist()
            })
        elif self.mode == "minmax" and self.is_fitted:
            d.update({
                "h_min": self.scaler_h.min_.tolist(),
                "h_scale": self.scaler_h.scale_.tolist(),
                "y_min": self.scaler_y.min_.tolist(),
                "y_scale": self.scaler_y.scale_.tolist()
            })
        return d

    @classmethod
    def from_dict(cls, d):
        scaler = cls(mode=d.get("mode", "delta"))
        scaler.is_fitted = d.get("is_fitted", False)
        
        if scaler.mode == "delta":
            scaler.mean_h = d.get("mean_h", 0.0)
            scaler.std_delta = d.get("std_delta", 1.0)
        elif scaler.mode == "standard" and scaler.is_fitted:
            scaler.scaler_h = StandardScaler()
            scaler.scaler_y = StandardScaler()
            scaler.scaler_h.mean_ = np.array(d["h_mean"])
            scaler.scaler_h.var_ = np.array(d["h_var"])
            scaler.scaler_h.scale_ = np.sqrt(scaler.scaler_h.var_)
            scaler.scaler_y.mean_ = np.array(d["y_mean"])
            scaler.scaler_y.var_ = np.array(d["y_var"])
            scaler.scaler_y.scale_ = np.sqrt(scaler.scaler_y.var_)
        elif scaler.mode == "minmax" and scaler.is_fitted:
            scaler.scaler_h = MinMaxScaler()
            scaler.scaler_y = MinMaxScaler()
            scaler.scaler_h.min_ = np.array(d["h_min"])
            scaler.scaler_h.scale_ = np.array(d["h_scale"])
            scaler.scaler_y.min_ = np.array(d["y_min"])
            scaler.scaler_y.scale_ = np.array(d["y_scale"])
        return scaler
