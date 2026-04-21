# MLP Architecture Definition
# model.py

import torch
import torch.nn as nn
from config import config

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = config["input_dim"],       
        hidden_dims: list = config["hidden_dims"],
        dropout: float = config["dropout"],
        activation_type: str = "GELU"
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims
        
        activation_map = {
            "GELU": nn.GELU,
            "ELU": nn.ELU,
            "ReLU": nn.ReLU,
            "SiLU": nn.SiLU,
            "LeakyReLU": nn.LeakyReLU
        }
        
        # Default to GELU if not found
        ActivationClass = activation_map.get(activation_type, nn.GELU)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ActivationClass())
            layers.append(nn.Dropout(dropout))

        # Output layer — single value, no activation for regression
        layers.append(nn.Linear(dims[-1], 1))

        self.network = nn.Sequential(*layers)

        print(self.network)

    def forward(self, x):
        return self.network(x).squeeze(1)  # shape (batch_size,)

class MLPDelta(MLP):
    def predict(self, x):
        delta = self.network(x).squeeze(1)
        noisy = x[:, -1]
        return noisy - delta  # Subtract delta (error) from noisy to predict the true value

class MLPFiLM(nn.Module):
    """
    Multi-Fidelity Feature-wise Linear Modulation (FiLM) Network.
    Uses the molecular fingerprint to predict scaling (gamma) and shifting (beta) factors,
    which modulate the noisy scalar signal to predict the clean signal.
    """
    def __init__(
        self,
        input_dim: int = 1024,       
        hidden_dims: list = [256, 64],
        dropout: float = 0.0,
        activation_type: str = "GELU"
    ):
        super().__init__()
        
        # Determine actual input dimension for the fingerprint (should be total minus 1)
        # We assume X has shape (batch, 1025) where 1024 is fp and 1 is noisy.
        fp_dim = input_dim - 1 if input_dim > 1024 else input_dim
        
        layers = []
        dims = [fp_dim] + hidden_dims
        
        activation_map = {
            "GELU": nn.GELU,
            "ELU": nn.ELU,
            "ReLU": nn.ReLU,
            "SiLU": nn.SiLU,
            "LeakyReLU": nn.LeakyReLU
        }
        
        # Default to GELU if not found
        ActivationClass = activation_map.get(activation_type, nn.GELU)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ActivationClass())
            layers.append(nn.Dropout(dropout))

        self.feature_extractor = nn.Sequential(*layers)
        
        # FiLM Heads: Output gamma (scaling) and beta (shifting)
        self.gamma_head = nn.Linear(dims[-1], 1)
        self.beta_head = nn.Linear(dims[-1], 1)
        
        # Softplus ensures gamma is strictly positive (meaningful scaling)
        self.gamma_activation = nn.Softplus()
        
        # Initialize gamma close to 1.0 (no scaling) and beta close to 0 (no shift)
        # To make Softplus(x) = 1, x approx 0.54
        nn.init.constant_(self.gamma_head.bias, 0.54)
        nn.init.constant_(self.gamma_head.weight, 0.0)
        nn.init.constant_(self.beta_head.bias, 0.0)
        nn.init.constant_(self.beta_head.weight, 0.0)

        print("Initialized MLPFiLM Modulator")
        print(self)

    def forward(self, x):
        # x is concatenated: [fingerprint (1024), noisy_scalar (1)]
        fp = x[:, :-1]
        noisy = x[:, -1]
        
        # Extract representation from the fingerprint
        features = self.feature_extractor(fp)
        
        # Predict modulation parameters
        gamma = self.gamma_activation(self.gamma_head(features)).squeeze(1)
        beta = self.beta_head(features).squeeze(1)
        
        # Apply transformation: V_corr = gamma * V_noisy + beta
        v_corr = gamma * noisy + beta
        
        # The pipeline's target for 'delta' is universally (noisy - clean).
        # We output (noisy - v_corr).
        # When Loss minimizes (noisy - v_corr) vs (noisy - clean), it structurally forces v_corr = clean.
        return noisy - v_corr

    def predict(self, x):
        # The test evaluation returns target in the original space: 
        # return noisy - delta
        # Since our delta is noisy - v_corr, this reconstructing: noisy - (noisy - v_corr) = v_corr
        delta = self.forward(x)
        noisy = x[:, -1]
        return noisy - delta

class MLPARS(nn.Module):
    """
    Adaptive Residual Scaling (ARS) Network tailored for Scaffold Splitting.
    Dual-head architecture predicting an Alpha trust-weight and Delta residual.
    Final output = (Alpha * V_low) + Delta
    """
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list = [256, 64],
        dropout: float = 0.25, # Scaffold-split awareness
        activation_type: str = "GELU"
    ):
        super().__init__()
        
        fp_dim = input_dim - 1 if input_dim > 1024 else input_dim
        
        layers = []
        dims = [fp_dim] + hidden_dims
        
        activation_map = {
            "GELU": nn.GELU,
            "ELU": nn.ELU,
            "ReLU": nn.ReLU,
            "SiLU": nn.SiLU,
            "LeakyReLU": nn.LeakyReLU
        }
        ActivationClass = activation_map.get(activation_type, nn.GELU)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ActivationClass())
            layers.append(nn.Dropout(dropout)) # Robust shared backbone dropout

        self.feature_extractor = nn.Sequential(*layers)
        
        # Dual Heads
        self.alpha_head = nn.Linear(dims[-1], 1)
        self.delta_head = nn.Linear(dims[-1], 1)
        
        # Initialize Alpha bias to 2.0 (Sigmoid defaults to ~0.88 trust in the baseline)
        nn.init.constant_(self.alpha_head.bias, 2.0)
        nn.init.constant_(self.alpha_head.weight, 0.0)
        
        nn.init.constant_(self.delta_head.bias, 0.0)
        nn.init.constant_(self.delta_head.weight, 0.0)

        print("Initialized MLPARS (Dual-Head)")
        print(self)

    def forward(self, x):
        fp = x[:, :-1]
        noisy = x[:, -1]
        
        features = self.feature_extractor(fp)
        
        alpha = torch.sigmoid(self.alpha_head(features)).squeeze(1)
        delta = self.delta_head(features).squeeze(1)
        
        y_pred = (alpha * noisy) + delta
        
        # Predict "noisy - y_pred" so that matching it against "noisy - clean" 
        # exactly forces y_pred to approximate clean securely with no custom loss needed.
        return noisy - y_pred

    def predict(self, x):
        # As evaluated in `test()`, returning noisy - delta reconstructs Y_pred
        delta = self.forward(x)
        noisy = x[:, -1]
        return noisy - delta

class GRNBlock(nn.Module):
    """
    Gated Residual Network Block.
    Solves vanishing signal for sparse fingerprints via GLU combined with Residual connections.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1, activation_type: str = "GELU"):
        super().__init__()
        
        activation_map = {
            "GELU": nn.GELU,
            "ELU": nn.ELU,
            "ReLU": nn.ReLU,
            "SiLU": nn.SiLU,
            "LeakyReLU": nn.LeakyReLU
        }
        ActivationClass = activation_map.get(activation_type, nn.GELU)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = ActivationClass()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # GLU components (takes input of hidden_dim and gates it)
        self.glu_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.glu_proj2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Base Projection
        out = self.norm1(self.act1(self.linear1(x)))
        out = self.dropout(out)
        
        # GLU Mechanism: Output = Linear1(x) * Sigmoid(Linear2(x))
        gating_signal = torch.sigmoid(self.glu_proj2(out))
        gated_out = self.glu_proj1(out) * gating_signal
        
        # Residual Connection + Final Normalization 
        return self.norm2(x + gated_out)

class DeltaGRNMLP(nn.Module):
    """
    Delta-Learning compatible MLP constructed with stacked GRNBlocks.
    Directly predicts Δ while avoiding vanishing gradients through sparse inputs.
    """
    def __init__(
        self,
        input_dim: int = 1025,
        hidden_dims: list = [256], # Project to first hidden dimension
        dropout: float = 0.1,
        activation_type: str = "GELU"
    ):
        super().__init__()
        
        proj_dim = hidden_dims[0] if hidden_dims else 256
        
        # 1. Project initial mixed input [1025] to the hidden dimension
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # 2. Stack 3 GRN Blocks
        self.grn_blocks = nn.Sequential(
            GRNBlock(proj_dim, dropout, activation_type),
            GRNBlock(proj_dim, dropout, activation_type),
            GRNBlock(proj_dim, dropout, activation_type)
        )
        
        # 3. Output Delta
        self.output_layer = nn.Linear(proj_dim, 1)

        print("Initialized DeltaGRNMLP (3-Block GRN Architecture)")
        print(self)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.grn_blocks(out)
        return self.output_layer(out).squeeze(1)

    def predict(self, x):
        delta = self.forward(x)
        noisy = x[:, -1]
        return noisy - delta

class GRNFiLMMLP(nn.Module):
    """
    Ultimate Hybrid: Uses stacked GRN blocks specifically processing the sparse fingerprint,
    then uses the FiLM heads to modulate the dense noisy scalar.
    """
    def __init__(self, input_dim: int = 1025, hidden_dims: list = [256], dropout: float = 0.1, activation_type: str = "GELU"):
        super().__init__()
        # Backbone (Fingerprint only -> 1024)
        fp_dim = input_dim - 1 if input_dim > 1024 else input_dim
        proj_dim = hidden_dims[0] if hidden_dims else 256
        
        self.input_layer = nn.Sequential(nn.Linear(fp_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.grn_blocks = nn.Sequential(
            GRNBlock(proj_dim, dropout, activation_type),
            GRNBlock(proj_dim, dropout, activation_type),
            GRNBlock(proj_dim, dropout, activation_type)
        )
        
        # FiLM Heads
        self.gamma_head = nn.Sequential(nn.Linear(proj_dim, 1), nn.Softplus())
        self.beta_head = nn.Linear(proj_dim, 1)
        
        # Initialize Gamma to ~1, Beta to ~0
        nn.init.constant_(self.gamma_head[0].weight, 0.0)
        nn.init.constant_(self.gamma_head[0].bias, 0.5413) # Softplus(0.5413) ≈ 1.0
        nn.init.constant_(self.beta_head.weight, 0.0)
        nn.init.constant_(self.beta_head.bias, 0.0)

    def forward(self, x):
        fp, noisy = x[:, :-1], x[:, -1]
        features = self.grn_blocks(self.input_layer(fp))
        
        gamma = self.gamma_head(features).squeeze(1)
        beta = self.beta_head(features).squeeze(1)
        
        v_corr = gamma * noisy + beta
        return noisy - v_corr

    def predict(self, x):
        # returns noisy - (noisy - v_corr) = v_corr
        delta = self.forward(x)
        noisy = x[:, -1]
        return noisy - delta

class GRNARSMLP(nn.Module):
    """
    Ultimate Hybrid: Uses stacked GRN blocks specifically processing the sparse fingerprint,
    then uses the ARS Dual-Heads (Alpha Trust & Delta) to modulate the dense noisy scalar.
    """
    def __init__(self, input_dim: int = 1025, hidden_dims: list = [256], dropout: float = 0.25, activation_type: str = "GELU"):
        super().__init__()
        fp_dim = input_dim - 1 if input_dim > 1024 else input_dim
        proj_dim = hidden_dims[0] if hidden_dims else 256
        
        self.input_layer = nn.Sequential(nn.Linear(fp_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.grn_blocks = nn.Sequential(
            GRNBlock(proj_dim, dropout, activation_type),
            GRNBlock(proj_dim, dropout, activation_type),
            GRNBlock(proj_dim, dropout, activation_type)
        )
        
        # ARS Dual-Heads
        self.alpha_head = nn.Linear(proj_dim, 1)
        self.delta_head = nn.Linear(proj_dim, 1)
        
        # Initialize Alpha bias to 2.0 (~0.88 baseline trust)
        nn.init.constant_(self.alpha_head.weight, 0.0)
        nn.init.constant_(self.alpha_head.bias, 2.0)
        nn.init.constant_(self.delta_head.weight, 0.0)
        nn.init.constant_(self.delta_head.bias, 0.0)

    def forward(self, x):
        fp, noisy = x[:, :-1], x[:, -1]
        features = self.grn_blocks(self.input_layer(fp))
        
        alpha = torch.sigmoid(self.alpha_head(features)).squeeze(1)
        delta = self.delta_head(features).squeeze(1)
        
        y_pred = (alpha * noisy) + delta
        return noisy - y_pred

    def predict(self, x):
        delta = self.forward(x)
        noisy = x[:, -1]
        return noisy - delta

