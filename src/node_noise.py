"""
Node-level noise injection for Chemprop v2 message-passing layers.

Wraps a standard MessagePassing module to inject configurable noise into the
atom-feature matrix (BatchMolGraph.V) *before* message passing begins.

The noise is only applied during training (self.training == True).

Noise configuration is a list of "layers" that stack on top of each other:
    [
        {"type": "normal",  "scale": 0.1},
        {"type": "uniform", "scale": 0.2},
        ...
    ]

Two fractions control where the noise lands:
    node_fraction : float  — fraction of nodes (atoms) randomly targeted per batch
    dim_fraction  : float  — fraction of feature dimensions targeted within those nodes

Both masks are re-sampled every forward pass so the model cannot memorise them.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


# ── noise generators ────────────────────────────────────────────────────────

def _generate_noise(shape: tuple[int, ...], noise_type: str, scale: float,
                    device: torch.device) -> Tensor:
    """Generate a noise tensor of the given shape, type and scale."""
    if scale <= 0:
        return torch.zeros(shape, device=device)

    if noise_type == "normal":
        return torch.randn(shape, device=device) * scale

    elif noise_type == "uniform":
        half = math.sqrt(3) * scale          # match variance of N(0,scale)
        return torch.empty(shape, device=device).uniform_(-half, half)

    elif noise_type == "bimodal":
        gauss = torch.randn(shape, device=device) * (scale * 0.866025)
        shift = torch.randint(0, 2, shape, device=device).float() * 2 - 1  # ±1
        return gauss + shift * (scale * 0.5)

    elif noise_type == "laplace":
        # Laplace(0, scale/sqrt(2)) gives the same variance as N(0, scale)
        u = torch.empty(shape, device=device).uniform_(-0.5 + 1e-7, 0.5 - 1e-7)
        b = scale / math.sqrt(2)
        return -b * u.sign() * torch.log1p(-2 * u.abs())

    # fallback → Gaussian
    return torch.randn(shape, device=device) * scale


# ── wrapper module ──────────────────────────────────────────────────────────

class NoisyMessagePassing(nn.Module):
    """
    Drop-in wrapper around any Chemprop ``MessagePassing`` module.

    During **training** it:
        1. selects a random subset of nodes  (controlled by *node_fraction*)
        2. selects a random subset of feature dimensions  (controlled by *dim_fraction*)
        3. stacks one or more noise layers onto the selected entries of ``bmg.V``

    During **eval / inference** it does nothing — the original message passing
    runs unmodified.

    Parameters
    ----------
    inner : nn.Module
        The original ``MessagePassing`` module (e.g. ``BondMessagePassing``).
    node_fraction : float
        Fraction of nodes to perturb (0.0–1.0).  Rounded up per molecule so
        at least 1 node is always targeted when > 0.
    dim_fraction : float
        Fraction of feature-vector dimensions to perturb (0.0–1.0).
    noise_layers : list[dict]
        Each dict has ``{"type": str, "scale": float}``.
        They are applied additively in order.
    """

    def __init__(
        self,
        inner: nn.Module,
        node_fraction: float = 0.0,
        dim_fraction: float = 0.0,
        noise_layers: list[dict] | None = None,
    ):
        super().__init__()
        self.inner = inner
        self.node_fraction = node_fraction
        self.dim_fraction = dim_fraction
        self.noise_layers = noise_layers or []

    # -- delegate every attribute the MPNN might look up on message_passing --
    def __getattr__(self, name: str):
        """Proxy attribute access to inner module when not found on self.

        This is essential because chemprop's MPNN accesses things like
        ``self.message_passing.W_d`` or ``self.message_passing.output_dim``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    # -- core logic ----------------------------------------------------------

    def _build_node_mask(self, batch: Tensor, n_nodes: int,
                         device: torch.device) -> Tensor:
        """Return a boolean mask of shape ``(n_nodes,)`` selecting the target nodes."""
        if self.node_fraction <= 0:
            return torch.zeros(n_nodes, dtype=torch.bool, device=device)
        if self.node_fraction >= 1.0:
            return torch.ones(n_nodes, dtype=torch.bool, device=device)

        mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)

        # per-molecule rounding so small molecules still get noise
        mol_ids = batch.unique()
        for mid in mol_ids:
            mol_mask = batch == mid
            mol_size = mol_mask.sum().item()
            k = max(1, math.ceil(mol_size * self.node_fraction))
            k = min(k, mol_size)
            # random k indices within this molecule
            mol_indices = mol_mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(mol_size, device=device)[:k]
            mask[mol_indices[perm]] = True

        return mask

    def _build_dim_mask(self, d_v: int, device: torch.device) -> Tensor:
        """Return a boolean mask of shape ``(d_v,)`` selecting the target dimensions."""
        if self.dim_fraction <= 0:
            return torch.zeros(d_v, dtype=torch.bool, device=device)
        if self.dim_fraction >= 1.0:
            return torch.ones(d_v, dtype=torch.bool, device=device)

        k = max(1, math.ceil(d_v * self.dim_fraction))
        k = min(k, d_v)
        perm = torch.randperm(d_v, device=device)[:k]
        mask = torch.zeros(d_v, dtype=torch.bool, device=device)
        mask[perm] = True
        return mask

    def forward(self, bmg, V_d: Tensor | None = None) -> Tensor:
        if self.training and self.noise_layers and self.node_fraction > 0 and self.dim_fraction > 0:
            device = bmg.V.device
            n_nodes, d_v = bmg.V.shape

            node_mask = self._build_node_mask(bmg.batch, n_nodes, device)
            dim_mask  = self._build_dim_mask(d_v, device)

            # Outer product → (n_nodes, d_v) selection grid
            # Only modify the entries where both masks are True
            for layer in self.noise_layers:
                noise_type = layer.get("type", "normal")
                noise_scale = layer.get("scale", 0.0)
                if noise_scale <= 0:
                    continue

                # Generate full noise, then zero-out non-selected entries
                n_selected_nodes = node_mask.sum().item()
                n_selected_dims  = dim_mask.sum().item()

                if n_selected_nodes == 0 or n_selected_dims == 0:
                    continue

                noise = _generate_noise(
                    (n_selected_nodes, n_selected_dims),
                    noise_type, noise_scale, device,
                )
                # Apply in-place via advanced indexing
                bmg.V[node_mask.unsqueeze(1) & dim_mask.unsqueeze(0)] += noise.reshape(-1)

        return self.inner(bmg, V_d)

    def get_config_summary(self) -> str:
        """Return a human-readable summary of the noise configuration."""
        if not self.noise_layers or self.node_fraction <= 0 or self.dim_fraction <= 0:
            return "Node noise: OFF"

        lines = [f"Node fraction: {self.node_fraction*100:.0f}%  |  Dim fraction: {self.dim_fraction*100:.0f}%"]
        for i, layer in enumerate(self.noise_layers):
            lines.append(f"  Layer {i+1}: {layer.get('type','normal')} (scale={layer.get('scale',0):.4f})")
        return "\n".join(lines)
