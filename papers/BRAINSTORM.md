# Shortcut Learning in Noisy Property Correction MLP

## Problem

Model ignores noisy input → learns `fingerprint → true value` directly via group additivity. Noise is i.i.d. so fingerprint alone is the optimal predictor. Noisy value is useless by design.

---

## Ideas

### 1. Delta Learning
- Target becomes `Δ = noisy - true`
- Forces the model to use the noisy input (target is a function of it)
- Inference: `true = noisy - model(fp, noisy)`
- Same idea as ΔML in QM (DFT → CCSD corrections)

### 2. Structure-Dependent Noise
- Make noise correlated with substructure (e.g. bias per functional group, scale with MW)
- Now fingerprint is needed to *correct* the measurement, not replace it
- More realistic, mirrors actual instrument artifacts

### 3. Fingerprint Masking
- Randomly drop bits during training
- Model can't always reconstruct true value from structure alone → must use noisy value as prior
---