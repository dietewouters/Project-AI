# RDKit Fingerprints & Masking Strategies

This document serves as a reference for the different fingerprint generation methods available in RDKit and advanced masking techniques designed to prevent Neural Networks from relying on data shortcuts (e.g., memorizing group additivity rules instead of denoising).

---

## Part 1: RDKit Fingerprints and Tweakable Parameters

### 1. Morgan Fingerprints (Circular Fingerprints)
Equivalent to ECFP or FCFP fingerprints, generating bit vectors based on circular atom neighborhoods.
* **`radius`**: Maximum neighborhood radius. (`2` = ECFP4, `3` = ECFP6).
* **`nBits`**: Array length. Defaults are often `1024` or `2048`.
* **`useChirality`** *(bool)*: Incorporates stereochemistry into the bits.
* **`useFeatures`** *(bool)*: If `True`, defines atoms by pharmacophore features (H-bond donors/acceptors) instead of exact atom types, generating FCFP.
* **`useBondTypes`** *(bool)*: Differentiates single, double, and aromatic bonds.
* **`invariants`**: A custom list defining how atomic properties are initially categorized.

### 2. RDKit Topological Fingerprints
Analyzes linear and branched structural paths across the molecule.
* **`minPath`**: Minimum path length (default: `1`).
* **`maxPath`**: Maximum bond path searched (default: `7`).
* **`fpSize`**: Resulting bit vector size.
* **`nBitsPerHash`**: Number of bits activated per unique path.
* **`branchedPaths`** *(bool)*: If `True`, looks at branched topologies rather than strict linear chains.
* **`tgtDensity`**: A target density threshold (between 0.0 and 1.0). The fingerprint folds onto itself until the density of `1`s meets this target.

### 3. Atom-Pair Fingerprints
Captures specific pairwise relationships between standard atom types.
* **`minLength` / `maxLength`**: Determines the range of the topological bond distance connecting the two atoms.
* **`use2D`** *(bool)*: If `True`, measures distance by bond counting. If `False`, it attempts to calculate literal 3D euclidean distance if 3D conformers exist.

### 4. Topological Torsion Fingerprints
Focuses on sequence paths exactly 4 atoms in length.
* **`targetSize`**: The path size targeted (default `4`).

### 5. MACCS Keys
A rigid standard of **166 bits** representing a predefined dictionary of exact structural fragments. 
* **Parameters**: None. 
* **Advantage**: It is completely interpretable. Every bit accurately represents the same fragment across every molecule (e.g., Bit 134 is always a Halogen).

---

## Part 2: Advanced Masking to Break Shortcut Learning

When fingerprints contain the exact variables used to generate target data (like group additivity), the model acts as a calculator. It takes a shortcut by ignoring noisy covariates and simply reconstructing the target via the fingerprint. Pure random masking (Dropout) is rarely enough to break this.

Here are the optimal strategies to break this dependency:

### 1. Adversarial (Feature-Importance) Masking
Target the specific bits the model uses to cheat.
* **Method**: Predict `h298` accurately using a Random Forest trained *only* on fingerprints. Extract the `feature_importances_`.
* **Execution**: In your PyTorch DataLoader, apply highly skewed masking weights. Very predictive bits should be masked at an extremely high rate (e.g., 80% drop chance), forcing the network to lean back onto the noisy starting values.

### 2. Curriculum Masking
Anchor the model's primary reliance on the noisy input early in training.
* **Method**: During Epoch 1, apply extreme dropout to the fingerprint inputs (e.g., 90% or even 100% masking). 
* **Execution**: Gradually decay this mask down to a normal 10% dropout over the first few epochs. This forces the model to structure its initial weights around the noisy variable, using fingerprints only for late-stage refinement.

### 3. MACCS Key / Substructure Masking
Take away the specific components required to make the group additivity mathematical equation work.
* **Method**: Move away from hashed fingerprints to MACCS keys, or generate your own logical array via RDKit SMARTS substructure querying.
* **Execution**: Since elements in these vectors map to specific chemical pieces, logically mask (drop) essential groups known to drive group additivity (e.g., always hide Carboxylic Acids randomly). If the group additivity sum is mathematically missing a variable, it is forced to bridge the gap using the noisy input tensor.

### 4. 1D Block Masking (Path Destruction)
Randomly dropping single bits against highly redundant hashed fingerprints often fails. 
* **Method**: Split the 2048-bit Morgan array into fixed blocks (e.g., 16 blocks of 128 bits). 
* **Execution**: Stochastically select and overwrite entire blocks to `0`. Wiping out massive contiguous chains mathematically restricts easy feature interpolation, serving as much stronger adversarial noise than uniform random dropouts.

### 5. Architectural Bootstrapping (Latent Bottlenecking)
Prevent the neural network from seeing the highly expressive 2048-vector alongside the 1D noisy input on equal footing.
* **Method**: Do not concatenate `[Fingerprint_2048, Noisy_1]` directly.
* **Execution**: Funnel the Fingerprint directly through its own isolated, highly constrained ML layer (e.g., compressing 2048 bits down to a small 8-dimensional tensor). Concatenate that severely compressed latent representation with the noisy input. The 8-dim space will lack the exact numeric accuracy required to perfectly rebuild the group additivity calculations.
