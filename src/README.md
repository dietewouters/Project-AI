# MoleculeDataset — Design Notes

MoleculeDataset is a PyTorch Dataset that loads molecule data from two CSV files — one containing the ECFP fingerprints and clean enthalpy values, the other containing molecule names and noisy enthalpy values. The two files are merged by row position and renamed to a consistent internal schema (molecule, fingerprint, input, target).
An optional indices file can be passed to slice the dataset into a specific split (train, val, or test) without duplicating data on disk. An optional scaler (e.g. StandardScaler) can be passed to normalise the input and target values, and must be fitted on the training set only before being passed to the validation and test datasets.
Each call to __getitem__ returns a concatenated input vector of shape (2049,) — 2048 fingerprint bits followed by the noisy enthalpy — paired with the clean target value.

The `.merge(on="name")` is important — it guarantees rows are aligned even if the two files are sorted differently, and will immediately crash if something is mismatched.

---

## 2. Purpose of the Neighbour Value

When added later, the neighbour value represents:

> *"Find the molecule most chemically similar to the current one that has a **known, clean** target value, and give that value to the model as a hint."*

This works because molecules with similar fingerprints tend to have similar properties (**QSAR principle**). So instead of predicting ΔHf from scratch, the model effectively learns:

```
predicted ΔHf ≈ neighbour ΔHf + correction(fingerprint, noisy ΔHf)
```

This shifts the problem from **pure regression** to **refinement**, which is a much easier task. The neighbour acts as an anchor — especially useful when the noisy value is very corrupted.

In practice this is computed via **k-nearest neighbours in fingerprint space** (e.g. Tanimoto similarity on ECFP), run once as a preprocessing step before training.

---

## Input vector summary

| Component | Dimensions | Description |
|---|---|---|
| `fp` | 2048 | ECFP fingerprint — encodes molecular structure |
| `noisy` | 1 | Noisy ΔHf — corrupted version of the target |
| `neighbour` *(future)* | 1 | ΔHf of the most similar known molecule |
| **Total** | **2049 → 2050** | Concatenated input to the MLP |

## To add 
- [ ] k-nn 
- [ ] use embedding of sparse vector 
- [ ] explore state of art 
- 