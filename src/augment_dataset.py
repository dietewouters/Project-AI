"""
Dataset augmentation script for molecular property prediction.

Expands a given CSV dataset (smiles, h298) using two chemically valid
augmentation techniques:

1. Mirror molecules (enantiomers) — flips all chiral centres (@  ↔  @@)
   and cis/trans double-bond stereo (E ↔ Z).
   Safe because H298 (standard enthalpy of formation) is a thermodynamic
   property that is *identical* for enantiomers.

2. Tautomer enumeration — generates valid tautomeric forms (e.g. keto ↔ enol,
   imine ↔ enamine) using RDKit's built-in TautomerEnumerator.
   Safe because tautomers in rapid equilibrium share the same thermodynamic
   reference state used by group-additivity H298 calculations.

Usage:
    python src/augment_dataset.py <input.csv> [options]

Options:
    --no-mirror       Skip enantiomer augmentation  (default: on)
    --no-tautomers    Skip tautomer augmentation    (default: on)
    --max-tautomers N Max tautomers per molecule    (default: 5)
    --output PATH     Output CSV path               (default: <input>_augmented.csv)
    --smiles-col COL  SMILES column name            (default: smiles)
    --target-col COL  Target column name            (default: h298)
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

# Suppress RDKit noisy warnings
RDLogger.logger().setLevel(RDLogger.ERROR)


# ---------------------------------------------------------------------------
#  Augmentation helpers
# ---------------------------------------------------------------------------

def mirror_molecule(smiles: str) -> str | None:
    """
    Create the enantiomer (mirror image) of a molecule by inverting all
    tetrahedral chiral centres  (@  ↔  @@)  and E/Z double-bond stereo.

    Returns the canonical SMILES of the enantiomer, or None when:
      - the molecule is invalid
      - the molecule has no stereocentres (mirror == original)
      - flipping produces the same canonical SMILES (meso compounds, etc.)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    has_stereo = False

    # ---- tetrahedral chirality ----
    for atom in mol.GetAtoms():
        chi = atom.GetChiralTag()
        if chi == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            has_stereo = True
        elif chi == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            has_stereo = True

    # ---- cis/trans double-bond stereo ----
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREOZ:
            bond.SetStereo(Chem.BondStereo.STEREOE)
            has_stereo = True
        elif stereo == Chem.BondStereo.STEREOE:
            bond.SetStereo(Chem.BondStereo.STEREOZ)
            has_stereo = True

    if not has_stereo:
        return None

    mirror_smi = Chem.MolToSmiles(mol)
    canonical_original = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    if mirror_smi == canonical_original:
        return None  # meso / pseudo-symmetric molecule

    return mirror_smi


def enumerate_tautomers(smiles: str, max_tautomers: int = 5) -> list[str]:
    """
    Generate up to *max_tautomers* distinct tautomers of the molecule,
    excluding the input SMILES itself.

    Uses RDKit's TautomerEnumerator (rule-based, chemically valid).
    All returned SMILES represent the same compound under different
    proton-transfer arrangements and share the same H298 reference.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    enumerator = rdMolStandardize.TautomerEnumerator()
    # Limit the search space to avoid combinatorial explosion on large mols
    enumerator.SetMaxTautomers(max_tautomers + 1)

    canonical_original = Chem.MolToSmiles(mol)
    tautomers: list[str] = []

    for taut in enumerator.Enumerate(mol):
        smi = Chem.MolToSmiles(taut)
        if smi != canonical_original and smi not in tautomers:
            tautomers.append(smi)
        if len(tautomers) >= max_tautomers:
            break

    return tautomers


# ---------------------------------------------------------------------------
#  Main augmentation pipeline
# ---------------------------------------------------------------------------

def augment_dataset(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    target_col: str = "h298",
    do_mirror: bool = True,
    do_tautomers: bool = True,
    max_tautomers: int = 5,
) -> pd.DataFrame:
    """
    Augment a molecular-property dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *smiles_col* and *target_col*.
    do_mirror : bool
        Add enantiomers for every chiral molecule.
    do_tautomers : bool
        Add tautomeric forms for each molecule.
    max_tautomers : int
        Maximum number of tautomers to add per molecule.

    Returns
    -------
    pd.DataFrame
        Original rows **plus** augmented rows.  A new ``augmentation``
        column labels each row as 'original', 'mirror', or 'tautomer'.
    """
    rows: list[dict] = []
    mirror_count = 0
    tautomer_count = 0

    for _, row in df.iterrows():
        smi = row[smiles_col]
        target = row[target_col]

        # keep the original
        rows.append({smiles_col: smi, target_col: target, "augmentation": "original"})

        # --- 1. mirror (enantiomer) ---
        if do_mirror:
            mirror_smi = mirror_molecule(smi)
            if mirror_smi is not None:
                rows.append({
                    smiles_col: mirror_smi,
                    target_col: target,   # H298 is identical for enantiomers
                    "augmentation": "mirror",
                })
                mirror_count += 1

        # --- 2. tautomers ---
        if do_tautomers:
            for taut_smi in enumerate_tautomers(smi, max_tautomers=max_tautomers):
                rows.append({
                    smiles_col: taut_smi,
                    target_col: target,   # same thermodynamic reference state
                    "augmentation": "tautomer",
                })
                tautomer_count += 1

    aug_df = pd.DataFrame(rows)

    print(f"\n{'='*55}")
    print(f"  Augmentation summary")
    print(f"{'='*55}")
    print(f"  Original rows          : {len(df):>8,}")
    print(f"  + mirror (enantiomers) : {mirror_count:>8,}")
    print(f"  + tautomers            : {tautomer_count:>8,}")
    print(f"  {'─'*35}")
    print(f"  Total rows             : {len(aug_df):>8,}")
    print(f"{'='*55}\n")

    return aug_df


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Augment a molecular-property CSV with enantiomers and tautomers.\n"
            "Both techniques preserve H298 (enthalpy of formation)."
        )
    )
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path.  Default: <input>_augmented.csv",
    )
    parser.add_argument(
        "--smiles-col", type=str, default="smiles",
        help="Name of the SMILES column (default: smiles).",
    )
    parser.add_argument(
        "--target-col", type=str, default="h298",
        help="Name of the target column (default: h298).",
    )
    parser.add_argument(
        "--no-mirror", action="store_true", default=False,
        help="Skip enantiomer augmentation.",
    )
    parser.add_argument(
        "--no-tautomers", action="store_true", default=False,
        help="Skip tautomer augmentation.",
    )
    parser.add_argument(
        "--max-tautomers", type=int, default=5,
        help="Maximum tautomers to add per molecule (default: 5).",
    )

    args = parser.parse_args()

    # Resolve output path
    inp = Path(args.input_csv)
    out = Path(args.output) if args.output else inp.parent / f"{inp.stem}_augmented{inp.suffix}"

    # Load & validate
    df = pd.read_csv(inp)
    for col in (args.smiles_col, args.target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {inp}")

    print(f"Loaded {len(df):,} rows from {inp}")

    # Augment
    aug_df = augment_dataset(
        df,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        do_mirror=not args.no_mirror,
        do_tautomers=not args.no_tautomers,
        max_tautomers=args.max_tautomers,
    )

    # Save
    aug_df.to_csv(out, index=False)
    print(f"Saved augmented dataset → {out}")


if __name__ == "__main__":
    main()
