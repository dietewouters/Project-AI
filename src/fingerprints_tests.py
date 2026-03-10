from fingerprints import smiles_to_fingerprint
import numpy as np


# Tests for smiles to fingerprint 

def test_returns_numpy_array():
    fp = smiles_to_fingerprint("CCO")
    assert isinstance(fp, np.ndarray)


def test_output_shape_matches_n_bits():
    fp = smiles_to_fingerprint("CCO", length=1024)
    assert fp.shape == (1024,)


def test_output_dtype_is_float32():
    fp = smiles_to_fingerprint("CCO")
    assert fp.dtype == np.float32


def test_fingerprint_is_binary():
    fp = smiles_to_fingerprint("CCO")
    assert np.all(np.isin(fp, [0.0, 1.0]))


def test_same_smiles_gives_same_fingerprint():
    fp1 = smiles_to_fingerprint("CCO")
    fp2 = smiles_to_fingerprint("CCO")
    assert np.array_equal(fp1, fp2)


def test_different_smiles_give_different_fingerprints():
    fp1 = smiles_to_fingerprint("CCO")
    fp2 = smiles_to_fingerprint("CCN")
    assert not np.array_equal(fp1, fp2)


def test_invalid_smiles_raises_value_error():
    try:
        smiles_to_fingerprint("not_a_smiles")
    except ValueError as e:
        assert "Invalid SMILES string" in str(e)
    else:
        assert False, "ValueError was not raised for invalid SMILES"


def test_radius_changes_fingerprint():
    fp1 = smiles_to_fingerprint("CCC(CC)CO", radius=1)
    fp2 = smiles_to_fingerprint("CCC(CC)CO", radius=3)
    assert not np.array_equal(fp1, fp2)


def test_n_bits_changes_output_length():
    fp1 = smiles_to_fingerprint("CCO", length=512)
    fp2 = smiles_to_fingerprint("CCO", length=2048)

    assert fp1.shape == (512,)
    assert fp2.shape == (2048,)


def test_nonempty_molecule_has_at_least_one_active_bit():
    fp = smiles_to_fingerprint("CCO")
    assert fp.sum() > 0


# Simple test runner
if __name__ == "__main__":
    tests = [
        test_returns_numpy_array,
        test_output_shape_matches_n_bits,
        test_output_dtype_is_float32,
        test_fingerprint_is_binary,
        test_same_smiles_gives_same_fingerprint,
        test_different_smiles_give_different_fingerprints,
        test_invalid_smiles_raises_value_error,
        test_radius_changes_fingerprint,
        test_n_bits_changes_output_length,
        test_nonempty_molecule_has_at_least_one_active_bit,
    ]

    for test in tests:
        test()
        print(f"{test.__name__} passed")

    print("\nAll tests passed!")