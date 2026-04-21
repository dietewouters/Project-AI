import pandas as pd
import os

def test_mapping():
    # 1. Create a tiny mock environment
    os.makedirs("test_verify", exist_ok=True)
    
    # Global target (10 rows)
    full_target = pd.DataFrame({'smiles': [f'M{i}' for i in range(10)], 'val': range(10)})
    full_target.to_csv("test_verify/full.csv", index=False)

    # Quarter 1 starts at row 5
    start_row = 5
    quarter_noisy = full_target.iloc[start_row:].copy()
    quarter_noisy['val'] = quarter_noisy['val'] + 100 # Offset to identify noisy file
    quarter_noisy.to_csv("test_verify/noisy_q1.csv", index=False)

    # 2. Perform the logic check exactly as Stage 4 does
    global_index = 7
    index_offset = start_row

    # Load files
    t_df = pd.read_csv("test_verify/full.csv")
    n_df = pd.read_csv("test_verify/noisy_q1.csv")

    # Mapping
    local_index = global_index - index_offset

    print(f"Goal: Retrieve Molecule {global_index}")
    print(f"---")
    print(f"Target File (Full) | iloc[{global_index}]   -> Value: {t_df.iloc[global_index]['val']}")
    print(f"Noisy File (Part)  | iloc[{local_index}]   -> Value: {n_df.iloc[local_index]['val']}")

    if t_df.iloc[global_index]['val'] == 7 and n_df.iloc[local_index]['val'] == 107:
        print("\n✅ LOGIC VERIFIED: Global index 7 maps perfectly to local index 2 in the partial file.")
    else:
        print("\n❌ LOGIC ERROR: Mismatch in mapping.")

if __name__ == "__main__":
    test_mapping()
