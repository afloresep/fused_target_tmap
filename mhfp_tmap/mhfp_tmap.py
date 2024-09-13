import warnings
from tqdm import tqdm
import tmap as tm
import pandas as pd
import os
import pickle
import warnings
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import tmap as tm 
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from tqdm import tqdm


def calculate_fingerprints(df, encoder):
    """
    Calculate fingerprints for SMILES strings in the dataframe.
    
    Args:
    df (pandas.DataFrame): DataFrame containing 'canonical_smiles' column
    encoder: The encoder object used to encode SMILES strings (e.g. MHFP)
    
    Returns:
    tuple: (list of encoded fingerprints, list of valid indices)
    """
    fingerprints = []
    valid_indices = []
    
    for idx, smiles in enumerate(tqdm(df['canonical_smiles'], desc="Calculating fingerprints from SMILES")):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                enc_fingerprint = tm.VectorUint(encoder.encode(smiles))
                if w and issubclass(w[-1].category, UserWarning):
                    print(f"Warning for Fingerprints {smiles}: {str(w[-1].message)}")
                    continue
            fingerprints.append(enc_fingerprint)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error encoding Fingerprints {smiles}: {str(e)}")
    
    return fingerprints, valid_indices

def main():
    # File paths
    csv_file = r'path_to_csv.csv'

    # Load data
    df = pd.read_csv(csv_file)

    # Initialize encoder
    perm = 512  # Number of permutations used by the MinHashing algorithm
    mhfp_enc = MHFPEncoder(perm)

    # Calculate or load fingerprints
    fingerprints, valid_indices = calculate_fingerprints(df, mhfp_enc)

    # Update DataFrame with valid entries only
    df = df.iloc[valid_indices].reset_index(drop=True)

    # LSH Indexing and coordinates generation
    print('Indexing...')
    lf = tm.LSHForest(perm)
    lf.batch_add(fingerprints)
    lf.index()

    # Get the coordinates and Layout Configuration
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 40
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 10
    cfg.k = 20
    cfg.sl_scaling_type = tm.RelativeToAvgLength
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)

    print("Plotting...")
    # Plotting
    f = Faerun(
        view="front",
        coords=False,
        title="",
        clear_color="#FFFFFF",
    )

    f.add_scatter(
        "mhfp_tmap_node140_TMAP",
        {
            "x": x,
            "y": y,
            "c": df['standard_value'].tolist(),
            "labels": df['canonical_smiles'],
        },
        point_scale=3,
        colormap=['rainbow'],
        has_legend=True,
        legend_title=['Standard_value'],
        categorical=[False],
        shader='smoothCircle'
    )

    f.add_tree("mhfp_tmap_node140_TMAP_tree", {"from": s, "to": t}, point_helper="mhfp_tmap_node140_TMAP")
    f.plot('mhfp_tmap_node140_TMAP', template='smiles')

if __name__ == "__main__":
    start_time = timer()
    main()
    end_time = timer()
    print('TMAP successfully generated.')
    print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")
