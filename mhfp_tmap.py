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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def encode_fp(fingerprint, encoder):
    """
    Encode a SMILES string into a fingerprint.
    """
    try:
        with warnings.catch_warnings(record=True) as w: 
            warnings.simplefilter("always") 
            enc_fingerprint = tm.VectorUint(encoder.encode(fingerprint))
            if w and issubclass(w[-1].category, UserWarning):
                print(f"Warning for Fingerprints {fingerprint}: {str(w[-1].message)}")
                return None
        return enc_fingerprint
    except Exception as e:
        print(f"Error encoding Fingerprints {fingerprint}: {str(e)}")
        return None


def calculate_fingerprints2(df, encoder, fingerprints_file):
    """
    Calculate fingerprints for SMILES strings, using a pickle file if available.
    """
    if os.path.exists(fingerprints_file):
        print("Loading DataFrame with fingerprints from pickle file...")
        return pd.read_pickle(fingerprints_file)

    print("Calculating fingerprints...")
    tqdm.pandas(desc="Encoding SMILES")
    df['fingerprint'] = df['canonical_smiles'].progress_apply(lambda x: encode_smiles(x, encoder))
    
    # Remove rows with None fingerprints
    df = df.dropna(subset=['fingerprint']).reset_index(drop=True)

    # Save DataFrame to pickle file
    df.to_pickle(fingerprints_file)

    return df


def calculate_fingerprints(df, encoder, fingerprints_file):
    """
    Calculate fingerprints for SMILES strings, using a pickle file if available.
    """

    # Apparently VectorUint objects are not pickable, instead I will save the new dataset with calculated fingerprints 
    """
    if os.path.exists(fingerprints_file):
        print("Loading fingerprints from pickle file...")
        with open(fingerprints_file, 'rb') as f:
            return pickle.load(f)
    """

    fingerprints = []
    valid_indices = []
    for idx, smiles in enumerate(tqdm(df['canonical_smiles'], desc="Calculating fingerprints from SMILES")):
        fingerprint = encode_fp(smiles, encoder)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
            valid_indices.append(idx)

    """    
    # Save fingerprints to pickle file
    with open(fingerprints_file, 'wb') as f:
        pickle.dump(fingerprints, f)
    """
    return fingerprints, valid_indices

def main():
    # File paths
    csv_file = r'C:\Users\biolab\Desktop\Alex\Chembl_data\alex_dataset.csv'
    fingerprints_file = 'fingerprints.pkl'

    # Load data
    df = pd.read_csv(csv_file)

    # Initialize encoder
    perm = 512  # Number of permutations used by the MinHashing algorithm
    enc = MHFPEncoder(perm)

    # Calculate or load fingerprints
    fingerprints, valid_indices = calculate_fingerprints(df, enc, fingerprints_file)

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

    print("\n")
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
    print('\n')
    print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")