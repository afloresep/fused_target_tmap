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
from mapchiral.mapchiral import encode


def calculate_fingerprints(df, encoder):
    """
    Calculate fingerprints for SMILES strings in the dataframe.
    
    Args:
    df (pandas.DataFrame): DataFrame containing 'canonical_smiles' column
    encoder: The encoder object used to encode SMILES strings (e.g. 'mhfp' or 'mapc)
    
    Returns:
    tuple: (list of encoded fingerprints, list of valid indices)
    """
    fingerprints = []
    valid_indices = []
    
    from mapchiral.mapchiral import encode as mapc_enc

    if encoder =='mapc':
        for idx, smiles in enumerate(tqdm(df['canonical_smiles'], desc="Calculating fingerprints from SMILES")):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    enc_fingerprint = tm.VectorUint(encode(Chem.MolFromSmiles(smiles), max_radius=2, n_permutations=2048, mapping=False))
                    if w and issubclass(w[-1].category, UserWarning):
                        print(f"Warning for Fingerprints {smiles}: {str(w[-1].message)}")
                        continue
                fingerprints.append(enc_fingerprint)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error encoding Fingerprints {smiles}: {str(e)}")
        
        return fingerprints, valid_indices
    
    
    if encoder =='mhfp':
        # Initialize encoder
        perm = 512  # Number of permutations used by the MinHashing algorithm
        mhfp_enc = MHFPEncoder(perm)

        for idx, smiles in enumerate(tqdm(df['canonical_smiles'], desc="Calculating fingerprints from SMILES")):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    enc_fingerprint = tm.VectorUint(mhfp_enc.encode(smiles))
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
    csv_file = r'C:\Users\biolab\Desktop\Alex\tmap_fused\alex_dataset.csv'

    # Load data
    df = pd.read_csv(csv_file)
    df_shortened = df.head(1000)

    # Initialize encoder
    perm = 512  # Number of permutations used by the MinHashing algorithm

    # Calculate or load fingerprints
    fingerprints, valid_indices = calculate_fingerprints(df_shortened, 'mapc')

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
        "mapc_tmap_node140_TMAP",
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
    f.add_tree("mapc_tmap_node140_TMAP_tree", {"from": s, "to": t}, point_helper="mapc_tmap_node140_TMAP")
    f.plot('mapc_tmap_node140_TMAP', template='smiles')

if __name__ == "__main__":
    start_time = timer()
    main()
    end_time = timer()
    print('TMAP successfully generated.')
    print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")
