import warnings
from tqdm import tqdm
import tmap as tm
import pandas as pd
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
    
def list_to_vectorUint(lst):
    '''
    This function changes lists or np.arrays to tm.VectorUint type of object
    '''
    return tm.VectorUint(lst)

def minhash_fingerprints(df, fingerprints, valid_indices):
    '''
    This function takes the fingerprints as tm.VectorUint type and transforms it to an array. Then merges all fingerprints with same target taking the minumum value at each index in the vector. 
    '''
    # Create a copy of the original DataFrame, filtered by valid_indices
    df_processed = df.loc[valid_indices].copy()
    
    # Add the fingerprint vectors as a new column
    df_processed['fingerprint_vector'] = pd.Series(fingerprints, index=valid_indices)

    # Convert fingerprint Vector Uint to numpy arrays to be able to perform the calculations
    df_processed['fingerprint_vector'] = df_processed['fingerprint_vector'].apply(np.array)

    # Group by Target_ID and find minimum values for every fingerprint with same Target_ID
    result = df_processed.groupby('Target_ID').agg({
        'fingerprint_vector': lambda x: np.min(np.vstack(x), axis=0).tolist(),
        **{col: 'first' for col in df_processed.columns if col != 'fingerprint_vector' and col != 'Target_ID'}
    }).reset_index()

    result.to_csv('df_with_combined_fp.csv', index=False)
    # Convert np.array back to tm.VectorUint to be able to perform TMAP
    result['fingerprint_vector'] = result['fingerprint_vector'].apply(list_to_vectorUint)
    
    # Save the results
    # result.to_csv('minhashed_fingerprints2.csv', index=False)
    return result['fingerprint_vector']

def main():
    # File paths
    csv_file = r'C:\Users\biolab\Desktop\Alex\tmap_fused\alex_dataset.csv'

    # Load data
    df = pd.read_csv(csv_file)

    # Initialize encoder
    perm = 512  # Number of permutations used by the MinHashing algorithm

    # Calculate or load fingerprints
    fingerprints, valid_indices = calculate_fingerprints(df, 'mapc')

    # Update DataFrame with valid entries only
    df = df.iloc[valid_indices].reset_index(drop=True)

    # Combine fingerprints for all fingerprints with same target. 
    print('Combining fingerprints')
    fused_fingerprints = minhash_fingerprints(df, fingerprints, valid_indices)
    print(f'Combination successful. Total of fused fingerprints is now: {len(fused_fingerprints)}')

    # LSH Indexing and coordinates generation
    print('Indexing...')
    lf = tm.LSHForest(perm)
    lf.batch_add(fused_fingerprints)
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

    # Function to safely create categories
    def safe_create_categories(series):
        # Convert all values to strings, replacing NaN with 'Unknown'
        series = series.fillna('Unknown').astype(str)
        return Faerun.create_categories(series)
    
    # Create categories for each of the specified columns
    target_id_labels, target_id_data = safe_create_categories(df['Target_ID'])
    protein_class_labels, protein_class_data = safe_create_categories(df['target_protein_class'])
    taxonomy_labels, taxonomy_data = safe_create_categories(df['Target_Taxonomy'])
    organism_labels, organism_data = safe_create_categories(df['Target_organism'])
    target_type_labels, target_type_data = safe_create_categories(df['Target_type'])
    
    # Create Faerun object
    f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
    
    # Add scatter plot
    f.add_scatter(
        "Target_Data",
        {
            "x": x,
            "y": y,
            "c": [
                  protein_class_data,
                  taxonomy_data,
                  organism_data,
                  target_type_data],
            "labels": df['Target_ID'].fillna('Unknown').astype(str),  # Using Target_ID as labels, converted to string
        },
        shader="sphere",
        point_scale=5,
        max_point_size=20,
        legend_labels=[ 
                       protein_class_labels, 
                       taxonomy_labels, 
                       organism_labels, 
                       target_type_labels],
        categorical=[True, True, True, True, True],
        colormap=['tab20', 'tab20', 'tab20', 'tab20', 'tab20'],  
        series_title=['Target ID', 'Protein Class', 'Taxonomy', 'Organism', 'Target Type'],
        has_legend=True,
    )
    
    # Add tree
    f.add_tree("Target_Data_tree", {"from": s, "to": t}, point_helper="Target_Data")
    
    # Plot
    f.plot('Target_Data', template='smiles')
if __name__ == "__main__":
    start_time = timer()
    main()
    end_time = timer()
    print('TMAP successfully generated.')
    print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")
