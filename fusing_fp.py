import warnings
from tqdm import tqdm
import tmap as tm
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from mapchiral.mapchiral import encode as mapc_enc
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_smiles(smiles, encoder):
    """
    Helper function to encode a single SMILES string based on the specified encoder.

    Args:
    smiles (str): The SMILES string to encode.
    encoder (str): The encoding method, 'mhfp' or 'mapc'.
    kwargs: Additional arguments for the encoder.

    Returns:
    tm.VectorUint: Encoded fingerprint.
    """
    if encoder == 'mapc':
        return tm.VectorUint(mapc_enc(Chem.MolFromSmiles(smiles), max_radius=2, n_permutations=2048, mapping=False))
    elif encoder == 'mhfp':
        perm = 512
        mhfp_enc = MHFPEncoder(perm)
        return tm.VectorUint(mhfp_enc.encode(smiles))
    return None

def calculate_fingerprints(df, encoder):
    """
    Calculate fingerprints for SMILES strings in the dataframe.
    
    Args:
    df (pandas.DataFrame): DataFrame containing 'canonical_smiles' column.
    encoder (str): The encoder object used to encode SMILES strings ('mhfp' or 'mapc').
    
    Returns:
    tuple: (list of encoded fingerprints, list of valid indices).
    """
    fingerprints = []
    valid_indices = []

    for idx, smiles in enumerate(tqdm(df['canonical_smiles'], desc=f"Calculating fingerprints ({encoder})")):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                enc_fingerprint = encode_smiles(smiles, encoder)
                if w and issubclass(w[-1].category, UserWarning):
                    logger.warning(f"Warning for Fingerprints {smiles}: {str(w[-1].message)}")
                    continue
            fingerprints.append(enc_fingerprint)
            valid_indices.append(idx)
        except Exception as e:
            logger.error(f"Error encoding Fingerprints {smiles}: {str(e)}")
        
    return fingerprints, valid_indices

def list_to_vectorUint(lst):
    """
    Convert list or numpy array to tm.VectorUint type.
    
    Args:
    lst (list or np.array): The list or array to convert.
    
    Returns:
    tm.VectorUint: Converted VectorUint object.
    """
    return tm.VectorUint(lst)

def minhash_fingerprints(df, fingerprints, valid_indices):
    """
    Combine fingerprints by taking the minimum values for identical targets.
    
    Args:
    df (pandas.DataFrame): The DataFrame with target IDs.
    fingerprints (list): List of tm.VectorUint fingerprints.
    valid_indices (list): Indices of valid entries in the original DataFrame.
    
    Returns:
    pandas.Series: Combined fingerprints as tm.VectorUint.
    """
    df_processed = df.loc[valid_indices].copy()
    df_processed['fingerprint_vector'] = pd.Series(fingerprints, index=valid_indices)
    df_processed['fingerprint_vector'] = df_processed['fingerprint_vector'].apply(np.array)

    result = df_processed.groupby('Target_ID').agg({
        'fingerprint_vector': lambda x: np.min(np.vstack(x), axis=0).tolist(),
        **{col: 'first' for col in df_processed.columns if col not in ['fingerprint_vector', 'Target_ID']}
    }).reset_index()

    result['fingerprint_vector'] = result['fingerprint_vector'].apply(list_to_vectorUint)
    return result['fingerprint_vector']

def safe_create_categories(series):
    """
    Create categories from a pandas Series, handling NaN values.
    
    Args:
    series (pandas.Series): The input data series.
    
    Returns:
    tuple: (labels, data) for Faerun plotting.
    """
    series = series.fillna('Unknown').astype(str)
    return Faerun.create_categories(series)

def plot_faerun(x, y, s, t, df):
    """
    Plot the data using Faerun.
    
    Args:
    x (list): X coordinates.
    y (list): Y coordinates.
    s (list): Source nodes for tree plot.
    t (list): Target nodes for tree plot.
    df (pandas.DataFrame): DataFrame with target data.
    """
    f = Faerun(view="front", coords=False, clear_color="#FFFFFF")

    # Create categories
    target_id_labels, target_id_data = safe_create_categories(df['Target_ID'])
    protein_class_labels, protein_class_data = safe_create_categories(df['target_protein_class'])
    taxonomy_labels, taxonomy_data = safe_create_categories(df['Target_Taxonomy'])
    organism_labels, organism_data = safe_create_categories(df['Target_organism'])
    target_type_labels, target_type_data = safe_create_categories(df['Target_type'])

    # Add scatter plot
    f.add_scatter(
        "Target_Data",
        {
            "x": x,
            "y": y,
            "c": [protein_class_data, taxonomy_data, organism_data, target_type_data],
            "labels": df['Target_ID'].fillna('Unknown').astype(str)
        },
        shader="sphere",
        point_scale=5,
        max_point_size=20,
        legend_labels=[protein_class_labels, taxonomy_labels, organism_labels, target_type_labels],
        categorical=[True, True, True, True],
        colormap=['tab20', 'tab20', 'tab20', 'tab20'],
        series_title=['Target ID', 'Protein Class', 'Taxonomy', 'Organism', 'Target Type'],
        has_legend=True,
    )

    # Add tree
    f.add_tree("Target_Data_tree", {"from": s, "to": t}, point_helper="Target_Data")
    
    # Plot
    f.plot('Target_Data', template='smiles')

def main():
    csv_file = r'C:\Users\biolab\Desktop\Alex\tmap_fused\alex_dataset.csv'
    df = pd.read_csv(csv_file)

    # Calculate fingerprints
    fingerprints, valid_indices = calculate_fingerprints(df, 'mapc')

    # Filter DataFrame
    df = df.iloc[valid_indices].reset_index(drop=True)

    # Combine fingerprints
    logger.info('Combining fingerprints')
    fused_fingerprints = minhash_fingerprints(df, fingerprints, valid_indices)
    logger.info(f'Combination successful. Total fused fingerprints: {len(fused_fingerprints)}')

    # TMAP layout and indexing
    logger.info('Indexing...')
    lf = tm.LSHForest(512)
    lf.batch_add(fused_fingerprints)
    lf.index()

    cfg = tm.LayoutConfiguration()
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 40
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 10
    cfg.k = 20
    cfg.sl_scaling_type = tm.RelativeToAvgLength
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)

    logger.info('Plotting...')
    plot_faerun(x, y, s, t, df)

if __name__ == "__main__":
    start_time = timer()
    main()
    end_time = timer()
    logger.info('TMAP successfully generated.')
    print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")
