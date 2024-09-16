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
import pickle
from pathlib import Path


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

def map_protein_class(value):
    if pd.isna(value):
        return np.nan
    
    value = value.lower().strip()
    
    if 'enzyme' in value:
        if 'kinase' in value:
            return 'kinase'
        elif 'protease' in value:
            return 'protease'
        elif 'cytochrome p450' in value:
            return 'cytochrome p450'
        else:
            return 'enzyme'
    elif 'ion channel' in value:
        return 'ion channel'
    elif 'transporter' in value:
        return 'transporter'
    elif 'transcription factor' in value:
        return 'transcription factor'
    elif 'membrane receptor' in value:
        return 'membrane receptor'
    elif 'epigenetic regulator' in value:
        return 'epigenetic regulator'
    else:
        return 'other'  # Default category for unmatched classes
    
def map_target_taxonomy(value): 
    if pd.isna(value):
        return np.nan
    value = value.lower().strip()
    
    if 'enzyme' in value:
        return 'Enzyme'    
    elif 'receptor' in value:
        return 'Receptor'
    elif 'transcription factor' in value:
        return 'Transcription Factor'
    elif 'nuclear hormone receptor' in value:
        return 'Nuclear Hormone Receptor'
    elif 'calcium channel' in value:
        return 'Calcium Channel'
    elif 'surface antigen' in value:
        return 'Surface Antigen'
    elif 'fungi' in value or 'viruses' in value or 'bacteria' in value:
        return 'Microorganism'
    elif 'eukaryotes' in value:
        return 'Eukaryotes'
    elif 'subcellular' in value:
        return 'Subcellular Component'
    elif 'small molecule' in value:
        return 'Small Molecule'
    elif 'lipid' in value:
        return 'Lipid'
    elif 'cellline' in value:
        return 'Cell Line'
    elif 'nucleic acid' in value:
        return 'Nucleic Acid'
    else:
        return 'Other'  # Default category for unmatched classes 

def map_target_organism(value):
    if 'sapiens' in value:
        return 'Homo sapiens'
    elif 'virus' in value:
        return 'Virus'
    elif 'rattus' in value or 'Musculus' in value:
        return 'Rat'
    elif 'taurus' in value:
        return 'Bovid'
    elif 'scrofa' in value or 'Macaca' in value or 'porcellus' in value or 'oryctolagus' in value or 'canis' in value or 'Cricetulus' in value:
        return 'Other mammals'
    elif 'Mycobacterium' in value or 'Escherichia' in value or 'Salmonella' in value or 'Staphylococcus' in value or 'Pseudomonas' in value or 'Bacillus' in value or 'Acinetobacter' in value:
        return 'Bacteria'
    elif 'Plasmodium' in value or 'Trypanosoma' in value or 'Schistosoma' in value or 'Leishmania' in value:
        return 'Parasites'
    else:
        return 'Others'
    
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
    protein_class_labels, protein_class_data = safe_create_categories(df['mapped_protein_class'])
    taxonomy_labels, taxonomy_data = safe_create_categories(df['map_target_taxonomy'])
    target_organism_labes, target_organism_data = safe_create_categories(df['map_target_organism'])
    target_type_labels, target_type_data = safe_create_categories(df['Target_type'])

    # Add scatter plot
    f.add_scatter(
        "cached_nice_labels",
        {
            "x": x,
            "y": y,
            "c": [protein_class_data, taxonomy_data, target_type_data, target_organism_data],
            "labels": df['Target_ID'].fillna('Unknown').astype(str)
        },
        shader="sphere",
        point_scale=5,
        max_point_size=20,
        legend_labels=[protein_class_labels, taxonomy_labels, target_type_labels, target_organism_labes],
        categorical=[True, True, True, True],
        colormap="tab10",
        series_title=['Protein Class', 'Target Taxonomy', 'Target Type', 'Target Organism'],
        has_legend=True,
    )

    # Add tree
    f.add_tree("cached_nice_labels_tree", {"from": s, "to": t}, point_helper="cached_nice_labels", color="#222222")
    
    # Plot
    f.plot("cached_nice_labels", template='smiles')

def main():
    csv_file = r'C:\Users\biolab\Desktop\Alex\tmap_fused\alex_dataset.csv'
    df = pd.read_csv(csv_file)
    
    # Define the path for saving/loading fingerprints
    fingerprints_file = Path(r'C:\Users\biolab\Desktop\Alex\tmap_fused\fused_fingerprints.pkl')
    
    if fingerprints_file.exists():
        logger.info('Loading pre-calculated fingerprints')
        with open(fingerprints_file, 'rb') as f:
            data = pickle.load(f)
        fused_fingerprints = data['fused_fingerprints']
        valid_indices = data['valid_indices']
        
    else:
        logger.info('Calculating fingerprints')
        # Calculate fingerprints
        fingerprints, valid_indices = calculate_fingerprints(df, 'mapc')

        # Combine fingerprints
        logger.info('Combining fingerprints')
        fused_fingerprints = minhash_fingerprints(df, fingerprints, valid_indices) 
        fused_fingerprints = fused_fingerprints.apply(list_to_vectorUint)

        logger.info(f'Combination successful. Total fused fingerprints: {len(fused_fingerprints)}')
        
        # Save the calculated fingerprints and valid_indices
        with open(fingerprints_file, 'wb') as f:
            pickle.dump({
                'fused_fingerprints': [list(fp) for fp in fused_fingerprints],
                'valid_indices': valid_indices
            }, f)
        logger.info('Saved calculated fingerprints for future use')

    # Filter DataFrame
    df = df.iloc[valid_indices].reset_index(drop=True)
    # Apply the mapping function to the column to reduce number of unique values and make it color codeable 
    df['mapped_protein_class'] = df['target_protein_class'].apply(map_protein_class)
    df['map_target_taxonomy'] = df['Target_Taxonomy'].apply(map_target_taxonomy)
    df['map_target_organism'] = df['Target_organism'].apply(map_target_organism)

    # TMAP layout and indexing
    logger.info('Indexing...')
    lf = tm.LSHForest(512, 128, store=True)
    
    # Convert back to VectorUint if loading from file
    if isinstance(fused_fingerprints[0], list):
        fused_fingerprints = [tm.VectorUint(fp) for fp in fused_fingerprints]
    
    lf.batch_add(fused_fingerprints)
    lf.index()

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
