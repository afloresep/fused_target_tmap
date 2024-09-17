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
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

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

def calculate_threshold(data):
    # Function to calculate threshold using IQR method
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    return threshold

def calculate_molecular_properties(smiles):
    """
    Calculate molecular properties using RDKit.
    
    Args:
    smiles (str): SMILES string of the molecule.
    
    Returns:
    tuple: (HAC, fraction_aromatic_atoms, number_of_rings, clogP, fraction_Csp3)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    hac = mol.GetNumHeavyAtoms()
    num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    fraction_aromatic_atoms = num_aromatic_atoms / hac if hac > 0 else 0
    number_of_rings = rdMolDescriptors.CalcNumRings(mol)
    clogP = Descriptors.MolLogP(mol)
    fraction_Csp3 = Descriptors.FractionCSP3(mol)
    
    return (hac, fraction_aromatic_atoms, number_of_rings, clogP, fraction_Csp3)

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
    return Faerun.create_categories(series.fillna('Unknown').astype(str))

def map_protein_class(value):
    """Map protein class to a simplified category."""
    if pd.isna(value):
        return np.nan
    
    value = value.lower().strip()
    
    if 'enzyme' in value:
            return 'Enzyme'
    elif 'membrane receptor' in value: 
        return 'Membrane receptor'
    elif ' ion channel' in value:
        return 'Ion Channel'
    elif 'transcription factor' in value: 
        return 'Transcription factor'
    elif 'epigenetic regulator' in value:
        return 'Epigenetic regulator'
    elif 'cytosolic protein' in value:
        return 'cytosolic protein'
    else:
        return 'Other'

def map_target_taxonomy(value):
    if 'Eukaryotes' in value:
        if 'Oxidoreductase' in value:
            return 'Oxidoreductase'
        elif 'Transferase' in value:
            return 'Transferase' 
        elif 'Hydrolase' in value:
            return 'Hydrolase'
        else:
            return 'Eukaryotes'
    elif 'Bacteria' in value: 
        return 'Bacteria'
    elif 'Fungi' in value:
        return 'Fungi'
    elif 'Viruses' in value: 
        return 'Viruses'
    elif 'unclassified' in value:
        return 'Unclassified'
    else:
        return 'Other'

def map_target_organism(value):
    """Map target organism to a simplified category."""
    if 'sapiens' in value:
        return 'Homo sapiens'
    elif 'virus' in value:
        return 'Virus'
    elif any(organism in value for organism in ['rattus', 'Musculus']):
        return 'Rat'
    elif 'taurus' in value:
        return 'Bovid'
    elif any(organism in value for organism in ['scrofa', 'Macaca', 'porcellus', 'oryctolagus', 'canis', 'Cricetulus']):
        return 'Other mammals'
    elif any(bacteria in value for bacteria in ['Mycobacterium', 'Escherichia', 'Salmonella', 'Staphylococcus', 'Pseudomonas', 'Bacillus', 'Acinetobacter']):
        return 'Bacteria'
    elif any(parasite in value for parasite in ['Plasmodium', 'Trypanosoma', 'Schistosoma', 'Leishmania']):
        return 'Parasites'
    else:
        return 'Others'

def select_value(group):
    """Select the correct value based on the 'standard_type' category."""
    greater_value_terms = ['Activity', 'Inhibition', 'Potency', '% Inhibition', 'Percent Effect']
    
    # Check if the standard_type is in the greater_value_terms
    if group['standard_type'].iloc[0] in greater_value_terms:
        # If the standard_type is in the list, select the row with the maximum standard_value
        return group.loc[group['standard_value'].idxmax()]
    else:
        # Otherwise, select the row with the minimum standard_value
        return group.loc[group['standard_value'].idxmin()]

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
    organism_labels, organism_data = safe_create_categories(df['map_target_organism'])

    labels = []
    hac_data = []
    frac_aromatic_data = []
    num_rings_data = []
    clogp_data = []
    frac_csp3_data = []

    for i, row in df.iterrows():
        labels.append(
                row['canonical_smiles']
                + '__'
                + f'<a target="_blank" href="https://www.ebi.ac.uk/chembl/target_report_card/{row["Target_ID"]}">{row["Target_ID"]}</a><br>'
            )
        
        # Calculate molecular properties
        properties = calculate_molecular_properties(row['canonical_smiles'])
        if properties:
            hac, frac_aromatic, num_rings, clogp, frac_csp3 = properties
            hac_data.append(hac)
            frac_aromatic_data.append(frac_aromatic)
            num_rings_data.append(num_rings)
            clogp_data.append(clogp)
            frac_csp3_data.append(frac_csp3)
        else:
            # Handle invalid SMILES
            hac_data.append(None)
            frac_aromatic_data.append(None)
            num_rings_data.append(None)
            clogp_data.append(None)
            frac_csp3_data.append(None)
   
    # Calculate threshold for hac_data using IQR
    hac_threshold = calculate_threshold(hac_data)
    frac_threshold = calculate_threshold(frac_aromatic_data)
    rings_threshold = calculate_threshold(num_rings_data)
    clogp_threshold = calculate_threshold(clogp_data)
    csp3_threshold = calculate_threshold(frac_csp3_data)

    # Function to apply thresholds and return filtered data
    def apply_thresholds(hac_data, frac_aromatic_data, num_rings_data, clogp_data, frac_csp3_data):
        filtered_hac = []
        filtered_frac_aromatic = []
        filtered_num_rings = []
        filtered_clogp = []
        filtered_frac_csp3 = []

        # Iterate through all data points and apply thresholds
        for hac, frac, rings, clogp, csp3 in zip(hac_data, frac_aromatic_data, num_rings_data, clogp_data, frac_csp3_data):
            if hac <= hac_threshold and frac <= frac_threshold and rings <= rings_threshold and clogp <= clogp_threshold and csp3 <= csp3_threshold:
                filtered_hac.append(hac)
                filtered_frac_aromatic.append(frac)
                filtered_num_rings.append(rings)
                filtered_clogp.append(clogp)
                filtered_frac_csp3.append(csp3)

        return filtered_hac, filtered_frac_aromatic, filtered_num_rings, filtered_clogp, filtered_frac_csp3

    filtered_hac,filtered_frac_aromatic, filtered_num_rings, filtered_clogp, filtered_frac_csp3 = apply_thresholds(hac_data, frac_aromatic_data, num_rings_data, clogp_data, frac_csp3_data)

    # Add scatter plot
    f.add_scatter(
        "mapc_nice_labels",
        {
            "x": x,
            "y": y,
            "c": [protein_class_data, taxonomy_data, organism_data, 
                  filtered_hac, filtered_frac_aromatic, filtered_num_rings, filtered_clogp, filtered_frac_csp3],
            "labels": labels,
        },
        shader="smoothCircle",
        point_scale=4.0,
        max_point_size=20,
        interactive=True,
        legend_labels=[protein_class_labels, taxonomy_labels, organism_labels],
        categorical=[True, True, True, False, False, False, False, False],
        colormap=['tab10', 'tab10', 'tab10', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis'],
        series_title=['Protein Class', 'Target Taxonomy', 'Target Organism',
                      'HAC', 'Fraction Aromatic Atoms', 'Number of Rings', 'clogP', 'Fraction Csp3'],
        has_legend=True,
    )

    # Add tree
    f.add_tree("mapc_nice_labels_tree", {"from": s, "to": t}, point_helper="mapc_nice_labels", color="#222222")
    
    # Plot
    f.plot('mapc_nice_labels', template='smiles')

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
        logger.info(f'Total fused fingerprints: {len(fused_fingerprints)}')
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
    most_active_compounds = df.groupby('Target_ID').apply(select_value).reset_index(drop=True)

    # Apply the mapping function to the column to reduce number of unique values and make it color codeable 
    most_active_compounds['mapped_protein_class'] = most_active_compounds['target_protein_class'].apply(map_protein_class)
    most_active_compounds['map_target_taxonomy'] = most_active_compounds['Target_Taxonomy'].apply(map_target_taxonomy)
    most_active_compounds['map_target_organism'] = most_active_compounds['Target_organism'].apply(map_target_organism)

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
    plot_faerun(x, y, s, t, most_active_compounds)

if __name__ == "__main__":
    start_time = timer()
    main()
    end_time = timer()
    logger.info('TMAP successfully generated.')
    print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")
    
