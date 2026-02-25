"""Central configuration for the clean chemical shift prediction pipeline."""

import numpy as np

# ============================================================================
# Paths
# ============================================================================
DATA_DIR = 'data'
PDB_DIR = 'data/pdbs'
DSSP_PATH = '/opt/homebrew/bin/dssp'

# ============================================================================
# Amino Acid Mappings
# ============================================================================
STANDARD_RESIDUES = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'UNK'
]
RESIDUE_TO_IDX = {r: i for i, r in enumerate(STANDARD_RESIDUES)}
N_RESIDUE_TYPES = len(STANDARD_RESIDUES)

AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

NUCLEOTIDES = {'DA', 'DC', 'DG', 'DT', 'DU', 'A', 'C', 'G', 'U', 'T',
               'ADE', 'CYT', 'GUA', 'THY', 'URA'}

# ============================================================================
# Secondary Structure
# ============================================================================
SS_TYPES = ['H', 'E', 'C', 'G', 'I', 'T', 'S', 'B', ' ', 'UNK']
SS_TO_IDX = {s: i for i, s in enumerate(SS_TYPES)}
N_SS_TYPES = len(SS_TYPES)

MISMATCH_TYPES = ['match', 'gap_in_cs', 'gap_in_structure', 'mismatch', 'protein_edge', 'UNK']
MISMATCH_TO_IDX = {m: i for i, m in enumerate(MISMATCH_TYPES)}
N_MISMATCH_TYPES = len(MISMATCH_TYPES)

# ============================================================================
# DSSP Columns
# ============================================================================
DSSP_COLS = [
    'rel_acc',
    'nh_o_1_relidx', 'nh_o_1_energy',
    'o_nh_1_relidx', 'o_nh_1_energy',
    'nh_o_2_relidx', 'nh_o_2_energy',
    'o_nh_2_relidx', 'o_nh_2_energy'
]

# ============================================================================
# Dense Distance Columns
# ============================================================================
BACKBONE_CORE_ATOMS = ['C', 'CA', 'N', 'O']
BACKBONE_H_ATOMS = ['H', 'CB', 'HA']

# ============================================================================
# Spatial Neighbor Parameters
# ============================================================================
K_SPATIAL_NEIGHBORS = 5
MIN_SEQ_SEPARATION = 4

# ============================================================================
# ESM-2 Parameters
# ============================================================================
ESM_MODEL_NAME = 'esm2_t36_3B_UR50D'
ESM_EMBED_DIM = 2560
ESM_REPR_LAYER = 36

# ============================================================================
# FAISS Parameters
# ============================================================================
FAISS_NPROBE = 64
K_RETRIEVED = 32

# ============================================================================
# Training Hyperparameters
# ============================================================================
LEARNING_RATE = 2e-4
BATCH_SIZE = 1024
EPOCHS = 200
CONTEXT_WINDOW = 5
HUBER_DELTA = 0.5
WEIGHT_DECAY = 0.05
OUTLIER_STD_THRESHOLD = 4.0

# Model architecture
DIST_ATTN_EMBED = 32
DIST_ATTN_HIDDEN = 256
CNN_CHANNELS = [256, 512, 768, 1024, 1280]
KERNEL_SIZE = 3
INPUT_DROPOUT = 0.10
LAYER_DROPOUTS = [0.40, 0.40, 0.40, 0.40, 0.40]
HEAD_DROPOUT = 0.45
SPATIAL_ATTN_HIDDEN = 192
RETRIEVAL_HIDDEN = 192
RETRIEVAL_HEADS = 4
RETRIEVAL_DROPOUT = 0.3
MAX_VALID_DISTANCES = 400

# ============================================================================
# RCSB / AlphaFold API
# ============================================================================
RCSB_SEARCH_URL = 'https://search.rcsb.org/rcsbsearch/v2/query'
ALPHAFOLD_DB_URL = 'https://alphafold.ebi.ac.uk/files'
ALPHAFOLD_DIR = 'data/alphafold'
BLAST_IDENTITY_CUTOFF = 0.30
