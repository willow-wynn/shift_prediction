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

# Non-standard residue mappings (map to standard)
NONSTANDARD_MAP = {
    'SEC': 'CYS', 'MSE': 'MET', 'PYL': 'LYS',
    'ASX': 'ASN', 'GLX': 'GLN', 'XAA': 'UNK',
    'UNK': 'UNK', 'NLW': 'UNK', 'NLE': 'UNK',
}

NUCLEOTIDES = {'DA', 'DC', 'DG', 'DT', 'DU', 'A', 'C', 'G', 'U', 'T',
               'ADE', 'CYT', 'GUA', 'THY', 'URA'}

# ============================================================================
# Chemical Shift Outlier Ranges (ppm)
# ============================================================================
SHIFT_RANGES = {
    'ca_shift': (40.0, 70.0),
    'cb_shift': (15.0, 75.0),
    'c_shift': (165.0, 185.0),
    'n_shift': (100.0, 140.0),
    'h_shift': (5.0, 12.0),
    'ha_shift': (2.0, 6.0),
}

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
# Dense Distance Columns (< 50% NaN in main dataset)
# ============================================================================
# 6 backbone core (approx 0% NaN)
BACKBONE_CORE_ATOMS = ['C', 'CA', 'N', 'O']
# Additional backbone atoms with lower coverage
BACKBONE_H_ATOMS = ['H', 'CB', 'HA']

# All backbone + near-backbone atoms for distance computation
DISTANCE_ATOMS = ['C', 'CA', 'CB', 'H', 'HA', 'N', 'O']

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
MAX_VALID_DISTANCES = 275

# ============================================================================
# BLOSUM62 Matrix
# ============================================================================
# Standard amino acids in order: ARNDCQEGHILKMFPSTWYV
BLOSUM62_ORDER = 'ARNDCQEGHILKMFPSTWYV'
BLOSUM62 = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],  # A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],  # R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],  # N
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],  # D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],  # C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],  # Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],  # E
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],  # G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],  # H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],  # I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],  # L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],  # K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],  # M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],  # F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],  # P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],  # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],  # T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],  # W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],  # Y
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],  # V
], dtype=np.float32)
