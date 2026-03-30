"""Central configuration for the clean chemical shift prediction pipeline."""

import os
import shutil

import numpy as np

# ============================================================================
# Paths
# ============================================================================
DATA_DIR = 'data'
PDB_DIR = 'data/pdbs'
DSSP_PATH = os.environ.get('DSSP_PATH') or shutil.which('mkdssp') or shutil.which('dssp') or 'mkdssp'

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
# Canonical Atom Vocabulary (union of all datasets: hybrid, alphafold, experimental)
# Every cache and model uses this vocabulary so they are interchangeable.
# ============================================================================
ATOM_TYPES = [
    'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
    'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'DD1',
    'H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3',
    'HB', 'HB1', 'HB2', 'HB3', 'HC',
    'HD1', 'HD11', 'HD12', 'HD13', 'HD2', 'HD21', 'HD22', 'HD23', 'HD3',
    'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HE3',
    'HG', 'HG1', 'HG11', 'HG12', 'HG13', 'HG2', 'HG21', 'HG22', 'HG23', 'HG3',
    'HH', 'HH11', 'HH12', 'HH2', 'HH21', 'HH22',
    'HN1', 'HN2', 'HXT', 'HZ', 'HZ1', 'HZ2', 'HZ3',
    'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ',
    'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT',
    'SD', 'SG',
]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
N_ATOM_TYPES = len(ATOM_TYPES)

# Dataset paths: --data flag maps to these directories
DATASET_DIRS = {
    'hybrid': 'data',
    'alphafold': 'data_alphafold',
    'experimental': 'data_experimental',
}

# ============================================================================
# Inter-Residue Bond Geometry Columns
# ============================================================================
BOND_GEOM_COLS = [
    'bond_ca_prev',       # ||CA(i) - CA(i-1)||
    'bond_ca_next',       # ||CA(i) - CA(i+1)||
    'bond_peptide_fwd',   # ||C(i) - N(i+1)||  (forward peptide bond)
    'bond_peptide_bkwd',  # ||C(i-1) - N(i)||  (backward peptide bond)
]
N_BOND_GEOM = len(BOND_GEOM_COLS)

# ============================================================================
# Dense Distance Columns
# ============================================================================
BACKBONE_CORE_ATOMS = ['C', 'CA', 'N', 'O']
BACKBONE_H_ATOMS = ['H', 'CB', 'HA']

# ============================================================================
# Compact Structural Feature Vector (49-dim per residue)
# Used by retrieval neighbor encoder for structural comparison
# ============================================================================
STRUCT_DIST_COLS = [
    'dist_C_CA', 'dist_C_CB', 'dist_C_H', 'dist_C_HA', 'dist_C_N', 'dist_C_O',
    'dist_CA_CB', 'dist_CA_H', 'dist_CA_HA', 'dist_CA_N', 'dist_CA_O',
    'dist_CB_H', 'dist_CB_HA', 'dist_CB_N', 'dist_CB_O',
    'dist_H_HA', 'dist_H_N', 'dist_H_O', 'dist_HA_N', 'dist_HA_O', 'dist_N_O',
]
STRUCT_SC_COLS = [
    'sc_n_resolved', 'sc_mean_dist_ca', 'sc_compactness', 'sc_max_extent',
    'sc_centroid_dist',
]
# Total: 21 + 5 + 4(angles) + 9(DSSP) + 10(SS one-hot) = 49
N_STRUCT_FEATURES = len(STRUCT_DIST_COLS) + len(STRUCT_SC_COLS) + 4 + len(DSSP_COLS) + len(SS_TYPES)

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
BACKBONE_LOSS_WEIGHT = 2.0

# Model architecture
DIST_ATTN_EMBED = 32
DIST_ATTN_HIDDEN = 256
CNN_CHANNELS = [256, 512, 768, 1024, 1280]
KERNEL_SIZE = 3
INPUT_DROPOUT = 0.10
LAYER_DROPOUTS = [0.40, 0.40, 0.40, 0.40, 0.40]
HEAD_DROPOUT = 0.45
SPATIAL_ATTN_HIDDEN = 192
RETRIEVAL_HIDDEN = 320
RETRIEVAL_HEADS = 8
RETRIEVAL_DROPOUT = 0.3
MAX_VALID_DISTANCES = 400

# ============================================================================
# RCSB / AlphaFold API
# ============================================================================
RCSB_SEARCH_URL = 'https://search.rcsb.org/rcsbsearch/v2/query'
ALPHAFOLD_DB_URL = 'https://alphafold.ebi.ac.uk/files'
ALPHAFOLD_DIR = 'data/alphafold'
MIN_SEQUENCE_IDENTITY = 0.80
