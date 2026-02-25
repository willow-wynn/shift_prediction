#!/usr/bin/env python3
"""
Memory-Efficient Dataset with Disk Caching for Retrieval-Augmented Shift Prediction
(Better Data Pipeline)

Adapted from homologies/dataset_cached.py with the following modifications:

1. Imports constants from config instead of redefining them:
   - STANDARD_RESIDUES, RESIDUE_TO_IDX, N_RESIDUE_TYPES
   - SS_TYPES, SS_TO_IDX, N_SS_TYPES
   - MISMATCH_TYPES, MISMATCH_TO_IDX, N_MISMATCH_TYPES
   - DSSP_COLS

2. Adds physics features to structural data:
   - _load_structural_data() also loads physics.npy if present
   - __getitem__() includes 'physics_features' in the output dict
   - create() saves physics feature array alongside other structural data

3. Backward compatibility: returns zeros if physics features don't exist

4. Keeps the same memory-mapped retrieval architecture (shifts.npy, shift_masks.npy, etc.)
5. Keeps the same create() and load() classmethods with checkpoint/resume support

Usage:
    # First run: builds and saves dataset
    dataset = CachedRetrievalDataset.create(
        df=train_df, ..., cache_dir='cache/fold1_train'
    )

    # Subsequent runs: loads from cache
    dataset = CachedRetrievalDataset.load('cache/fold1_train', ...)
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import (
    STANDARD_RESIDUES, RESIDUE_TO_IDX, N_RESIDUE_TYPES,
    SS_TYPES, SS_TO_IDX, N_SS_TYPES,
    MISMATCH_TYPES, MISMATCH_TO_IDX, N_MISMATCH_TYPES,
    DSSP_COLS,
)


# ============================================================================
# Utility Functions
# ============================================================================

def parse_distance_columns(columns):
    """Parse distance column names to extract atom pairs."""
    dist_cols = []
    pattern = re.compile(r'^dist_([A-Z0-9]+)_([A-Z0-9]+)$')

    for col in columns:
        match = pattern.match(col)
        if match:
            atom1, atom2 = match.groups()
            dist_cols.append((col, atom1, atom2))

    return dist_cols


def build_atom_vocabulary(dist_col_info):
    """Build vocabulary of unique atom types from distance columns."""
    atoms = set()
    for _, atom1, atom2 in dist_col_info:
        atoms.add(atom1)
        atoms.add(atom2)

    atom_list = sorted(atoms)
    atom_to_idx = {a: i for i, a in enumerate(atom_list)}
    return atom_list, atom_to_idx


def parse_shift_columns(columns):
    """Get chemical shift columns."""
    return sorted([c for c in columns if c.endswith('_shift')])


def get_dssp_columns(df_columns):
    """Get available DSSP columns."""
    return [c for c in DSSP_COLS if c in df_columns]


# ============================================================================
# Physics Feature Columns
# ============================================================================

# Default physics feature columns expected in the dataset.
# The exact list may vary; the dataset handles missing columns gracefully.
PHYSICS_COLS = [
    'ring_current_h', 'ring_current_ha',
    'hse_up', 'hse_down', 'hse_ratio',
    'hbond_dist_1', 'hbond_energy_1',
    'hbond_dist_2', 'hbond_energy_2',
    'order_parameter',
]


def get_physics_columns(df_columns):
    """Get available physics feature columns from the dataframe."""
    return [c for c in PHYSICS_COLS if c in df_columns]


# ============================================================================
# Memory-Efficient Cached Dataset
# ============================================================================

class CachedRetrievalDataset(Dataset):
    """
    Memory-efficient dataset with disk caching.

    Storage strategy:
    - Structural features: stored in memory (compact)
    - Physics features: stored in memory (compact, ~28 floats per residue)
    - Retrieved shifts: stored as memory-mapped numpy array on disk
    - Sample indices: stored in memory (small)
    """

    def __init__(
        self,
        cache_dir: str,
        n_shifts: int,
        k_retrieved: int = 32,
        embedding_lookup=None,  # Only needed for create(), not load()
        stats: dict = None,     # Normalization stats for retrieved shifts
        shift_cols: list = None,  # Shift column names (for stats lookup)
    ):
        """
        Load a cached dataset from disk.

        Use CachedRetrievalDataset.create() or CachedRetrievalDataset.load() instead.
        """
        self.cache_dir = Path(cache_dir)
        self.embedding_lookup = embedding_lookup  # May be None for loaded datasets
        self.n_shifts = n_shifts
        self.k_retrieved = k_retrieved

        # Load config
        with open(self.cache_dir / 'config.json', 'r') as f:
            config = json.load(f)

        self.n_atom_types = config['n_atom_types']
        self.n_dssp = config['n_dssp']
        self.n_physics = config.get('n_physics', 0)
        self.window_size = config['window_size']
        self.k_spatial = config['k_spatial']
        self.max_valid_distances = config['max_valid_distances']
        self.total_residues = config['total_residues']

        # Load stats from config if not provided (for backward compatibility)
        if stats is None and 'stats' in config:
            stats = config['stats']
        if shift_cols is None and 'shift_cols' in config:
            shift_cols = config['shift_cols']

        # Store normalization stats for retrieved shifts
        self._setup_retrieval_normalization(stats, shift_cols)

        # Load samples list
        self.samples = np.load(self.cache_dir / 'samples.npy')

        # Load compact structural data (in memory)
        self._load_structural_data()

        # Memory-map retrieval data (NOT in RAM)
        self._mmap_retrieval_data()

    def _setup_retrieval_normalization(self, stats, shift_cols):
        """Setup normalization parameters for retrieved shifts."""
        if stats is None or shift_cols is None:
            # No normalization - warn user
            print("WARNING: No stats provided - retrieved shifts will NOT be normalized!")
            self.retrieval_means = None
            self.retrieval_stds = None
            return

        # Build mean/std tensors for fast normalization
        means = []
        stds = []
        for col in shift_cols:
            if col in stats:
                means.append(stats[col]['mean'])
                stds.append(stats[col]['std'])
            else:
                means.append(0.0)
                stds.append(1.0)

        self.retrieval_means = torch.tensor(means, dtype=torch.float32)
        self.retrieval_stds = torch.tensor(stds, dtype=torch.float32)
        self.retrieval_stds = torch.where(
            self.retrieval_stds > 1e-6,
            self.retrieval_stds,
            torch.ones_like(self.retrieval_stds)
        )

    def _load_structural_data(self):
        """Load compact structural features into memory."""
        sd = self.cache_dir / 'structural'

        self.flat_residue_idx = torch.from_numpy(np.load(sd / 'residue_idx.npy'))
        self.flat_ss_idx = torch.from_numpy(np.load(sd / 'ss_idx.npy'))
        self.flat_mismatch_idx = torch.from_numpy(np.load(sd / 'mismatch_idx.npy'))
        self.flat_dist_count = torch.from_numpy(np.load(sd / 'dist_count.npy'))
        self.flat_dssp = torch.from_numpy(np.load(sd / 'dssp.npy'))
        self.flat_shifts = torch.from_numpy(np.load(sd / 'shifts.npy'))
        self.flat_shift_mask = torch.from_numpy(np.load(sd / 'shift_mask.npy'))
        self.flat_angles = torch.from_numpy(np.load(sd / 'angles.npy'))
        self.flat_window_idx = torch.from_numpy(np.load(sd / 'window_idx.npy'))

        # Spatial neighbors
        self.flat_spatial_ids = torch.from_numpy(np.load(sd / 'spatial_ids.npy'))
        self.flat_spatial_dist = torch.from_numpy(np.load(sd / 'spatial_dist.npy'))
        self.flat_spatial_seq_sep = torch.from_numpy(np.load(sd / 'spatial_seq_sep.npy'))

        # Protein lookup
        self.flat_res_id_lookup = torch.from_numpy(np.load(sd / 'res_id_lookup.npy'))
        self.protein_offsets = torch.from_numpy(np.load(sd / 'protein_offsets.npy'))
        self.protein_lookup_offsets = torch.from_numpy(np.load(sd / 'protein_lookup_offsets.npy'))
        self.protein_min_res = torch.from_numpy(np.load(sd / 'protein_min_res.npy'))
        self.protein_max_res = torch.from_numpy(np.load(sd / 'protein_max_res.npy'))

        # Sparse distances - load as mmap for memory efficiency
        self.flat_dist_atom1 = np.load(sd / 'dist_atom1.npy', mmap_mode='r')
        self.flat_dist_atom2 = np.load(sd / 'dist_atom2.npy', mmap_mode='r')
        self.flat_dist_values = np.load(sd / 'dist_values.npy', mmap_mode='r')

        # BMRB ID mapping for retrieval
        with open(sd / 'bmrb_mapping.json', 'r') as f:
            self.idx_to_bmrb = json.load(f)

        # Global index to residue ID mapping (load once, not per-item)
        with open(sd / 'global_to_resid.json', 'r') as f:
            self.global_to_resid = json.load(f)

        # Physics features (NEW)
        physics_path = sd / 'physics.npy'
        if physics_path.exists():
            self.flat_physics = torch.from_numpy(np.load(physics_path))
        else:
            # Backward compatibility: no physics features available
            self.flat_physics = None

    def _mmap_retrieval_data(self):
        """Memory-map retrieval data from disk."""
        rd = self.cache_dir / 'retrieval'

        # These are memory-mapped, NOT loaded into RAM
        self.retrieved_shifts = np.load(rd / 'shifts.npy', mmap_mode='r')
        self.retrieved_shift_masks = np.load(rd / 'shift_masks.npy', mmap_mode='r')
        self.retrieved_residue_codes = np.load(rd / 'residue_codes.npy', mmap_mode='r')
        self.retrieved_distances = np.load(rd / 'distances.npy', mmap_mode='r')
        self.retrieved_valid = np.load(rd / 'valid.npy', mmap_mode='r')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        global_idx, prot_idx = self.samples[idx]
        global_idx = int(global_idx)
        prot_idx = int(prot_idx)

        W = self.window_size
        M = self.max_valid_distances
        K = self.k_retrieved

        window_idx = self.flat_window_idx[global_idx]
        is_valid = (window_idx >= 0)
        safe_idx = window_idx.clamp(min=0)

        # Distance features (from mmap)
        atom1_idx = torch.full((W, M), self.n_atom_types, dtype=torch.long)
        atom2_idx = torch.full((W, M), self.n_atom_types, dtype=torch.long)
        distances = torch.zeros(W, M, dtype=torch.float32)
        dist_mask = torch.zeros(W, M, dtype=torch.bool)

        for w in range(W):
            if is_valid[w]:
                res_idx = safe_idx[w].item()
                n_valid = self.flat_dist_count[res_idx].item()
                if n_valid > 0:
                    atom1_idx[w, :n_valid] = torch.from_numpy(
                        self.flat_dist_atom1[res_idx, :n_valid].astype(np.int64)
                    )
                    atom2_idx[w, :n_valid] = torch.from_numpy(
                        self.flat_dist_atom2[res_idx, :n_valid].astype(np.int64)
                    )
                    distances[w, :n_valid] = torch.from_numpy(
                        self.flat_dist_values[res_idx, :n_valid].astype(np.float32)
                    )
                    dist_mask[w, :n_valid] = True

        # Residue/SS/Mismatch indices
        residue_idx = torch.where(is_valid, self.flat_residue_idx[safe_idx],
                                   torch.tensor(N_RESIDUE_TYPES, dtype=torch.long))
        ss_idx = torch.where(is_valid, self.flat_ss_idx[safe_idx],
                             torch.tensor(N_SS_TYPES, dtype=torch.long))
        mismatch_idx = torch.where(is_valid, self.flat_mismatch_idx[safe_idx],
                                   torch.tensor(N_MISMATCH_TYPES, dtype=torch.long))

        # DSSP features
        dssp_features = self.flat_dssp[safe_idx] * is_valid.unsqueeze(-1).float()

        # Spatial neighbors
        spatial_res_ids = self.flat_spatial_ids[global_idx]
        spatial_dist = self.flat_spatial_dist[global_idx]
        spatial_seq_sep = self.flat_spatial_seq_sep[global_idx]

        neighbor_valid = (spatial_res_ids >= 0)

        lookup_offset = self.protein_lookup_offsets[prot_idx].item()
        min_res = self.protein_min_res[prot_idx].item()
        max_res = self.protein_max_res[prot_idx].item()
        span = max_res - min_res + 1

        neighbor_res_idx = torch.full((self.k_spatial,), N_RESIDUE_TYPES, dtype=torch.long)
        neighbor_ss_idx = torch.full((self.k_spatial,), N_SS_TYPES, dtype=torch.long)
        neighbor_angles = torch.zeros(self.k_spatial, 4, dtype=torch.float32)

        for k in range(self.k_spatial):
            nb_res_id = spatial_res_ids[k].item()
            if nb_res_id >= 0:
                lookup_idx = int(nb_res_id) - min_res
                if 0 <= lookup_idx < span:
                    nb_global = self.flat_res_id_lookup[lookup_offset + lookup_idx].item()
                    if nb_global >= 0:
                        neighbor_res_idx[k] = self.flat_residue_idx[nb_global]
                        neighbor_ss_idx[k] = self.flat_ss_idx[nb_global]
                        neighbor_angles[k] = self.flat_angles[nb_global]
                    else:
                        neighbor_valid[k] = False
                else:
                    neighbor_valid[k] = False

        # Targets
        shift_target = self.flat_shifts[global_idx]
        shift_mask = self.flat_shift_mask[global_idx]

        # Clean NaN
        neighbor_angles = torch.where(torch.isnan(neighbor_angles),
                                       torch.zeros_like(neighbor_angles), neighbor_angles)
        spatial_dist = torch.where(torch.isnan(spatial_dist),
                                    torch.zeros_like(spatial_dist), spatial_dist)
        dssp_features = torch.where(torch.isnan(dssp_features),
                                     torch.zeros_like(dssp_features), dssp_features)

        # Query residue code
        query_residue_code = self.flat_residue_idx[global_idx]

        # Retrieval data (from mmap)
        retrieved_shifts = torch.from_numpy(
            self.retrieved_shifts[global_idx].astype(np.float32)
        )
        retrieved_shift_masks = torch.from_numpy(
            self.retrieved_shift_masks[global_idx].astype(bool)
        )
        retrieved_residue_codes = torch.from_numpy(
            self.retrieved_residue_codes[global_idx].astype(np.int64)
        )
        retrieved_distances = torch.from_numpy(
            self.retrieved_distances[global_idx].astype(np.float32)
        )
        retrieved_valid = torch.from_numpy(
            self.retrieved_valid[global_idx].astype(bool)
        )

        # Normalize retrieved shifts (CRITICAL: must match target normalization)
        if self.retrieval_means is not None:
            # Shape: (K, n_shifts) - normalize each shift column
            retrieved_shifts = (retrieved_shifts - self.retrieval_means) / self.retrieval_stds
            # Clamp to reasonable range and handle NaN
            retrieved_shifts = torch.clamp(retrieved_shifts, -10, 10)
            retrieved_shifts = torch.where(
                torch.isnan(retrieved_shifts),
                torch.zeros_like(retrieved_shifts),
                retrieved_shifts
            )

        # Physics features (NEW)
        if self.flat_physics is not None:
            physics_features = self.flat_physics[global_idx].float()
            physics_features = torch.where(
                torch.isnan(physics_features),
                torch.zeros_like(physics_features),
                physics_features
            )
        else:
            # Backward compatibility: return zeros
            physics_features = torch.zeros(self.n_physics, dtype=torch.float32)

        result = {
            # Structural features
            'atom1_idx': atom1_idx,
            'atom2_idx': atom2_idx,
            'distances': distances,
            'dist_mask': dist_mask,
            'residue_idx': residue_idx,
            'ss_idx': ss_idx,
            'mismatch_idx': mismatch_idx,
            'is_valid': is_valid.float(),
            'dssp_features': dssp_features,
            'neighbor_res_idx': neighbor_res_idx,
            'neighbor_ss_idx': neighbor_ss_idx,
            'neighbor_dist': spatial_dist,
            'neighbor_seq_sep': spatial_seq_sep,
            'neighbor_angles': neighbor_angles,
            'neighbor_valid': neighbor_valid,

            # Targets
            'shift_target': shift_target,
            'shift_mask': shift_mask,

            # Retrieval features
            'query_residue_code': query_residue_code,
            'retrieved_shifts': retrieved_shifts,
            'retrieved_shift_masks': retrieved_shift_masks,
            'retrieved_residue_codes': retrieved_residue_codes,
            'retrieved_distances': retrieved_distances,
            'retrieved_valid': retrieved_valid,

            # Physics features (NEW)
            'physics_features': physics_features,
        }

        return result

    def _get_residue_id(self, global_idx):
        """Get residue_id from global index (uses pre-loaded mapping)."""
        return self.global_to_resid.get(str(global_idx), -1)

    @classmethod
    def load(cls, cache_dir: str, n_shifts: int, k_retrieved: int = 32,
             stats: dict = None, shift_cols: list = None):
        """Load a cached dataset from disk.

        Args:
            cache_dir: Path to cached dataset
            n_shifts: Number of shift types
            k_retrieved: Number of retrieved neighbors
            stats: Normalization stats dict (REQUIRED for proper retrieval normalization)
            shift_cols: List of shift column names (REQUIRED for proper retrieval normalization)
        """
        return cls(cache_dir, n_shifts, k_retrieved, stats=stats, shift_cols=shift_cols)

    @classmethod
    def exists(cls, cache_dir: str) -> bool:
        """Check if a cached dataset exists."""
        cache_path = Path(cache_dir)
        return (cache_path / 'config.json').exists()

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        shift_cols: list,
        dist_col_info: list,
        dssp_cols: list,
        atom_to_idx: dict,
        stats: dict,
        embedding_lookup,
        retriever,
        cache_dir: str,
        context_window: int = 5,
        k_spatial: int = 5,
        k_retrieved: int = 32,
        max_valid_distances: int = 275,
        retrieval_batch_size: int = 5000,
        physics_cols: list = None,
    ):
        """
        Create a new cached dataset.

        This builds all the data and saves to disk, then returns a loaded dataset.
        Supports checkpoint/resume - if it crashes, run again and it picks up where it left off.

        MODIFIED from original:
        - Accepts physics_cols parameter for physics feature columns
        - Saves physics.npy alongside other structural data
        """
        cache_path = Path(cache_dir)

        # Auto-detect physics columns if not provided
        if physics_cols is None:
            physics_cols = get_physics_columns(df.columns)
        n_physics = len(physics_cols)

        # Check if we're resuming from a crash
        checkpoint_file = cache_path / 'retrieval' / '_checkpoint.txt'
        is_resuming = checkpoint_file.exists()

        if is_resuming:
            print(f"    *** FOUND CHECKPOINT - RESUMING BUILD ***")
        else:
            # Only remove existing cache if NOT resuming
            if cache_path.exists():
                shutil.rmtree(cache_path)

            cache_path.mkdir(parents=True)
            (cache_path / 'structural').mkdir()
            (cache_path / 'retrieval').mkdir()

        n_shifts = len(shift_cols)
        n_atom_types = len(atom_to_idx)
        n_dssp = len(dssp_cols)
        window_size = 2 * context_window + 1

        proteins = list(df.groupby('bmrb_id'))
        total_residues = sum(len(pdf) for _, pdf in proteins)
        n_proteins = len(proteins)

        M = max_valid_distances
        K = k_retrieved

        # Pre-compute atom indices
        col_to_atoms = []
        for col, atom1, atom2 in dist_col_info:
            a1_idx = atom_to_idx.get(atom1, n_atom_types)
            a2_idx = atom_to_idx.get(atom2, n_atom_types)
            col_to_atoms.append((a1_idx, a2_idx))
        col_to_atoms = np.array(col_to_atoms, dtype=np.int64)

        print("  Building cached dataset...")
        print(f"    Total residues: {total_residues:,}, proteins: {n_proteins}")
        print(f"    Physics features: {n_physics} columns")

        # ========== Allocate structural arrays ==========
        # Note: structural data rebuilds on resume (fast), only retrieval checkpoints
        flat_residue_idx = np.zeros(total_residues, dtype=np.int32)
        flat_ss_idx = np.zeros(total_residues, dtype=np.int32)
        flat_mismatch_idx = np.zeros(total_residues, dtype=np.int32)

        flat_dist_atom1 = np.full((total_residues, M), n_atom_types, dtype=np.int16)
        flat_dist_atom2 = np.full((total_residues, M), n_atom_types, dtype=np.int16)
        flat_dist_values = np.zeros((total_residues, M), dtype=np.float16)
        flat_dist_count = np.zeros(total_residues, dtype=np.int16)

        flat_dssp = np.zeros((total_residues, n_dssp), dtype=np.float16)
        flat_shifts = np.zeros((total_residues, n_shifts), dtype=np.float32)
        flat_shift_mask = np.zeros((total_residues, n_shifts), dtype=bool)
        flat_angles = np.zeros((total_residues, 4), dtype=np.float16)

        flat_spatial_ids = np.full((total_residues, k_spatial), -1, dtype=np.int32)
        flat_spatial_dist = np.zeros((total_residues, k_spatial), dtype=np.float16)
        flat_spatial_seq_sep = np.zeros((total_residues, k_spatial), dtype=np.int16)

        flat_window_idx = np.full((total_residues, window_size), -1, dtype=np.int32)

        # NEW: Physics features array
        flat_physics = np.zeros((total_residues, n_physics), dtype=np.float16) if n_physics > 0 else None

        # Protein tracking
        protein_offsets = []
        protein_lookup_offsets = []
        protein_min_res = []
        protein_max_res = []

        total_lookup_size = 0
        for _, pdf in proteins:
            rids = pdf['residue_id'].values
            total_lookup_size += int(rids.max()) - int(rids.min()) + 1

        flat_res_id_lookup = np.full(total_lookup_size, -1, dtype=np.int32)

        dist_cols = [info[0] for info in dist_col_info]

        current_offset = 0
        current_lookup_offset = 0
        samples_list = []

        # For retrieval
        idx_to_bmrb = {}
        global_to_resid = {}

        for prot_idx, (protein_id, pdf) in enumerate(tqdm(proteins, desc="    Processing structure")):
            pdf = pdf.sort_values('residue_id').reset_index(drop=True)
            n = len(pdf)
            start_idx = current_offset

            residue_ids = pdf['residue_id'].values.astype(np.int32)
            min_res, max_res = int(residue_ids.min()), int(residue_ids.max())
            span = max_res - min_res + 1

            protein_offsets.append(current_offset)
            protein_lookup_offsets.append(current_lookup_offset)
            protein_min_res.append(min_res)
            protein_max_res.append(max_res)

            for local_idx, rid in enumerate(residue_ids):
                global_idx = start_idx + local_idx
                flat_res_id_lookup[current_lookup_offset + (rid - min_res)] = global_idx
                idx_to_bmrb[str(global_idx)] = str(protein_id)
                global_to_resid[str(global_idx)] = int(rid)

            # Residue types
            for i, code in enumerate(pdf['residue_code'].fillna('UNK').values):
                flat_residue_idx[start_idx + i] = RESIDUE_TO_IDX.get(str(code).upper(), RESIDUE_TO_IDX['UNK'])

            # Secondary structure
            if 'secondary_structure' in pdf.columns:
                for i, ss in enumerate(pdf['secondary_structure'].fillna('C').values):
                    flat_ss_idx[start_idx + i] = SS_TO_IDX.get(str(ss), SS_TO_IDX['UNK'])

            # Mismatch type
            if 'mismatch_type' in pdf.columns:
                for i, mtype in enumerate(pdf['mismatch_type'].fillna('UNK').values):
                    flat_mismatch_idx[start_idx + i] = MISMATCH_TO_IDX.get(str(mtype), MISMATCH_TO_IDX['UNK'])

            # Sparse distances
            dist_matrix = pdf[dist_cols].values

            for i in range(n):
                global_idx = start_idx + i
                row = dist_matrix[i]

                valid_mask = ~np.isnan(row)
                valid_indices = np.where(valid_mask)[0]
                n_valid = min(len(valid_indices), M)

                flat_dist_count[global_idx] = n_valid

                if n_valid > 0:
                    valid_indices = valid_indices[:n_valid]
                    atom_pairs = col_to_atoms[valid_indices]
                    flat_dist_atom1[global_idx, :n_valid] = atom_pairs[:, 0]
                    flat_dist_atom2[global_idx, :n_valid] = atom_pairs[:, 1]

                    valid_values = row[valid_indices] / 10.0
                    valid_values = np.clip(valid_values, -5, 10)
                    flat_dist_values[global_idx, :n_valid] = valid_values

            # DSSP features
            for di, col in enumerate(dssp_cols):
                if col in pdf.columns:
                    vals = pdf[col].values
                    valid = ~np.isnan(vals)
                    if 'dssp' in stats and col in stats['dssp']:
                        mean = stats['dssp'][col]['mean']
                        std = stats['dssp'][col]['std']
                        if std > 1e-6:
                            normalized = np.where(valid, (vals - mean) / std, 0.0)
                        else:
                            normalized = np.where(valid, vals - mean, 0.0)
                    else:
                        normalized = np.where(valid, vals, 0.0)
                    normalized = np.clip(normalized, -10, 10)
                    flat_dssp[start_idx:start_idx + n, di] = normalized

            # Shifts
            for si, col in enumerate(shift_cols):
                if col in pdf.columns:
                    vals = pdf[col].values
                    valid = ~np.isnan(vals)
                    flat_shift_mask[start_idx:start_idx + n, si] = valid

                    if col in stats:
                        mean, std = stats[col]['mean'], stats[col]['std']
                        if std > 1e-6:
                            normalized = np.where(valid, (vals - mean) / std, 0.0)
                        else:
                            normalized = np.where(valid, vals - mean, 0.0)
                        normalized = np.clip(normalized, -10, 10)
                        flat_shifts[start_idx:start_idx + n, si] = normalized

            # Angles
            for i, angle_col in enumerate(['phi', 'psi']):
                if angle_col in pdf.columns:
                    vals = pdf[angle_col].values
                    valid = ~np.isnan(vals)
                    rad = np.where(valid, np.radians(vals), 0.0)
                    flat_angles[start_idx:start_idx + n, i*2] = np.sin(rad)
                    flat_angles[start_idx:start_idx + n, i*2 + 1] = np.cos(rad)

            # Spatial neighbors
            for k in range(k_spatial):
                id_col = f'spatial_neighbor_{k}_id'
                dist_col_name = f'spatial_neighbor_{k}_dist'
                sep_col = f'spatial_neighbor_{k}_seq_sep'

                if id_col in pdf.columns:
                    ids = pdf[id_col].values
                    valid = ~np.isnan(ids) & (ids >= 0)
                    flat_spatial_ids[start_idx:start_idx + n, k] = np.where(valid, ids, -1)

                if dist_col_name in pdf.columns:
                    dists = pdf[dist_col_name].values
                    valid = ~np.isnan(dists)
                    flat_spatial_dist[start_idx:start_idx + n, k] = np.where(valid, dists, 0.0)

                if sep_col in pdf.columns:
                    seps = pdf[sep_col].values
                    valid = ~np.isnan(seps)
                    flat_spatial_seq_sep[start_idx:start_idx + n, k] = np.where(valid, seps, 0)

            # Window indices
            for local_idx, rid in enumerate(residue_ids):
                global_idx = start_idx + local_idx
                for w, offset in enumerate(range(-context_window, context_window + 1)):
                    neighbor_rid = rid + offset
                    lookup_idx = neighbor_rid - min_res
                    if 0 <= lookup_idx < span:
                        neighbor_global = flat_res_id_lookup[current_lookup_offset + lookup_idx]
                        if neighbor_global >= 0:
                            flat_window_idx[global_idx, w] = neighbor_global

            # Physics features (NEW)
            if flat_physics is not None and n_physics > 0:
                for pi, col in enumerate(physics_cols):
                    if col in pdf.columns:
                        vals = pdf[col].values
                        valid = ~np.isnan(vals)
                        # Store raw values (normalization can be done at model level or here)
                        flat_physics[start_idx:start_idx + n, pi] = np.where(valid, vals, 0.0)

            # Build samples
            for local_idx in range(n):
                global_idx = start_idx + local_idx
                if flat_shift_mask[global_idx].any():
                    samples_list.append((global_idx, prot_idx))

            current_offset += n
            current_lookup_offset += span

        # ========== Save structural data ==========
        print("    Saving structural data...")
        sd = cache_path / 'structural'

        np.save(sd / 'residue_idx.npy', flat_residue_idx)
        np.save(sd / 'ss_idx.npy', flat_ss_idx)
        np.save(sd / 'mismatch_idx.npy', flat_mismatch_idx)
        np.save(sd / 'dist_atom1.npy', flat_dist_atom1)
        np.save(sd / 'dist_atom2.npy', flat_dist_atom2)
        np.save(sd / 'dist_values.npy', flat_dist_values)
        np.save(sd / 'dist_count.npy', flat_dist_count)
        np.save(sd / 'dssp.npy', flat_dssp)
        np.save(sd / 'shifts.npy', flat_shifts)
        np.save(sd / 'shift_mask.npy', flat_shift_mask)
        np.save(sd / 'angles.npy', flat_angles)
        np.save(sd / 'window_idx.npy', flat_window_idx)
        np.save(sd / 'spatial_ids.npy', flat_spatial_ids)
        np.save(sd / 'spatial_dist.npy', flat_spatial_dist)
        np.save(sd / 'spatial_seq_sep.npy', flat_spatial_seq_sep)
        np.save(sd / 'res_id_lookup.npy', flat_res_id_lookup)
        np.save(sd / 'protein_offsets.npy', np.array(protein_offsets, dtype=np.int32))
        np.save(sd / 'protein_lookup_offsets.npy', np.array(protein_lookup_offsets, dtype=np.int32))
        np.save(sd / 'protein_min_res.npy', np.array(protein_min_res, dtype=np.int32))
        np.save(sd / 'protein_max_res.npy', np.array(protein_max_res, dtype=np.int32))

        # Save physics features (NEW)
        if flat_physics is not None:
            np.save(sd / 'physics.npy', flat_physics)

        with open(sd / 'bmrb_mapping.json', 'w') as f:
            json.dump(idx_to_bmrb, f)
        with open(sd / 'global_to_resid.json', 'w') as f:
            json.dump(global_to_resid, f)

        samples = np.array(samples_list, dtype=np.int32)
        np.save(cache_path / 'samples.npy', samples)

        # Free structural memory
        del flat_dist_atom1, flat_dist_atom2, flat_dist_values
        del flat_residue_idx, flat_ss_idx, flat_mismatch_idx
        del flat_dssp, flat_shifts, flat_shift_mask, flat_angles
        del flat_spatial_ids, flat_spatial_dist, flat_spatial_seq_sep
        del flat_window_idx, flat_res_id_lookup
        if flat_physics is not None:
            del flat_physics

        import gc
        gc.collect()

        # ========== Build retrieval data (with checkpoint/resume) ==========
        print("    Building retrieval data (in batches)...")

        rd = cache_path / 'retrieval'
        checkpoint_file = rd / '_checkpoint.txt'

        # Check for existing checkpoint to resume from
        resume_from = 0
        if checkpoint_file.exists():
            try:
                resume_from = int(checkpoint_file.read_text().strip())
                print(f"    RESUMING from batch starting at index {resume_from}")
            except:
                resume_from = 0

        # Use 'r+' mode if resuming (preserves existing data), 'w+' if starting fresh
        mmap_mode = 'r+' if resume_from > 0 else 'w+'

        # Pre-allocate retrieval arrays as memory-mapped files
        retrieved_shifts = np.lib.format.open_memmap(
            rd / 'shifts.npy', mode=mmap_mode, dtype=np.float16,
            shape=(total_residues, K, n_shifts)
        )
        retrieved_shift_masks = np.lib.format.open_memmap(
            rd / 'shift_masks.npy', mode=mmap_mode, dtype=bool,
            shape=(total_residues, K, n_shifts)
        )
        retrieved_residue_codes = np.lib.format.open_memmap(
            rd / 'residue_codes.npy', mode=mmap_mode, dtype=np.int16,
            shape=(total_residues, K)
        )
        retrieved_distances = np.lib.format.open_memmap(
            rd / 'distances.npy', mode=mmap_mode, dtype=np.float16,
            shape=(total_residues, K)
        )
        retrieved_valid = np.lib.format.open_memmap(
            rd / 'valid.npy', mode=mmap_mode, dtype=bool,
            shape=(total_residues, K)
        )

        # Process in batches to limit memory usage
        all_global_indices = list(range(total_residues))

        # Calculate total batches for progress
        total_batches = (total_residues + retrieval_batch_size - 1) // retrieval_batch_size
        start_batch = resume_from // retrieval_batch_size

        for batch_start in tqdm(range(resume_from, total_residues, retrieval_batch_size),
                                 desc="    Retrieval batches",
                                 initial=start_batch,
                                 total=total_batches):
            batch_end = min(batch_start + retrieval_batch_size, total_residues)
            batch_indices = all_global_indices[batch_start:batch_end]

            # Collect all query (bmrb_id, residue_id) pairs for this batch
            batch_bmrb_ids = []
            batch_res_ids = []

            for global_idx in batch_indices:
                bmrb_id = idx_to_bmrb[str(global_idx)]
                rid = global_to_resid[str(global_idx)]
                batch_bmrb_ids.append(bmrb_id)
                batch_res_ids.append(rid)

            # BATCH LOOKUP: Get all query embeddings in one call (grouped by protein internally)
            batch_embeddings, batch_valid_mask = embedding_lookup.get_batch(batch_bmrb_ids, batch_res_ids)

            # Retrieve for valid embeddings only
            valid_indices = np.where(batch_valid_mask)[0]

            if len(valid_indices) > 0:
                valid_embeddings = batch_embeddings[valid_indices]
                valid_bmrb_ids = [batch_bmrb_ids[i] for i in valid_indices]

                results = retriever.retrieve(
                    query_embeddings=valid_embeddings,
                    query_bmrb_ids=valid_bmrb_ids,
                    k=K,
                )

                # Store retrieval results (vectorized writes)
                global_indices = batch_start + valid_indices
                retrieved_shifts[global_indices] = results['shifts']
                retrieved_shift_masks[global_indices] = results['shift_masks']
                retrieved_residue_codes[global_indices] = results['residue_codes']
                retrieved_distances[global_indices] = results['distances']
                retrieved_valid[global_indices] = results['indices'] >= 0

            # Flush and checkpoint EVERY batch for crash safety
            retrieved_shifts.flush()
            retrieved_shift_masks.flush()
            retrieved_residue_codes.flush()
            retrieved_distances.flush()
            retrieved_valid.flush()

            # Save checkpoint so we can resume if crash
            checkpoint_file.write_text(str(batch_end))

            # Garbage collect every 10 batches
            if (batch_end % (retrieval_batch_size * 10)) == 0:
                gc.collect()

        # Final flush
        retrieved_shifts.flush()
        retrieved_shift_masks.flush()
        retrieved_residue_codes.flush()
        retrieved_distances.flush()
        retrieved_valid.flush()

        # Remove checkpoint file - we completed successfully!
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("    Retrieval data complete - checkpoint removed")

        del retrieved_shifts, retrieved_shift_masks, retrieved_residue_codes
        del retrieved_distances, retrieved_valid
        gc.collect()

        # ========== Save config ==========
        # Convert stats to JSON-serializable format
        stats_for_json = {}
        for col in shift_cols:
            if col in stats:
                stats_for_json[col] = {
                    'mean': float(stats[col]['mean']),
                    'std': float(stats[col]['std']),
                }

        config = {
            'n_atom_types': n_atom_types,
            'n_dssp': n_dssp,
            'n_physics': n_physics,
            'n_shifts': n_shifts,
            'window_size': window_size,
            'k_spatial': k_spatial,
            'k_retrieved': k_retrieved,
            'max_valid_distances': max_valid_distances,
            'total_residues': total_residues,
            'n_proteins': n_proteins,
            'n_samples': len(samples_list),
            'shift_cols': shift_cols,        # Save for retrieval normalization
            'stats': stats_for_json,         # Save for retrieval normalization
            'physics_cols': physics_cols,     # Save physics col names
        }

        with open(cache_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"    Built {len(samples_list):,} samples")
        print(f"    Cache saved to {cache_dir}")

        # Return loaded dataset
        return cls.load(cache_dir, n_shifts, k_retrieved)
