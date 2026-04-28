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

2. Keeps the same memory-mapped retrieval architecture (shifts.npy, shift_masks.npy, etc.)
3. Keeps the same create() and load() classmethods with checkpoint/resume support

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
    DSSP_COLS, N_BOND_GEOM,
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


def build_atom_vocabulary(dist_col_info=None):
    """Return canonical atom vocabulary from config.

    Always returns the same vocabulary regardless of which distance columns
    are present, ensuring caches and models are interchangeable across datasets.
    """
    from config import ATOM_TYPES, ATOM_TO_IDX
    return list(ATOM_TYPES), dict(ATOM_TO_IDX)


def parse_shift_columns(columns):
    """Get chemical shift columns."""
    return sorted([c for c in columns if c.endswith('_shift')])


def get_dssp_columns(df_columns):
    """Get available DSSP columns."""
    return [c for c in DSSP_COLS if c in df_columns]


# ============================================================================
# Memory-Efficient Cached Dataset
# ============================================================================

class CachedRetrievalDataset(Dataset):
    """
    Memory-efficient dataset with disk caching.

    Storage strategy:
    - Structural features: stored in memory (compact)
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

        # Per-AA normalization (if available in stats)
        self._setup_per_aa_normalization(stats, shift_cols)

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

    def _setup_per_aa_normalization(self, stats, shift_cols):
        """Build per-(AA, shift) normalization tensors.

        When available, targets are re-normalized from global z-scores to
        per-AA z-scores on the fly: z_aa = (raw - aa_mean) / aa_std.
        This prevents large errors on shifts that span different chemical
        environments across amino acids (e.g., CD1: LEU=24 ppm, TYR=132 ppm).
        """
        from config import STANDARD_RESIDUES

        self.per_aa_means = None
        self.per_aa_stds = None
        self.use_per_aa_norm = False

        if stats is None or shift_cols is None:
            return
        per_aa = stats.get('per_aa')
        if not per_aa:
            return

        n_aa = len(STANDARD_RESIDUES)
        n_shifts = len(shift_cols)

        # Build (n_aa, n_shifts) tensors of per-AA mean/std
        # Fall back to global stats for missing AA/shift combos
        means = torch.zeros(n_aa, n_shifts)
        stds = torch.ones(n_aa, n_shifts)

        for aa_idx, aa_name in enumerate(STANDARD_RESIDUES):
            aa_stats = per_aa.get(aa_name, {})
            for si, col in enumerate(shift_cols):
                if col in aa_stats:
                    means[aa_idx, si] = aa_stats[col]['mean']
                    stds[aa_idx, si] = max(aa_stats[col]['std'], 0.1)
                elif col in stats:
                    means[aa_idx, si] = stats[col]['mean']
                    stds[aa_idx, si] = max(stats[col]['std'], 0.1)

        self.per_aa_means = means
        self.per_aa_stds = stds
        self.use_per_aa_norm = True

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

        # Compact structural feature vector (for retrieval neighbor encoder)
        query_struct_path = sd / 'query_struct.npy'
        if query_struct_path.exists():
            self.flat_query_struct = torch.from_numpy(np.load(query_struct_path))
            self.n_struct_features = self.flat_query_struct.shape[1]
        else:
            self.flat_query_struct = None
            self.n_struct_features = 0

        # Inter-residue bond geometry
        bond_geom_path = sd / 'bond_geom.npy'
        if bond_geom_path.exists():
            self.flat_bond_geom = torch.from_numpy(np.load(bond_geom_path))
        else:
            self.flat_bond_geom = None

    def _mmap_retrieval_data(self):
        """Memory-map retrieval data from disk.

        If the retrieval files are missing (no_retrieval cache), all retrieval
        attributes are set to None and __getitem__ synthesizes zero tensors.
        This avoids allocating tens of GB of zero-filled mmaps for caches built
        with --no_retrieval.
        """
        rd = self.cache_dir / 'retrieval'

        if not (rd / 'shifts.npy').exists():
            self.retrieved_shifts = None
            self.retrieved_shift_masks = None
            self.retrieved_residue_codes = None
            self.retrieved_distances = None
            self.retrieved_valid = None
            self.retrieved_neighbor_struct = None
            self.retrieval_compact = False
            return

        # Compact format (_compact.flag present): retrieval arrays are indexed by
        # sample idx (0..n_samples) instead of global_idx, so only rows for
        # shift-having samples are stored — ~4x smaller on disk, fits in RAM.
        self.retrieval_compact = (rd / '_compact.flag').exists()

        # For compact caches the full retrieval arrays are small enough to live
        # in RAM (~3.5 GB per fold × 5 folds = ~17 GB total). Loading fully
        # eliminates random HDD reads during shuffled training. For non-compact
        # caches the arrays are 4x bigger and we stay on mmap.
        load_mode = None if self.retrieval_compact else 'r'

        self.retrieved_shifts = np.load(rd / 'shifts.npy', mmap_mode=load_mode)
        self.retrieved_shift_masks = np.load(rd / 'shift_masks.npy', mmap_mode=load_mode)
        self.retrieved_residue_codes = np.load(rd / 'residue_codes.npy', mmap_mode=load_mode)
        self.retrieved_distances = np.load(rd / 'distances.npy', mmap_mode=load_mode)
        self.retrieved_valid = np.load(rd / 'valid.npy', mmap_mode=load_mode)

        # Neighbor structural features (may not exist in older caches)
        nbr_struct_path = rd / 'neighbor_struct.npy'
        if nbr_struct_path.exists():
            self.retrieved_neighbor_struct = np.load(nbr_struct_path, mmap_mode=load_mode)
        else:
            self.retrieved_neighbor_struct = None

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

        # Distance features for spatial neighbors
        neighbor_atom1_idx = torch.full((self.k_spatial, M), self.n_atom_types, dtype=torch.long)
        neighbor_atom2_idx = torch.full((self.k_spatial, M), self.n_atom_types, dtype=torch.long)
        neighbor_distances = torch.zeros(self.k_spatial, M, dtype=torch.float32)
        neighbor_dist_mask = torch.zeros(self.k_spatial, M, dtype=torch.bool)

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

                        # Look up distance features for this neighbor
                        n_valid = self.flat_dist_count[nb_global].item()
                        if n_valid > 0:
                            neighbor_atom1_idx[k, :n_valid] = torch.from_numpy(
                                self.flat_dist_atom1[nb_global, :n_valid].astype(np.int64)
                            )
                            neighbor_atom2_idx[k, :n_valid] = torch.from_numpy(
                                self.flat_dist_atom2[nb_global, :n_valid].astype(np.int64)
                            )
                            neighbor_distances[k, :n_valid] = torch.from_numpy(
                                self.flat_dist_values[nb_global, :n_valid].astype(np.float32)
                            )
                            neighbor_dist_mask[k, :n_valid] = True
                    else:
                        neighbor_valid[k] = False
                else:
                    neighbor_valid[k] = False

        # Bond geometry per window position
        if self.flat_bond_geom is not None:
            bond_geom = torch.zeros(W, N_BOND_GEOM, dtype=torch.float32)
            for w in range(W):
                if is_valid[w]:
                    res_idx_w = safe_idx[w].item()
                    bond_geom[w] = self.flat_bond_geom[res_idx_w]
        else:
            bond_geom = torch.zeros(W, N_BOND_GEOM, dtype=torch.float32)

        # Targets — re-normalize per-AA if stats available
        shift_target = self.flat_shifts[global_idx]  # globally z-normalized
        shift_mask = self.flat_shift_mask[global_idx]

        if self.use_per_aa_norm and self.per_aa_means is not None:
            aa_idx = self.flat_residue_idx[global_idx].item()
            if aa_idx < self.per_aa_means.shape[0]:
                # Convert: global_z -> raw_ppm -> per_aa_z
                raw = shift_target * self.retrieval_stds + self.retrieval_means
                shift_target = (raw - self.per_aa_means[aa_idx]) / self.per_aa_stds[aa_idx]
                shift_target = torch.where(shift_mask, shift_target, torch.zeros_like(shift_target))

        # Clean NaN
        neighbor_angles = torch.where(torch.isnan(neighbor_angles),
                                       torch.zeros_like(neighbor_angles), neighbor_angles)
        spatial_dist = torch.where(torch.isnan(spatial_dist),
                                    torch.zeros_like(spatial_dist), spatial_dist)
        dssp_features = torch.where(torch.isnan(dssp_features),
                                     torch.zeros_like(dssp_features), dssp_features)

        # Query residue code
        query_residue_code = self.flat_residue_idx[global_idx]

        # Retrieval data (from mmap, or zero-filled for no_retrieval caches).
        # When cache is compact, retrieval rows are indexed by sample idx
        # (== the __getitem__ arg); otherwise by global_idx (legacy).
        retr_idx = idx if self.retrieval_compact else global_idx

        if self.retrieved_shifts is None:
            retrieved_shifts = torch.zeros(K, self.n_shifts, dtype=torch.float32)
            retrieved_shift_masks = torch.zeros(K, self.n_shifts, dtype=torch.bool)
            retrieved_residue_codes = torch.zeros(K, dtype=torch.int64)
            retrieved_distances = torch.zeros(K, dtype=torch.float32)
            retrieved_valid = torch.zeros(K, dtype=torch.bool)
        else:
            retrieved_shifts = torch.from_numpy(
                self.retrieved_shifts[retr_idx].astype(np.float32)
            )
            retrieved_shift_masks = torch.from_numpy(
                self.retrieved_shift_masks[retr_idx].astype(bool)
            )
            retrieved_residue_codes = torch.from_numpy(
                self.retrieved_residue_codes[retr_idx].astype(np.int64)
            )
            retrieved_distances = torch.from_numpy(
                self.retrieved_distances[retr_idx].astype(np.float32)
            )
            retrieved_valid = torch.from_numpy(
                self.retrieved_valid[retr_idx].astype(bool)
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

        # Query structural features
        if self.flat_query_struct is not None:
            query_struct = self.flat_query_struct[global_idx].float()
            query_struct = torch.where(
                torch.isnan(query_struct),
                torch.zeros_like(query_struct),
                query_struct
            )
        else:
            query_struct = torch.zeros(max(self.n_struct_features, 1), dtype=torch.float32)

        # Neighbor structural features
        if self.retrieved_neighbor_struct is not None:
            neighbor_struct = torch.from_numpy(
                self.retrieved_neighbor_struct[retr_idx].astype(np.float32))
            neighbor_struct = torch.where(
                torch.isnan(neighbor_struct),
                torch.zeros_like(neighbor_struct),
                neighbor_struct
            )
        else:
            neighbor_struct = torch.zeros(K, max(self.n_struct_features, 1), dtype=torch.float32)

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
            'neighbor_atom1_idx': neighbor_atom1_idx,
            'neighbor_atom2_idx': neighbor_atom2_idx,
            'neighbor_distances': neighbor_distances,
            'neighbor_dist_mask': neighbor_dist_mask,
            'bond_geom': bond_geom,

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

            # Structural feature vectors (for retrieval neighbor encoder)
            'query_struct': query_struct,
            'neighbor_struct': neighbor_struct,
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

