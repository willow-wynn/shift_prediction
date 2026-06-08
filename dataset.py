#!/usr/bin/env python3
"""
Memory-efficient structure-only shift-prediction dataset backed by the
memory-mapped cache that 05_build_training_cache.py produces.

Imports constants from config (STANDARD_RESIDUES, SS_TYPES, MISMATCH_TYPES,
DSSP_COLS, ...). The class name CachedRetrievalDataset is retained for
back-compatibility with existing checkpoints/callers; there is no retrieval
pathway — it loads structural/cross/spatial/DSSP arrays only.

Usage:
    dataset = CachedRetrievalDataset.load('cache/fold_1', n_shifts,
                                          stats=stats, shift_cols=shift_cols)
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
    MAX_CROSS_DISTANCES, N_CROSS_OFFSET_TYPES,
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
        stats: dict = None,     # Normalization stats (mean/std per shift, + per-AA)
        shift_cols: list = None,  # Shift column names (for stats lookup)
        mmap_structural: bool = False,  # If False, load all structural arrays into RAM
    ):
        """Load a cached dataset from disk. Use CachedRetrievalDataset.load()."""
        self.cache_dir = Path(cache_dir)
        self.n_shifts = n_shifts
        self.mmap_structural = mmap_structural

        # Load config
        with open(self.cache_dir / 'config.json', 'r') as f:
            config = json.load(f)

        self.n_atom_types = config['n_atom_types']
        self.n_dssp = config['n_dssp']
        self.window_size = config['window_size']
        self.k_spatial = config['k_spatial']
        self.max_valid_distances = config['max_valid_distances']
        self.total_residues = config['total_residues']
        # Cross-residue distance features (Phase 1). Default to global config
        # constants so backward-compat caches without these keys still work.
        self.max_cross_distances = config.get('max_cross_distances', MAX_CROSS_DISTANCES)
        self.n_cross_offset_types = config.get('n_cross_offset_types', N_CROSS_OFFSET_TYPES)

        # Load stats from config if not provided (for backward compatibility)
        if stats is None and 'stats' in config:
            stats = config['stats']
        if shift_cols is None and 'shift_cols' in config:
            shift_cols = config['shift_cols']

        # Global per-shift normalization stats (mean/std), used to map cached
        # global-z targets back to ppm before per-AA renormalization.
        self._setup_shift_normalization(stats, shift_cols)

        # Per-AA normalization (if available in stats)
        self._setup_per_aa_normalization(stats, shift_cols)

        # Load samples list
        self.samples = np.load(self.cache_dir / 'samples.npy')

        # Load compact structural data (in memory)
        self._load_structural_data()


    def _setup_shift_normalization(self, stats, shift_cols):
        """Set up global per-shift mean/std (for global-z <-> ppm conversion)."""
        if stats is None or shift_cols is None:
            # No normalization - warn user
            print("WARNING: No stats provided - shifts will NOT be normalized!")
            self.shift_means = None
            self.shift_stds = None
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

        self.shift_means = torch.tensor(means, dtype=torch.float32)
        self.shift_stds = torch.tensor(stds, dtype=torch.float32)
        self.shift_stds = torch.where(
            self.shift_stds > 1e-6,
            self.shift_stds,
            torch.ones_like(self.shift_stds)
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

        # Sparse distances. Default: load fully into RAM (≤ 5 GB / fold) so
        # shuffled training doesn't thrash a spinning disk via mmap. Pass
        # mmap_structural=True only if the host is RAM-constrained.
        load_mode = 'r' if self.mmap_structural else None
        self.flat_dist_atom1 = np.load(sd / 'dist_atom1.npy', mmap_mode=load_mode)
        self.flat_dist_atom2 = np.load(sd / 'dist_atom2.npy', mmap_mode=load_mode)
        self.flat_dist_values = np.load(sd / 'dist_values.npy', mmap_mode=load_mode)

        # Cross-residue distances (Phase 1). Backward-compat: if the cross
        # arrays don't exist, set to None and __getitem__ will zero-fill.
        cross_a1_path = sd / 'cross_atom1.npy'
        if cross_a1_path.exists():
            self.flat_cross_atom1 = np.load(sd / 'cross_atom1.npy', mmap_mode=load_mode)
            self.flat_cross_atom2 = np.load(sd / 'cross_atom2.npy', mmap_mode=load_mode)
            self.flat_cross_offset = np.load(sd / 'cross_offset.npy', mmap_mode=load_mode)
            self.flat_cross_values = np.load(sd / 'cross_values.npy', mmap_mode=load_mode)
            self.flat_cross_count = torch.from_numpy(np.load(sd / 'cross_count.npy'))
            self.has_cross_features = True
        else:
            self.flat_cross_atom1 = None
            self.flat_cross_atom2 = None
            self.flat_cross_offset = None
            self.flat_cross_values = None
            self.flat_cross_count = None
            self.has_cross_features = False

        # global index -> BMRB ID (used for same-protein grouping in analysis)
        with open(sd / 'bmrb_mapping.json', 'r') as f:
            self.idx_to_bmrb = json.load(f)

        # Global index to residue ID mapping (load once, not per-item)
        with open(sd / 'global_to_resid.json', 'r') as f:
            self.global_to_resid = json.load(f)

        # Inter-residue bond geometry (4 features per residue)
        bond_geom_path = sd / 'bond_geom.npy'
        if bond_geom_path.exists():
            self.flat_bond_geom = torch.from_numpy(np.load(bond_geom_path))
        else:
            self.flat_bond_geom = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        global_idx, prot_idx = self.samples[idx]
        global_idx = int(global_idx)
        prot_idx = int(prot_idx)

        W = self.window_size
        M = self.max_valid_distances

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
                    # res_id_lookup maps (residue_id - min_res) -> global row.
                    # For proteins with duplicate residue_ids it resolves to the
                    # FIRST copy (05 builds it first-write-wins; see Finding #7),
                    # so 05 and this consumer agree on which copy a neighbor id
                    # points at.
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

        # Cross-residue distance features at the center position (Phase 1).
        # Only the center residue gets cross-pairs — window non-center positions
        # get only intra (their CNN aggregation already mixes spatial context).
        # Backward-compat: zero-filled tensors when cross arrays absent.
        M_CR = self.max_cross_distances
        cross_atom1_idx = torch.full((M_CR,), self.n_atom_types, dtype=torch.long)
        cross_atom2_idx = torch.full((M_CR,), self.n_atom_types, dtype=torch.long)
        cross_offset_idx = torch.full((M_CR,), self.n_cross_offset_types, dtype=torch.long)
        cross_distances = torch.zeros(M_CR, dtype=torch.float32)
        cross_dist_mask = torch.zeros(M_CR, dtype=torch.bool)
        if self.has_cross_features and self.flat_cross_count is not None:
            nc = int(self.flat_cross_count[global_idx].item())
            if nc > 0:
                nc = min(nc, M_CR)
                cross_atom1_idx[:nc] = torch.from_numpy(
                    self.flat_cross_atom1[global_idx, :nc].astype(np.int64))
                cross_atom2_idx[:nc] = torch.from_numpy(
                    self.flat_cross_atom2[global_idx, :nc].astype(np.int64))
                cross_offset_idx[:nc] = torch.from_numpy(
                    self.flat_cross_offset[global_idx, :nc].astype(np.int64))
                cross_distances[:nc] = torch.from_numpy(
                    self.flat_cross_values[global_idx, :nc].astype(np.float32))
                cross_dist_mask[:nc] = True

        # Targets — re-normalize per-AA if stats available
        shift_target = self.flat_shifts[global_idx]  # globally z-normalized
        shift_mask = self.flat_shift_mask[global_idx]

        if self.use_per_aa_norm and self.per_aa_means is not None:
            aa_idx = self.flat_residue_idx[global_idx].item()
            if aa_idx < self.per_aa_means.shape[0]:
                # Convert: global_z -> raw_ppm -> per_aa_z
                raw = shift_target * self.shift_stds + self.shift_means
                shift_target = (raw - self.per_aa_means[aa_idx]) / self.per_aa_stds[aa_idx]
                shift_target = torch.where(shift_mask, shift_target, torch.zeros_like(shift_target))

        # Clean NaN
        neighbor_angles = torch.where(torch.isnan(neighbor_angles),
                                       torch.zeros_like(neighbor_angles), neighbor_angles)
        spatial_dist = torch.where(torch.isnan(spatial_dist),
                                    torch.zeros_like(spatial_dist), spatial_dist)
        dssp_features = torch.where(torch.isnan(dssp_features),
                                     torch.zeros_like(dssp_features), dssp_features)

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

            # Cross-residue distance features (center residue only)
            'cross_atom1_idx': cross_atom1_idx,
            'cross_atom2_idx': cross_atom2_idx,
            'cross_offset_idx': cross_offset_idx,
            'cross_distances': cross_distances,
            'cross_dist_mask': cross_dist_mask,

            # Targets
            'shift_target': shift_target,
            'shift_mask': shift_mask,
        }

        return result

    def _get_residue_id(self, global_idx):
        """Get residue_id from global index (uses pre-loaded mapping)."""
        return self.global_to_resid.get(str(global_idx), -1)

    @classmethod
    def load(cls, cache_dir: str, n_shifts: int,
             stats: dict = None, shift_cols: list = None,
             mmap_structural: bool = False):
        """Load a cached dataset from disk.

        Args:
            cache_dir: Path to cached dataset
            n_shifts: Number of shift types
            stats: Normalization stats dict (REQUIRED for proper normalization)
            shift_cols: List of shift column names (REQUIRED for proper normalization)
            mmap_structural: If True, mmap structural arrays (low RAM, slow on
                spinning disk). Default False = load fully into RAM (~5 GB/fold).
        """
        return cls(cache_dir, n_shifts,
                   stats=stats, shift_cols=shift_cols,
                   mmap_structural=mmap_structural)

    @classmethod
    def exists(cls, cache_dir: str) -> bool:
        """Check if a cached dataset exists."""
        cache_path = Path(cache_dir)
        return (cache_path / 'config.json').exists()

