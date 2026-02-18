#!/usr/bin/env python3
"""
Imputation Dataset: extends CachedRetrievalDataset with observed shift context.

Adds per-residue observed shifts as a context window input, alongside the
existing structural and retrieval features. Each sample is a (residue, shift_type)
pair where the target shift is masked from the input.

Cache structure adds to the existing cache:
  cache/fold_{k}/imputation/
    shift_context_values.npy   # (N, n_shifts) z-normalized observed shifts
    shift_context_masks.npy    # (N, n_shifts) availability masks
    imputation_samples.npy     # (M, 3) = (global_idx, prot_idx, shift_type_idx)

The imputation dataset reuses all structural and retrieval data from the parent
CachedRetrievalDataset cache. It only adds the shift context arrays and a
different sample index (per-residue-per-shift instead of per-residue).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    N_RESIDUE_TYPES, N_SS_TYPES, N_MISMATCH_TYPES,
    DSSP_COLS,
)
from dataset import CachedRetrievalDataset


class ImputationDataset(Dataset):
    """Dataset for shift imputation: (residue, shift_type) pairs with context.

    Wraps a CachedRetrievalDataset and adds:
    - Per-residue observed shifts and masks (for context window)
    - Sample index: (global_idx, prot_idx, shift_type_idx)
    - __getitem__ masks the target shift from context before returning

    Curriculum masking:
    - extra_mask_rate: fraction of additional shifts to randomly mask (beyond target)
    - Set via set_extra_mask_rate() during training for curriculum schedule
    """

    def __init__(
        self,
        base_dataset: CachedRetrievalDataset,
        imputation_cache_dir: str,
        n_shifts: int,
        context_window: int = 5,
        extra_mask_rate: float = 0.0,
    ):
        self.base = base_dataset
        self.cache_dir = Path(imputation_cache_dir)
        self.n_shifts = n_shifts
        self.context_window = context_window
        self.window_size = 2 * context_window + 1
        self.extra_mask_rate = extra_mask_rate

        # Load imputation-specific data
        self.shift_values = torch.from_numpy(
            np.load(self.cache_dir / 'shift_context_values.npy')
        )  # (N, n_shifts) float32
        self.shift_masks = torch.from_numpy(
            np.load(self.cache_dir / 'shift_context_masks.npy')
        )  # (N, n_shifts) bool

        self.samples = np.load(self.cache_dir / 'imputation_samples.npy')  # (M, 3)

        # Pre-compute shift_type one-hots
        self._shift_type_templates = torch.zeros(n_shifts, n_shifts, dtype=torch.float32)
        for i in range(n_shifts):
            self._shift_type_templates[i, i] = 1.0

    def set_extra_mask_rate(self, rate: float):
        """Set the curriculum extra masking rate (0.0 to 1.0)."""
        self.extra_mask_rate = rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        global_idx, prot_idx, shift_type_idx = self.samples[idx]
        global_idx = int(global_idx)
        prot_idx = int(prot_idx)
        shift_type_idx = int(shift_type_idx)

        # Get base structural + retrieval features
        # We need to find the base sample index for this global_idx.
        # The base dataset's __getitem__ expects an index into base.samples,
        # but we need the raw global_idx. We'll reconstruct the batch manually
        # by calling into the base dataset's data arrays directly.
        base_batch = self._get_structural_and_retrieval(global_idx, prot_idx)

        # Build shift context window
        W = self.window_size
        center = self.context_window

        # Use base dataset's window indices
        window_idx = self.base.flat_window_idx[global_idx]  # (W,)
        is_valid = (window_idx >= 0)
        safe_idx = window_idx.clamp(min=0)

        # Context residue indices
        context_residue_idx = torch.where(
            is_valid,
            self.base.flat_residue_idx[safe_idx],
            torch.tensor(N_RESIDUE_TYPES, dtype=torch.long),
        )

        # Context observed shifts and masks
        context_observed_shifts = torch.zeros(W, self.n_shifts, dtype=torch.float32)
        context_shift_masks = torch.zeros(W, self.n_shifts, dtype=torch.float32)

        for w in range(W):
            if is_valid[w]:
                w_idx = safe_idx[w].item()
                context_observed_shifts[w] = self.shift_values[w_idx].float()
                context_shift_masks[w] = self.shift_masks[w_idx].float()

        # Mask the target shift at center position
        context_observed_shifts[center, shift_type_idx] = 0.0
        context_shift_masks[center, shift_type_idx] = 0.0

        # Curriculum: randomly mask additional shifts
        if self.extra_mask_rate > 0 and self.extra_mask_rate < 1.0:
            extra_mask = torch.rand(W, self.n_shifts) < self.extra_mask_rate
            # Don't double-mask the target (already masked)
            extra_mask[center, shift_type_idx] = False
            context_observed_shifts[extra_mask] = 0.0
            context_shift_masks[extra_mask] = 0.0

        # Center observed shifts (for retrieval conditioning)
        center_observed_shifts = context_observed_shifts[center].clone()
        center_shift_masks_vec = context_shift_masks[center].clone()

        # Target value
        target_value = self.shift_values[global_idx, shift_type_idx].float()

        # Shift type one-hot
        shift_type = self._shift_type_templates[shift_type_idx].clone()

        context_is_valid = is_valid.float()

        base_batch.update({
            # Shift context
            'context_residue_idx': context_residue_idx,
            'context_observed_shifts': context_observed_shifts,
            'context_shift_masks': context_shift_masks,
            'context_is_valid': context_is_valid,
            # Retrieval conditioning
            'center_observed_shifts': center_observed_shifts,
            'center_shift_masks': center_shift_masks_vec,
            # Shift type
            'shift_type': shift_type,
            # Target
            'target_value': target_value,
            'target_shift_idx': torch.tensor(shift_type_idx, dtype=torch.long),
        })

        return base_batch

    def _get_structural_and_retrieval(self, global_idx, prot_idx):
        """Extract structural and retrieval features for a single residue.

        This mirrors CachedRetrievalDataset.__getitem__ but takes raw
        global_idx/prot_idx instead of a sample index.
        """
        base = self.base
        W = base.window_size
        M = base.max_valid_distances
        K = base.k_retrieved

        window_idx = base.flat_window_idx[global_idx]
        is_valid = (window_idx >= 0)
        safe_idx = window_idx.clamp(min=0)

        # Distance features
        atom1_idx = torch.full((W, M), base.n_atom_types, dtype=torch.long)
        atom2_idx = torch.full((W, M), base.n_atom_types, dtype=torch.long)
        distances = torch.zeros(W, M, dtype=torch.float32)
        dist_mask = torch.zeros(W, M, dtype=torch.bool)

        for w in range(W):
            if is_valid[w]:
                res_idx = safe_idx[w].item()
                n_valid = base.flat_dist_count[res_idx].item()
                if n_valid > 0:
                    atom1_idx[w, :n_valid] = torch.from_numpy(
                        base.flat_dist_atom1[res_idx, :n_valid].astype(np.int64))
                    atom2_idx[w, :n_valid] = torch.from_numpy(
                        base.flat_dist_atom2[res_idx, :n_valid].astype(np.int64))
                    distances[w, :n_valid] = torch.from_numpy(
                        base.flat_dist_values[res_idx, :n_valid].astype(np.float32))
                    dist_mask[w, :n_valid] = True

        # Residue/SS/Mismatch
        residue_idx = torch.where(is_valid, base.flat_residue_idx[safe_idx],
                                   torch.tensor(N_RESIDUE_TYPES, dtype=torch.long))
        ss_idx = torch.where(is_valid, base.flat_ss_idx[safe_idx],
                             torch.tensor(N_SS_TYPES, dtype=torch.long))
        mismatch_idx = torch.where(is_valid, base.flat_mismatch_idx[safe_idx],
                                   torch.tensor(N_MISMATCH_TYPES, dtype=torch.long))

        dssp_features = base.flat_dssp[safe_idx] * is_valid.unsqueeze(-1).float()

        # Spatial neighbors
        spatial_res_ids = base.flat_spatial_ids[global_idx]
        spatial_dist = base.flat_spatial_dist[global_idx]
        spatial_seq_sep = base.flat_spatial_seq_sep[global_idx]

        neighbor_valid = (spatial_res_ids >= 0)

        lookup_offset = base.protein_lookup_offsets[prot_idx].item()
        min_res = base.protein_min_res[prot_idx].item()
        max_res = base.protein_max_res[prot_idx].item()
        span = max_res - min_res + 1

        k_spatial = base.k_spatial
        neighbor_res_idx = torch.full((k_spatial,), N_RESIDUE_TYPES, dtype=torch.long)
        neighbor_ss_idx = torch.full((k_spatial,), N_SS_TYPES, dtype=torch.long)
        neighbor_angles = torch.zeros(k_spatial, 4, dtype=torch.float32)

        for k in range(k_spatial):
            nb_res_id = spatial_res_ids[k].item()
            if nb_res_id >= 0:
                lookup_idx = int(nb_res_id) - min_res
                if 0 <= lookup_idx < span:
                    nb_global = base.flat_res_id_lookup[lookup_offset + lookup_idx].item()
                    if nb_global >= 0:
                        neighbor_res_idx[k] = base.flat_residue_idx[nb_global]
                        neighbor_ss_idx[k] = base.flat_ss_idx[nb_global]
                        neighbor_angles[k] = base.flat_angles[nb_global]
                    else:
                        neighbor_valid[k] = False
                else:
                    neighbor_valid[k] = False

        # Clean NaN
        neighbor_angles = torch.where(torch.isnan(neighbor_angles),
                                       torch.zeros_like(neighbor_angles), neighbor_angles)
        spatial_dist = torch.where(torch.isnan(spatial_dist),
                                    torch.zeros_like(spatial_dist), spatial_dist)
        dssp_features = torch.where(torch.isnan(dssp_features),
                                     torch.zeros_like(dssp_features), dssp_features)

        query_residue_code = base.flat_residue_idx[global_idx]

        # Retrieval data
        retrieved_shifts = torch.from_numpy(
            base.retrieved_shifts[global_idx].astype(np.float32))
        retrieved_shift_masks = torch.from_numpy(
            base.retrieved_shift_masks[global_idx].astype(bool))
        retrieved_residue_codes = torch.from_numpy(
            base.retrieved_residue_codes[global_idx].astype(np.int64))
        retrieved_distances = torch.from_numpy(
            base.retrieved_distances[global_idx].astype(np.float32))
        retrieved_valid = torch.from_numpy(
            base.retrieved_valid[global_idx].astype(bool))

        # Normalize retrieved shifts
        if base.retrieval_means is not None:
            retrieved_shifts = (retrieved_shifts - base.retrieval_means) / base.retrieval_stds
            retrieved_shifts = torch.clamp(retrieved_shifts, -10, 10)
            retrieved_shifts = torch.where(
                torch.isnan(retrieved_shifts), torch.zeros_like(retrieved_shifts),
                retrieved_shifts)

        # Physics features
        if base.flat_physics is not None:
            physics_features = base.flat_physics[global_idx].float()
            physics_features = torch.where(
                torch.isnan(physics_features), torch.zeros_like(physics_features),
                physics_features)
        else:
            physics_features = torch.zeros(base.n_physics, dtype=torch.float32)

        return {
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
            'query_residue_code': query_residue_code,
            'retrieved_shifts': retrieved_shifts,
            'retrieved_shift_masks': retrieved_shift_masks,
            'retrieved_residue_codes': retrieved_residue_codes,
            'retrieved_distances': retrieved_distances,
            'retrieved_valid': retrieved_valid,
            'physics_features': physics_features,
        }


# ============================================================================
# Cache Builder
# ============================================================================

def build_imputation_cache(
    base_dataset: CachedRetrievalDataset,
    imputation_cache_dir: str,
    n_shifts: int,
):
    """Build the imputation-specific cache arrays.

    Reads shift values/masks from the base dataset's structural cache
    and builds the per-(residue, shift_type) sample index.

    Args:
        base_dataset: Already-loaded CachedRetrievalDataset
        imputation_cache_dir: Where to save imputation arrays
        n_shifts: Number of shift types
    """
    cache_path = Path(imputation_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    N = base_dataset.total_residues

    # Shift values and masks are already in the base dataset
    shift_values = base_dataset.flat_shifts.numpy().astype(np.float32)
    shift_masks = base_dataset.flat_shift_mask.numpy().astype(np.bool_)

    print(f"  Building imputation cache: {N:,} residues, {n_shifts} shift types")

    # Save shift context arrays
    np.save(cache_path / 'shift_context_values.npy', shift_values)
    np.save(cache_path / 'shift_context_masks.npy', shift_masks)

    # Build sample index: (global_idx, prot_idx, shift_type_idx)
    # for every (residue, shift) pair where the shift is available
    samples_list = []

    # Use base dataset's samples to get (global_idx, prot_idx) pairs
    # but expand to per-shift samples
    base_samples = base_dataset.samples  # (M, 2) = (global_idx, prot_idx)

    for i in range(len(base_samples)):
        global_idx, prot_idx = base_samples[i]
        global_idx = int(global_idx)
        prot_idx = int(prot_idx)

        for s in range(n_shifts):
            if shift_masks[global_idx, s]:
                samples_list.append((global_idx, prot_idx, s))

    samples = np.array(samples_list, dtype=np.int32)
    np.save(cache_path / 'imputation_samples.npy', samples)

    # Save config
    config = {
        'n_residues': N,
        'n_shifts': n_shifts,
        'n_samples': len(samples_list),
        'n_base_samples': len(base_samples),
    }
    with open(cache_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Imputation samples: {len(samples_list):,}")
    print(f"  Cache saved to {imputation_cache_dir}")

    return samples


def load_imputation_dataset(
    base_dataset: CachedRetrievalDataset,
    imputation_cache_dir: str,
    n_shifts: int,
    context_window: int = 5,
) -> ImputationDataset:
    """Load an imputation dataset from cache.

    If the cache doesn't exist, builds it first.
    """
    cache_path = Path(imputation_cache_dir)

    if not (cache_path / 'imputation_samples.npy').exists():
        print(f"  Imputation cache not found, building...")
        build_imputation_cache(base_dataset, imputation_cache_dir, n_shifts)

    return ImputationDataset(
        base_dataset=base_dataset,
        imputation_cache_dir=imputation_cache_dir,
        n_shifts=n_shifts,
        context_window=context_window,
    )


# Need numpy for _get_structural_and_retrieval
import numpy as np  # noqa: E402 (already imported at top, but making dependency explicit)
