"""
Canonical shift normalization + model-loading helpers.

This module exists to kill a recurring drift hazard: per-AA shift
(de)normalization used to be reimplemented three times (training_utils.py,
inference.py, 07_evaluate.py) with subtly diverging out-of-range handling, and
``load_model`` was duplicated (inference.py / 07_evaluate.py) with diverging
n_shifts probing. Normalization<->inference parity is this project's single
most recurring bug source, so all of train/eval/inference now route through the
ONE implementation here.

Per-AA z-score convention (must match dataset.py / 05_build_training_cache.py):
    z = (ppm - mean_aa) / std_aa          # per-AA when available
    z = (ppm - mean_col) / std_col        # global fallback (AA lacks per-AA stats)
and the inverse for denormalization:
    ppm = z * std_aa  + mean_aa
    ppm = z * std_col + mean_col

Out-of-range / unknown AA policy (deterministic, documented):
    If the residue code does not map to a STANDARD_RESIDUES entry, OR that AA
    has no per-AA stats for a given column, we fall back to that column's
    GLOBAL mean/std. This matches the two historical *scalar* variants
    (training_utils.py, inference.py), which mapped out-of-range AA -> None ->
    global fallback. The historical *vectorized* variant (07_evaluate.py)
    instead clamped out-of-range codes into [0, n_aa-1] and used AA 0 (ALA)'s
    per-AA stats -- that was the odd one out and is NOT preserved (it used a
    wrong AA's stats for unknown residues). For in-range residues with stats,
    all three variants were numerically identical, so this change does not move
    any current numeric result for valid residues.

    If a column is absent from BOTH the per-AA block and the global stats dict,
    we leave it in z-space (mean 0, std 1) -- i.e. identity. In practice this
    never happens during training/eval because the caller's stats always
    contain every column it normalizes; it mirrors the safe behavior of the
    inference / 07_evaluate variants for a stats/shift_cols mismatch.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import (
    STANDARD_RESIDUES, RESIDUE_TO_IDX, N_RESIDUE_TYPES,
    N_SS_TYPES, N_MISMATCH_TYPES, ATOM_TO_IDX,
)
from model import ShiftPredictor

# Canonical set of the 6 backbone shift columns. Kept here as the single
# source of truth so train/eval/inference agree on what "backbone" means.
BACKBONE_SHIFTS = ('h_shift', 'ha_shift', 'c_shift', 'ca_shift', 'cb_shift', 'n_shift')

_STD_FLOOR = 1e-6  # guard against degenerate (std==0) per-AA stats


# ============================================================================
# Per-AA normalization
# ============================================================================

def _build_aa_tables(stats, shift_cols):
    """Build (n_aa + 1, n_shifts) mean/std tables applying the per-AA-with-global
    fallback convention documented at module top.

    Rows 0..n_aa-1 correspond to STANDARD_RESIDUES; the extra trailing row
    (index n_aa) is a pure GLOBAL-fallback row used for out-of-range / unknown
    residue codes -- it never carries any specific AA's per-AA stats, matching
    the two scalar variants' None -> global behavior exactly.

    Defaults are the identity transform (mean 0, std 1) so that a column absent
    from every stats source is left untouched in z-space.
    """
    per_aa_stats = stats.get('per_aa', {}) if isinstance(stats, dict) else {}
    n_aa = len(STANDARD_RESIDUES)
    n_shifts = len(shift_cols)
    means = np.zeros((n_aa + 1, n_shifts), dtype=np.float64)
    stds = np.ones((n_aa + 1, n_shifts), dtype=np.float64)
    for si, col in enumerate(shift_cols):
        has_global = isinstance(stats, dict) and col in stats
        g_mean = stats[col]['mean'] if has_global else 0.0
        g_std = max(stats[col]['std'], _STD_FLOOR) if has_global else 1.0
        # per-AA rows (with per-column global fallback)
        for ai, aa_name in enumerate(STANDARD_RESIDUES):
            aa_cell = per_aa_stats.get(aa_name, {}).get(col)
            if aa_cell:
                means[ai, si] = aa_cell['mean']
                stds[ai, si] = max(aa_cell['std'], _STD_FLOOR)
            else:
                means[ai, si] = g_mean
                stds[ai, si] = g_std
        # trailing global-fallback row for out-of-range AA
        means[n_aa, si] = g_mean
        stds[n_aa, si] = g_std
    return means, stds, n_aa


def _resolve_codes(residue_codes, n_aa):
    """Map raw residue codes -> safe row indices into the (n_aa + 1, n_shifts)
    tables. Out-of-range codes (incl. negative) point at the trailing
    global-fallback row (index n_aa) -- deterministic and AA-agnostic, never a
    wrong specific AA's per-AA stats.
    """
    codes = np.asarray(residue_codes).astype(np.int64).reshape(-1)
    in_range = (codes >= 0) & (codes < n_aa)
    safe = np.where(in_range, codes, n_aa)  # n_aa = trailing global-fallback row
    return safe


def denormalize(pred, residue_codes, stats, shift_cols):
    """Invert per-AA z-scoring -> ppm. THE canonical denorm for the project.

    Args:
        pred: (N, n_shifts) array/tensor of z-scored predictions (or targets).
        residue_codes: (N,) center-window AA index per row.
        stats: dict with optional 'per_aa' block and global per-column entries.
        shift_cols: list of shift-column names (column order of pred).
    Returns:
        np.ndarray (N, n_shifts) of denormalized ppm values.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred, dtype=np.float64)
    means, stds, n_aa = _build_aa_tables(stats, shift_cols)
    safe = _resolve_codes(residue_codes, n_aa)
    return pred * stds[safe] + means[safe]


def normalize(ppm, residue_codes, stats, shift_cols):
    """Forward per-AA z-scoring (ppm -> z). Inverse of ``denormalize``; provided
    for completeness/parity. Uses the identical tables and fallback policy.
    """
    if isinstance(ppm, torch.Tensor):
        ppm = ppm.detach().cpu().numpy()
    ppm = np.asarray(ppm, dtype=np.float64)
    means, stds, n_aa = _build_aa_tables(stats, shift_cols)
    safe = _resolve_codes(residue_codes, n_aa)
    return (ppm - means[safe]) / stds[safe]


# ============================================================================
# Model loading (single auto-detect implementation)
# ============================================================================

def load_model(checkpoint_path, device):
    """Load a trained ShiftPredictor from a checkpoint, auto-detecting the
    architecture from the saved weights. ONE implementation shared by
    inference.py and 07_evaluate.py (previously duplicated and diverged on
    n_shifts probing).

    Returns:
        (model, info) where info has a superset of the keys both callers use:
        'stats', 'shift_cols', 'atom_to_idx', 'n_atom_types', 'n_shifts',
        'epoch', 'checkpoint_path'. Returns (None, None) if the file is absent.
    """
    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Strip torch.compile's _orig_mod. prefix.
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    n_atom_types = clean_sd['distance_attention.atom_embed.weight'].shape[0] - 1
    n_dssp = clean_sd['dssp_proj.weight'].shape[1] if 'dssp_proj.weight' in clean_sd else 0

    stats = checkpoint.get('stats', {}) or {}
    shift_cols = checkpoint.get('shift_cols', []) or []

    # n_shifts probing (union of both historical strategies, most-specific
    # first): per-shift heads -> struct_head final layer -> len(shift_cols).
    n_shifts = sum(1 for k in clean_sd
                   if k.startswith('shift_heads.') and k.endswith('.0.weight'))
    if n_shifts == 0:
        struct_keys = sorted(k for k in clean_sd
                             if k.startswith('struct_head.') and k.endswith('.weight'))
        if struct_keys:
            n_shifts = clean_sd[struct_keys[-1]].shape[0]
        elif shift_cols:
            n_shifts = len(shift_cols)

    cnn_channels = []
    for i in range(0, 10, 2):
        key = f'cnn.{i}.conv1.weight'
        if key in clean_sd:
            cnn_channels.append(clean_sd[key].shape[0])

    spatial_hidden = (clean_sd['spatial_attention.fallback_embed'].shape[0]
                      if 'spatial_attention.fallback_embed' in clean_sd else 192)

    model = ShiftPredictor(
        n_atom_types=n_atom_types,
        n_residue_types=N_RESIDUE_TYPES,
        n_ss_types=N_SS_TYPES,
        n_mismatch_types=N_MISMATCH_TYPES,
        n_dssp=n_dssp,
        n_shifts=n_shifts,
        cnn_channels=cnn_channels,
        spatial_hidden=spatial_hidden,
    ).to(device)

    # Old checkpoints carry deprecated physics_encoder.* keys; drop them.
    filtered = {k: v for k, v in clean_sd.items()
                if not k.startswith('physics_encoder.')}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys on load (first: {missing[:3]})")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys in checkpoint (first: {unexpected[:3]})")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model ({n_params:,} params, n_shifts={n_shifts}, n_atom_types={n_atom_types})")

    return model, {
        'stats': stats,
        'shift_cols': shift_cols,
        'atom_to_idx': checkpoint.get('atom_to_idx', ATOM_TO_IDX),
        'n_atom_types': n_atom_types,
        'n_shifts': n_shifts,
        'epoch': checkpoint.get('epoch', '?'),
        'checkpoint_path': checkpoint_path,
    }
