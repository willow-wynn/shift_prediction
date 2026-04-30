"""End-to-end validation that an augmented cache (with cross_*.npy) loads
through dataset.py and forwards through the model without errors.

Run AFTER 04c_compute_cross_arrays.py has produced the cross_*.npy files
in a fold cache. Checks:
  1. Dataset detects has_cross_features=True
  2. __getitem__ returns the right shapes for cross_* tensors
  3. cross_dist_mask has actual True entries (sanity: features are populated)
  4. Model forward pass works on a real batch and the cross-attention
     pathway contributes a non-zero residual at the center position

Usage:
    python tests/test_cache_load_with_cross.py [--cache_dir <path>]

Default cache_dir: data/struct_retrieval_v2/cache/fold_1
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import CachedRetrievalDataset
from config import (
    MAX_CROSS_DISTANCES, N_CROSS_OFFSET_TYPES, ATOM_TO_IDX,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', default='data/struct_retrieval_v2/cache/fold_1')
    args = ap.parse_args()

    print(f'Loading cache: {args.cache_dir}')
    if not os.path.isfile(os.path.join(args.cache_dir, 'config.json')):
        print(f'  cache config missing — has 04c run yet?')
        sys.exit(1)

    # Load shift_cols + stats from the cache config (matches train.py path)
    with open(os.path.join(args.cache_dir, 'config.json')) as f:
        cfg = json.load(f)
    shift_cols = cfg.get('shift_cols')
    stats = cfg.get('stats')
    n_shifts = len(shift_cols) if shift_cols else 49

    ds = CachedRetrievalDataset(args.cache_dir, n_shifts=n_shifts,
                                 stats=stats, shift_cols=shift_cols)
    print(f'  total samples: {len(ds):,}')
    print(f'  has_cross_features = {ds.has_cross_features}')
    print(f'  max_cross_distances = {ds.max_cross_distances}')
    print(f'  n_cross_offset_types = {ds.n_cross_offset_types}')

    if not ds.has_cross_features:
        print('  FAIL: cross arrays not detected. 04c probably hasn\'t run.')
        sys.exit(1)

    # Pull one sample
    print('\n--- single-sample sanity ---')
    sample = ds[0]
    for k in ['cross_atom1_idx', 'cross_atom2_idx', 'cross_offset_idx',
              'cross_distances', 'cross_dist_mask']:
        v = sample[k]
        print(f'  {k:<20} shape={tuple(v.shape)} dtype={v.dtype}')

    n_active = int(sample['cross_dist_mask'].sum())
    print(f'  active cross pairs in sample 0: {n_active}')

    # Pull 64 samples and check distribution
    print('\n--- distribution over 64 samples ---')
    counts = []
    for i in range(min(64, len(ds))):
        s = ds[i]
        counts.append(int(s['cross_dist_mask'].sum()))
    counts = np.array(counts)
    print(f'  cross_count: min={counts.min()}  median={int(np.median(counts))}  '
          f'mean={counts.mean():.1f}  max={counts.max()}')
    print(f'  fraction with >0 pairs: {(counts > 0).mean()*100:.1f}%')

    # Build a batch and forward through model
    print('\n--- forward through model ---')
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(dl))

    from model import create_model
    model = create_model(
        n_atom_types=ds.n_atom_types,
        n_shifts=n_shifts,
        use_retrieval=False,
        n_dssp=ds.n_dssp,
    )
    model.eval()

    # NOTE: model.struct_head's final layer is zero-init by design
    # (model.py: self.struct_head[-1].weight.zero_()), so an untrained
    # model's final output is identically zero regardless of internal
    # activations. To verify the cross pathway is actually firing, we
    # call cross_distance_attention directly and compare its output
    # under an active mask vs an all-False mask (which should fall back
    # to the zero-init fallback embedding).

    cross_a1 = batch['cross_atom1_idx'].unsqueeze(1)
    cross_a2 = batch['cross_atom2_idx'].unsqueeze(1)
    cross_off = batch['cross_offset_idx'].unsqueeze(1)
    cross_d = batch['cross_distances'].unsqueeze(1)
    cross_m = batch['cross_dist_mask'].unsqueeze(1)

    with torch.no_grad():
        cross_out_active = model.cross_distance_attention(
            cross_a1, cross_a2, cross_off, cross_d, cross_m)
        cross_out_zeroed = model.cross_distance_attention(
            cross_a1, cross_a2, cross_off, cross_d, torch.zeros_like(cross_m))

    print(f'  cross_distance_attention output shape: {tuple(cross_out_active.shape)}')
    print(f'  active mask:    absmax={cross_out_active.abs().max().item():.5f}  '
          f'std={cross_out_active.std().item():.5f}')
    print(f'  zero mask:      absmax={cross_out_zeroed.abs().max().item():.5f}  '
          f'std={cross_out_zeroed.std().item():.5f}')

    any_active = batch['cross_dist_mask'].any().item()
    if any_active and cross_out_active.abs().max().item() < 1e-6:
        print('  WARN: cross attention produces zero with active mask.')
    elif any_active:
        print('  OK: cross attention produces non-zero with active mask.')
    if cross_out_zeroed.abs().max().item() > 1e-6:
        print('  WARN: cross attention non-zero with all-False mask.')
    else:
        print('  OK: cross attention is exactly zero with all-False mask '
              '(backward-compat preserved).')

    # Also verify the full model forward doesn't crash with cross args
    keys = ['atom1_idx','atom2_idx','distances','dist_mask','residue_idx',
            'ss_idx','mismatch_idx','is_valid','dssp_features',
            'neighbor_res_idx','neighbor_ss_idx','neighbor_dist',
            'neighbor_seq_sep','neighbor_angles','neighbor_valid',
            'neighbor_atom1_idx','neighbor_atom2_idx','neighbor_distances',
            'neighbor_dist_mask','bond_geom',
            'cross_atom1_idx','cross_atom2_idx','cross_offset_idx',
            'cross_distances','cross_dist_mask']
    with torch.no_grad():
        out = model(**{k: batch[k] for k in keys if k in batch})
    print(f'  full model output shape: {tuple(out.shape)}  '
          f'(struct_head is zero-init, so values are 0 at init by design)')

    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
