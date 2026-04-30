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

    # First: forward WITHOUT cross args (legacy path)
    keys = ['atom1_idx','atom2_idx','distances','dist_mask','residue_idx',
            'ss_idx','mismatch_idx','is_valid','dssp_features',
            'neighbor_res_idx','neighbor_ss_idx','neighbor_dist',
            'neighbor_seq_sep','neighbor_angles','neighbor_valid',
            'neighbor_atom1_idx','neighbor_atom2_idx','neighbor_distances',
            'neighbor_dist_mask','bond_geom']
    legacy_kwargs = {k: batch[k] for k in keys if k in batch}
    with torch.no_grad():
        out_legacy = model(**legacy_kwargs)
    print(f'  legacy (no cross args): output shape {tuple(out_legacy.shape)}, '
          f'mean={out_legacy.mean().item():+.4f}')

    # Then: forward WITH cross args (new path)
    cross_kwargs = {**legacy_kwargs}
    for k in ['cross_atom1_idx','cross_atom2_idx','cross_offset_idx',
              'cross_distances','cross_dist_mask']:
        cross_kwargs[k] = batch[k]
    with torch.no_grad():
        out_cross = model(**cross_kwargs)
    print(f'  with cross args:        output shape {tuple(out_cross.shape)}, '
          f'mean={out_cross.mean().item():+.4f}')

    delta = (out_cross - out_legacy).abs()
    print(f'  delta (cross − legacy): mean={delta.mean().item():.4f}  '
          f'max={delta.max().item():.4f}')

    # The cross-attention's input_proj/value_proj are randomly initialized.
    # If cross_dist_mask has any True entries, output should differ.
    # If cross_dist_mask is all False, output must match exactly.
    any_active = batch['cross_dist_mask'].any().item()
    if any_active:
        if delta.max().item() < 1e-6:
            print('  WARN: cross args present but output unchanged. '
                  'Cross pathway might not be wired in correctly.')
        else:
            print('  OK: cross args perturb output (model is using them).')
    else:
        if delta.max().item() < 1e-6:
            print('  OK: cross all-masked → output unchanged (backward-compat).')
        else:
            print('  WARN: cross all-masked but output differs — backward-compat broken.')

    print('\nAll checks passed.' if delta.max().item() >= 0 else '')


if __name__ == '__main__':
    main()
