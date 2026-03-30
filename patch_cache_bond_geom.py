#!/usr/bin/env python3
"""
Patch existing training caches to add bond_geom.npy.

Reads bond geometry columns from the CSV and creates bond_geom.npy in each
fold's structural/ directory, matching the existing residue ordering.
Much faster than rebuilding the entire cache (skips FAISS retrieval).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BOND_GEOM_COLS, N_BOND_GEOM


def patch_fold(fold_dir, df_by_bmrb, stats_dssp=None):
    """Add bond_geom.npy to an existing fold cache."""
    fold_path = Path(fold_dir)
    sd = fold_path / 'structural'

    if not sd.exists():
        print(f"  SKIP: {fold_dir} (no structural/ dir)")
        return

    # Load config to get total_residues
    with open(fold_path / 'config.json') as f:
        config = json.load(f)
    total_residues = config['total_residues']

    # Load bmrb mapping and residue order
    with open(sd / 'bmrb_mapping.json') as f:
        idx_to_bmrb = json.load(f)
    with open(sd / 'global_to_resid.json') as f:
        global_to_resid = json.load(f)

    # Allocate
    flat_bond_geom = np.zeros((total_residues, N_BOND_GEOM), dtype=np.float32)

    # Build lookup: (bmrb_id, residue_id) -> bond geometry values
    # Group global indices by protein for efficient lookup
    protein_indices = {}  # bmrb_id -> [(global_idx, residue_id), ...]
    for gidx_str, bmrb_id in idx_to_bmrb.items():
        gidx = int(gidx_str)
        rid = global_to_resid.get(gidx_str)
        if rid is not None:
            if bmrb_id not in protein_indices:
                protein_indices[bmrb_id] = []
            protein_indices[bmrb_id].append((gidx, int(rid)))

    # Fill bond geometry from CSV data
    n_filled = 0
    for bmrb_id, indices in tqdm(protein_indices.items(), desc=f"  {fold_dir}"):
        pdf = df_by_bmrb.get(bmrb_id)
        if pdf is None:
            continue

        # Index CSV rows by residue_id for this protein
        pdf_indexed = pdf.set_index('residue_id')

        for gidx, rid in indices:
            if rid not in pdf_indexed.index:
                continue
            row = pdf_indexed.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            for bi, col in enumerate(BOND_GEOM_COLS):
                val = row.get(col)
                if val is not None and not pd.isna(val):
                    flat_bond_geom[gidx, bi] = val / 10.0  # normalize
                    n_filled += 1

    np.save(sd / 'bond_geom.npy', flat_bond_geom)
    print(f"  Saved {sd / 'bond_geom.npy'}: {total_residues} residues, "
          f"{n_filled} values filled")


def main():
    cache_dir = 'data/cache'
    csv_path = 'data/structure_data_hybrid.csv'

    print("=" * 70)
    print("PATCH: Adding bond_geom.npy to existing caches")
    print("=" * 70)

    t0 = time.time()

    # Load just the needed columns from CSV
    print(f"\nLoading CSV columns: bmrb_id, residue_id, {BOND_GEOM_COLS}")
    needed = ['bmrb_id', 'residue_id'] + BOND_GEOM_COLS
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols_to_load = [c for c in needed if c in available]
    df = pd.read_csv(csv_path, usecols=cols_to_load, dtype={'bmrb_id': str})
    print(f"  Loaded {len(df):,} rows")

    # Group by protein once
    df_by_bmrb = {str(bid): grp for bid, grp in df.groupby('bmrb_id')}
    del df

    # Patch each fold
    for fold_name in sorted(os.listdir(cache_dir)):
        fold_dir = os.path.join(cache_dir, fold_name)
        if os.path.isdir(fold_dir) and os.path.exists(os.path.join(fold_dir, 'config.json')):
            patch_fold(fold_dir, df_by_bmrb)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == '__main__':
    main()
