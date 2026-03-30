#!/usr/bin/env python3
"""
Build FAISS retrieval index and training caches from structure embeddings.

Wrapper that handles the memory issue with the full CSV by creating a
lightweight CSV with only the columns needed for index/cache building,
then calls existing 04 and 05 scripts.

Usage:
    python build_struct_retrieval.py
"""

import os
import sys
import subprocess
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_index_csv(full_csv, output_csv):
    """Create a minimal CSV for 04_build_retrieval_index.py.

    04 only needs: bmrb_id, residue_id, residue_code, split, *_shift columns.
    The full CSV has ~1400 dist_* columns making it 6.6GB — too large for pandas.
    This minimal version is ~200MB and loads fast.
    """
    print(f"Creating index CSV from {full_csv}...")

    all_cols = pd.read_csv(full_csv, nrows=0).columns.tolist()

    shift_cols = sorted([c for c in all_cols if c.endswith('_shift')])
    meta_cols = ['bmrb_id', 'residue_id', 'residue_code', 'split']
    meta_cols = [c for c in meta_cols if c in all_cols]
    needed = meta_cols + shift_cols

    print(f"  Full CSV: {len(all_cols)} columns")
    print(f"  Index CSV: {len(needed)} columns")

    reader = pd.read_csv(full_csv, usecols=needed, dtype={'bmrb_id': str},
                          chunksize=200000, low_memory=False)
    for ci, chunk in enumerate(tqdm(reader, desc="  Writing")):
        chunk.to_csv(output_csv, mode='a' if ci > 0 else 'w',
                     index=False, header=(ci == 0))

    print(f"  Saved: {output_csv}")


def main():
    base_dir = 'data/struct_retrieval'
    full_csv = 'data/structure_data_hybrid.csv'
    index_csv = os.path.join(base_dir, 'structure_data_index.csv')
    embeddings_h5 = os.path.join(base_dir, 'struct_embeddings.h5')
    index_dir = os.path.join(base_dir, 'retrieval_indices')
    cache_dir = os.path.join(base_dir, 'cache')

    os.makedirs(base_dir, exist_ok=True)

    print("=" * 70)
    print("BUILD STRUCTURE-BOOTSTRAPPED RETRIEVAL PIPELINE")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Create minimal CSV for index building (04 loads full CSV into RAM)
    if not os.path.exists(index_csv):
        create_index_csv(full_csv, index_csv)
    else:
        print(f"\nIndex CSV already exists: {index_csv}")

    # Step 2: Build FAISS indices (using minimal CSV)
    print(f"\n{'='*70}")
    print("Step 2: Building FAISS indices")
    print(f"{'='*70}")

    # 04 looks for structure_data_hybrid.csv in data_dir
    # Symlink the minimal CSV so 04 finds it
    csv_link = os.path.join(base_dir, 'structure_data_hybrid.csv')
    if os.path.exists(csv_link) or os.path.islink(csv_link):
        os.remove(csv_link)
    os.symlink(os.path.abspath(index_csv), csv_link)

    cmd = [
        sys.executable, '04_build_retrieval_index.py',
        '--data_dir', base_dir,
        '--embeddings', embeddings_h5,
        '--output_dir', index_dir,
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: FAISS index build failed (exit code {result.returncode})")
        sys.exit(1)

    # Step 3: Build training caches (using full CSV for distance columns)
    print(f"\n{'='*70}")
    print("Step 3: Building training caches")
    print(f"{'='*70}")

    # Point symlink to full CSV for 05 (needs dist_* columns)
    os.remove(csv_link)
    os.symlink(os.path.abspath(full_csv), csv_link)

    cmd = [
        sys.executable, '05_build_training_cache.py',
        '--data_dir', base_dir,
        '--embeddings', embeddings_h5,
        '--index_dir', index_dir,
        '--output_dir', cache_dir,
        '--device', 'cuda',
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Cache build failed (exit code {result.returncode})")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"  Indices: {index_dir}")
    print(f"  Caches:  {cache_dir}")
    print(f"\nTo train:")
    print(f"  python 06_train.py --data_dir {base_dir} --cache_dir {cache_dir} "
          f"--output_dir {base_dir}/checkpoints --fold 1 --batch_size 512")


if __name__ == '__main__':
    main()
