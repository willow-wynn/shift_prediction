#!/usr/bin/env python3
"""
Patch existing structure_data_hybrid.csv to add inter-residue bond geometry columns.

Much faster than re-running 01_build_datasets.py from scratch:
- CA-CA distances computed from existing ca_x/ca_y/ca_z columns in the CSV
- Peptide bond lengths (C-N) computed by parsing PDB files from build_log.csv

Outputs a new CSV with 4 additional columns:
  bond_ca_prev, bond_ca_next, bond_peptide_fwd, bond_peptide_bkwd
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdb_utils import parse_pdb
from config import AA_3_TO_1, BOND_GEOM_COLS


def get_backbone_coords_from_pdb(pdb_path, chain_id):
    """Parse PDB and extract C, N, CA coords per residue."""
    try:
        residues = parse_pdb(pdb_path, chain_id=chain_id)
    except Exception:
        return {}

    coords = {}
    for (chain, res_id), rdata in residues.items():
        atoms = rdata.get('atoms', {})
        entry = {}
        for atom_name in ('CA', 'C', 'N'):
            if atom_name in atoms and np.all(np.isfinite(atoms[atom_name])):
                entry[atom_name] = atoms[atom_name]
        if entry:
            coords[res_id] = entry
    return coords


def main():
    data_dir = 'data'
    csv_path = os.path.join(data_dir, 'structure_data_hybrid.csv')
    log_path = os.path.join(data_dir, 'build_log.csv')
    output_path = os.path.join(data_dir, 'structure_data_hybrid_bonded.csv')

    print("=" * 70)
    print("PATCH: Adding inter-residue bond geometry columns")
    print("=" * 70)

    t0 = time.time()

    # --- Step 1: Load build log to get PDB paths ---
    print("\nLoading build log...")
    log_df = pd.read_csv(log_path, dtype={'bmrb_id': str})
    log_df = log_df[(log_df['dataset'] == 'hybrid') & (log_df['status'] == 'success')]
    pdb_info = {}
    for _, row in log_df.iterrows():
        pdb_info[row['bmrb_id']] = (row['pdb_path'], row['chain_id'])
    print(f"  {len(pdb_info)} proteins with PDB info")

    # --- Step 2: Parse PDB files to get C/N coordinates ---
    print("\nParsing PDB files for backbone C/N coordinates...")
    protein_backbone = {}  # bmrb_id -> {res_id -> {CA, C, N}}
    failed = 0
    for bmrb_id, (pdb_path, chain_id) in tqdm(pdb_info.items(), desc="  Parsing PDBs"):
        if not os.path.exists(pdb_path):
            failed += 1
            continue
        coords = get_backbone_coords_from_pdb(pdb_path, chain_id)
        if coords:
            protein_backbone[bmrb_id] = coords
        else:
            failed += 1

    print(f"  Parsed {len(protein_backbone)} proteins, {failed} failed")

    # --- Step 3: Read CSV and compute bond geometry ---
    print(f"\nReading CSV: {csv_path}")
    # Only load the columns we need for the computation
    needed_cols = ['bmrb_id', 'residue_id', 'ca_x', 'ca_y', 'ca_z']
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    light_df = pd.read_csv(csv_path, usecols=needed_cols, dtype={'bmrb_id': str})
    print(f"  {len(light_df):,} rows loaded")

    # Initialize new columns
    bond_ca_prev = np.full(len(light_df), np.nan, dtype=np.float64)
    bond_ca_next = np.full(len(light_df), np.nan, dtype=np.float64)
    bond_peptide_fwd = np.full(len(light_df), np.nan, dtype=np.float64)
    bond_peptide_bkwd = np.full(len(light_df), np.nan, dtype=np.float64)

    print("\nComputing bond geometry per protein...")
    n_computed = 0

    for bmrb_id, group in tqdm(light_df.groupby('bmrb_id'), desc="  Proteins"):
        idx = group.index.values
        rids = group['residue_id'].values
        ca_x = group['ca_x'].values
        ca_y = group['ca_y'].values
        ca_z = group['ca_z'].values

        # Sort by residue_id
        order = np.argsort(rids)
        idx = idx[order]
        rids = rids[order]
        ca_x = ca_x[order]
        ca_y = ca_y[order]
        ca_z = ca_z[order]

        # Get PDB backbone coords for peptide bonds
        pdb_coords = protein_backbone.get(str(bmrb_id), {})

        n = len(rids)
        for i in range(n):
            ca_valid_i = np.isfinite(ca_x[i]) and np.isfinite(ca_y[i]) and np.isfinite(ca_z[i])
            ca_i = np.array([ca_x[i], ca_y[i], ca_z[i]]) if ca_valid_i else None
            rid_i = int(rids[i])
            pdb_i = pdb_coords.get(rid_i, {})

            # Previous residue
            if i > 0:
                ca_valid_prev = np.isfinite(ca_x[i-1]) and np.isfinite(ca_y[i-1]) and np.isfinite(ca_z[i-1])
                if ca_valid_i and ca_valid_prev:
                    ca_prev = np.array([ca_x[i-1], ca_y[i-1], ca_z[i-1]])
                    bond_ca_prev[idx[i]] = np.linalg.norm(ca_i - ca_prev)

                # Peptide bond backward: C(i-1) -> N(i)
                rid_prev = int(rids[i-1])
                pdb_prev = pdb_coords.get(rid_prev, {})
                if 'C' in pdb_prev and 'N' in pdb_i:
                    bond_peptide_bkwd[idx[i]] = np.linalg.norm(pdb_i['N'] - pdb_prev['C'])

            # Next residue
            if i < n - 1:
                ca_valid_next = np.isfinite(ca_x[i+1]) and np.isfinite(ca_y[i+1]) and np.isfinite(ca_z[i+1])
                if ca_valid_i and ca_valid_next:
                    ca_next = np.array([ca_x[i+1], ca_y[i+1], ca_z[i+1]])
                    bond_ca_next[idx[i]] = np.linalg.norm(ca_i - ca_next)

                # Peptide bond forward: C(i) -> N(i+1)
                rid_next = int(rids[i+1])
                pdb_next = pdb_coords.get(rid_next, {})
                if 'C' in pdb_i and 'N' in pdb_next:
                    bond_peptide_fwd[idx[i]] = np.linalg.norm(pdb_next['N'] - pdb_i['C'])

        n_computed += 1

    # Stats
    for name, arr in [('bond_ca_prev', bond_ca_prev), ('bond_ca_next', bond_ca_next),
                       ('bond_peptide_fwd', bond_peptide_fwd), ('bond_peptide_bkwd', bond_peptide_bkwd)]:
        valid = np.isfinite(arr)
        vals = arr[valid]
        print(f"  {name}: {valid.sum():,}/{len(arr):,} valid "
              f"(mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"min={vals.min():.3f}, max={vals.max():.3f})")

    # --- Step 4: Write updated CSV ---
    print(f"\nWriting patched CSV: {output_path}")
    print("  Reading full CSV in chunks and appending new columns...")

    # Read and write in chunks to avoid loading 6GB+ into memory
    CHUNK_SIZE = 100_000
    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, dtype={'bmrb_id': str},
                          low_memory=False)

    row_offset = 0
    for ci, chunk in enumerate(tqdm(reader, desc="  Writing chunks")):
        n_rows = len(chunk)
        chunk['bond_ca_prev'] = bond_ca_prev[row_offset:row_offset + n_rows]
        chunk['bond_ca_next'] = bond_ca_next[row_offset:row_offset + n_rows]
        chunk['bond_peptide_fwd'] = bond_peptide_fwd[row_offset:row_offset + n_rows]
        chunk['bond_peptide_bkwd'] = bond_peptide_bkwd[row_offset:row_offset + n_rows]

        chunk.to_csv(output_path, mode='a' if ci > 0 else 'w',
                     index=False, header=(ci == 0))
        row_offset += n_rows

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Output: {output_path}")
    print(f"  Rows: {row_offset:,}")
    print(f"\nTo use: update the symlink:")
    print(f"  rm data/structure_data_hybrid.csv")
    print(f"  mv {output_path} data/structure_data_hybrid.csv")


if __name__ == '__main__':
    main()
