#!/usr/bin/env python3
"""
Add DSSP columns (phi, psi, secondary_structure, rel_acc, H-bond features)
to an existing structure_data CSV that's missing them.

Reads the CSV in chunks, runs DSSP on each protein's PDB file,
and writes a new CSV with the DSSP columns added.

Usage:
    python add_dssp_to_csv.py --input data/structure_data_hybrid.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DSSP_COLS, PDB_DIR, ALPHAFOLD_DIR, DATA_DIR
from pdb_utils import run_dssp, lookup_with_chain_fallback

# All DSSP-derived columns we want to add
DSSP_OUTPUT_COLS = ['secondary_structure', 'phi', 'psi'] + list(DSSP_COLS)


def get_pdb_path_for_protein(bmrb_id, pairs_map, alphafold_map):
    """Find the PDB file for a protein (experimental or AlphaFold)."""
    # Try experimental PDB first
    if bmrb_id in pairs_map:
        for pdb_id in pairs_map[bmrb_id]:
            for d in [PDB_DIR, 'data/pdbs']:
                for ext in ['.pdb']:
                    path = os.path.join(d, f'{pdb_id.upper()}{ext}')
                    if os.path.exists(path):
                        return path
    # Try AlphaFold
    if bmrb_id in alphafold_map:
        af_path = alphafold_map[bmrb_id]
        if os.path.exists(af_path):
            return af_path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV missing DSSP columns')
    parser.add_argument('--output', default=None, help='Output CSV (default: overwrites input)')
    parser.add_argument('--chunksize', type=int, default=50000)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    # Check if DSSP columns already exist
    header = pd.read_csv(args.input, nrows=0).columns.tolist()
    if 'phi' in header and 'psi' in header and 'secondary_structure' in header:
        # Check if they actually have data (not all NaN)
        sample = pd.read_csv(args.input, usecols=['phi'], nrows=100)
        if sample['phi'].notna().any():
            print("DSSP columns already present with data. Nothing to do.")
            return
        print("DSSP columns exist but are empty. Populating...")

    print(f"Adding DSSP columns to {args.input}")

    # Build PDB path lookup
    # Load pairs.csv for experimental PDB mapping
    import json
    pairs_map = {}
    pairs_path = os.path.join(DATA_DIR, 'pairs.csv')
    if os.path.exists(pairs_path):
        pairs_df = pd.read_csv(pairs_path, dtype={'Entry_ID': str})
        for _, row in pairs_df.iterrows():
            bid = str(row['Entry_ID'])
            pdbs = str(row.get('pdb_ids', '')).split(',')
            pairs_map[bid] = [p.strip().upper() for p in pdbs if p.strip()]

    # Load AlphaFold mapping
    alphafold_map = {}
    from alphafold_utils import AF_MODEL_VERSION
    for mapping_file in ['bmrb_uniprot_mapping.json', 'new_uniprot_mappings.json']:
        path = os.path.join(ALPHAFOLD_DIR, mapping_file)
        if os.path.exists(path):
            with open(path) as f:
                mapping = json.load(f)
            for bid, uid in mapping.items():
                af_path = os.path.join(ALPHAFOLD_DIR, f'AF-{uid}-F1-{AF_MODEL_VERSION}.pdb')
                if os.path.exists(af_path):
                    alphafold_map[bid] = af_path
    print(f"  PDB mappings: {len(pairs_map)}, AlphaFold mappings: {len(alphafold_map)}")

    # Cache DSSP results per PDB file (many residues share same PDB)
    dssp_cache = {}

    def get_dssp_for_protein(bmrb_id):
        pdb_path = get_pdb_path_for_protein(bmrb_id, pairs_map, alphafold_map)
        if pdb_path is None:
            return None
        if pdb_path not in dssp_cache:
            dssp_cache[pdb_path] = run_dssp(pdb_path)
        return dssp_cache[pdb_path]

    # Process in chunks
    tmp_output = args.output + '.tmp'
    header_written = False
    total_rows = 0
    dssp_found = 0
    dssp_missing = 0

    reader = pd.read_csv(args.input, chunksize=args.chunksize,
                         dtype={'bmrb_id': str}, low_memory=False)

    for chunk_idx, chunk in enumerate(reader):
        # Add empty DSSP columns
        for col in DSSP_OUTPUT_COLS:
            if col not in chunk.columns:
                chunk[col] = np.nan

        # Group by protein for DSSP lookup
        for bmrb_id, group_idx in chunk.groupby('bmrb_id').groups.items():
            dssp_data = get_dssp_for_protein(str(bmrb_id))
            if dssp_data is None or not dssp_data:
                dssp_missing += len(group_idx)
                continue

            for idx in group_idx:
                rid = int(chunk.at[idx, 'residue_id'])
                # Try multiple chain keys
                entry = None
                for chain_key in ['A', '', ' ']:
                    if (chain_key, rid) in dssp_data:
                        entry = dssp_data[(chain_key, rid)]
                        break
                if entry is None:
                    continue

                chunk.at[idx, 'secondary_structure'] = entry.get('secondary_structure', 'C')
                phi = entry.get('phi')
                psi = entry.get('psi')
                chunk.at[idx, 'phi'] = phi if phi is not None else np.nan
                chunk.at[idx, 'psi'] = psi if psi is not None else np.nan
                for col in DSSP_COLS:
                    val = entry.get(col)
                    if val is not None:
                        chunk.at[idx, col] = val
                dssp_found += 1

        # Write chunk
        chunk.to_csv(tmp_output, mode='a' if header_written else 'w',
                     index=False, header=not header_written)
        header_written = True
        total_rows += len(chunk)

        # Clear DSSP cache periodically to avoid memory buildup
        if len(dssp_cache) > 5000:
            dssp_cache.clear()

        print(f"  Chunk {chunk_idx}: {total_rows:,} rows, "
              f"DSSP found: {dssp_found:,}, missing: {dssp_missing:,}")

    # Replace original
    if os.path.exists(tmp_output):
        os.replace(tmp_output, args.output)
        print(f"\nDone. Wrote {args.output}")
        print(f"  Total rows: {total_rows:,}")
        print(f"  DSSP found: {dssp_found:,}")
        print(f"  DSSP missing: {dssp_missing:,}")


if __name__ == '__main__':
    main()
