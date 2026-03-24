#!/usr/bin/env python3
"""
Add pLDDT column to an AlphaFold structure_data CSV by reading B-factors from PDB files.

Usage:
    python add_plddt_to_csv.py --input data/structure_data_alphafold.csv --pdb_dir data/alphafold
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdb_utils import extract_bfactors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--pdb_dir', default='data/alphafold')
    parser.add_argument('--chunksize', type=int, default=50000)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    header = pd.read_csv(args.input, nrows=0).columns
    if 'plddt' in header:
        sample = pd.read_csv(args.input, usecols=['plddt'], nrows=100)
        if sample['plddt'].notna().any():
            print("pLDDT column already present. Nothing to do.")
            return

    print(f"Adding pLDDT to {args.input} from {args.pdb_dir}")

    # Build mapping: bmrb_id -> AlphaFold PDB path
    # The CSV has bmrb_id; we need to find the corresponding AF PDB
    # Read a sample to get bmrb_ids, then search for matching PDBs

    # Cache B-factors per PDB file
    bfactor_cache = {}

    # We need a mapping from bmrb_id to PDB file. Check if there's a uniprot mapping.
    import json
    bmrb_to_pdb = {}
    for mapping_file in ['bmrb_uniprot_mapping.json', 'new_uniprot_mappings.json']:
        path = os.path.join(args.pdb_dir, mapping_file)
        if os.path.exists(path):
            with open(path) as f:
                mapping = json.load(f)
            for bid, uid in mapping.items():
                pdb_path = os.path.join(args.pdb_dir, f'AF-{uid}-F1-model_v6.pdb')
                if os.path.exists(pdb_path):
                    bmrb_to_pdb[bid] = pdb_path

    # Also try direct PDB lookup patterns
    if not bmrb_to_pdb:
        # Fall back: scan PDB dir and match by any available method
        print("  No uniprot mapping found, scanning PDB directory...")

    print(f"  Found {len(bmrb_to_pdb)} bmrb_id -> PDB mappings")

    tmp_output = args.output + '.tmp'
    header_written = False
    total = 0
    found = 0

    for chunk in pd.read_csv(args.input, chunksize=args.chunksize,
                              dtype={'bmrb_id': str}, low_memory=False):
        if 'plddt' not in chunk.columns:
            chunk['plddt'] = np.nan

        for bmrb_id, group_idx in chunk.groupby('bmrb_id').groups.items():
            bid = str(bmrb_id)
            pdb_path = bmrb_to_pdb.get(bid)
            if pdb_path is None:
                continue

            if pdb_path not in bfactor_cache:
                bfactor_cache[pdb_path] = extract_bfactors(pdb_path)

            bfacs = bfactor_cache[pdb_path]
            for idx in group_idx:
                rid = int(chunk.at[idx, 'residue_id'])
                # Try chain A (standard for AlphaFold)
                bfac = bfacs.get(('A', rid))
                if bfac is not None:
                    chunk.at[idx, 'plddt'] = bfac
                    found += 1

        chunk.to_csv(tmp_output, mode='a' if header_written else 'w',
                     index=False, header=not header_written)
        header_written = True
        total += len(chunk)

        if len(bfactor_cache) > 5000:
            bfactor_cache.clear()

        print(f"  {total:,} rows, pLDDT found: {found:,}")

    if os.path.exists(tmp_output):
        os.replace(tmp_output, args.output)
        print(f"\nDone. {total:,} rows, pLDDT found: {found:,}")


if __name__ == '__main__':
    main()
