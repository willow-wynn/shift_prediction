#!/usr/bin/env python3
"""
Step 00: Fetch BMRB chemical shift data.

Downloads chemical shift entries from the BMRB REST API, pivots into
one-row-per-residue format, and saves raw (unfiltered) shift data.

Quality filtering is handled downstream in 01_build_datasets.py.

Usage:
    # Download all ~14k BMRB entries (with checkpoint/resume)
    python 00_fetch_bmrb_shifts.py --download --checkpoint-dir data/checkpoints

    # Load from existing local file (no network)
    python 00_fetch_bmrb_shifts.py --input data/chemical_shifts.csv

Outputs:
    data/chemical_shifts.csv  -- raw pivoted shifts (no filtering applied)
"""

import argparse
import os
import sys
import time
import json
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR


# ============================================================================
# BMRB API
# ============================================================================

BMRB_LIST_URL = 'https://api.bmrb.io/v2/list_entries?database=macromolecules'
BMRB_ENTRY_URL = 'https://api.bmrb.io/v2/entry'

# Atoms we want to pivot into columns
TARGET_ATOMS = [
    'c', 'ca', 'cb', 'cd1', 'cd2', 'cd', 'ce1', 'ce2', 'ce',
    'cg1', 'cg2', 'cg', 'cz', 'h', 'ha2', 'ha3', 'ha',
    'hb1', 'hb2', 'hb3', 'hb', 'hd11', 'hd12', 'hd13', 'hd1',
    'hd21', 'hd22', 'hd23', 'hd2', 'hd3', 'he1', 'he21', 'he22',
    'he2', 'he3', 'he', 'hg11', 'hg12', 'hg13', 'hg21', 'hg22',
    'hg23', 'hg2', 'hg3', 'hg', 'hz', 'n', 'nd2', 'ne2',
]


def get_all_bmrb_entry_ids():
    """Fetch the list of all macromolecule entry IDs from BMRB.

    Returns:
        List of entry ID strings, or empty list on failure.
    """
    for attempt in range(3):
        try:
            req = urllib.request.Request(BMRB_LIST_URL)
            req.add_header('Accept', 'application/json')
            req.add_header('User-Agent', 'he_lab_pipeline/1.0')
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            # Response is a list of entry IDs
            if isinstance(data, list):
                return [str(eid) for eid in data]
            if isinstance(data, dict):
                for key in ('data', 'result', 'entry_ids'):
                    if key in data and isinstance(data[key], list):
                        return [str(eid) for eid in data[key]]
            return []
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: Failed to fetch BMRB entry list: {e}")
                return []
    return []


def download_bmrb_entry(bmrb_id, max_retries=3):
    """Download chemical shift data for a single BMRB entry via REST API.

    Returns a list of dicts with: bmrb_id, residue_id, residue_code, atom_id, value, ambiguity_code
    Returns empty list on failure.
    """
    url = f'{BMRB_ENTRY_URL}/{bmrb_id}?format=json'

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: Failed to download BMRB {bmrb_id}: {e}")
                return []

    rows = []
    try:
        for saveframe in data.get('data', []):
            if not isinstance(saveframe, dict):
                continue
            loops = saveframe.get('loops', [])
            for loop in loops:
                tags = loop.get('tags', [])
                loop_data = loop.get('data', [])

                if 'Atom_chem_shift.Val' not in tags and 'Val' not in tags:
                    continue

                col_map = {t.split('.')[-1] if '.' in t else t: i for i, t in enumerate(tags)}

                seq_idx = col_map.get('Seq_ID')
                comp_idx = col_map.get('Comp_ID')
                atom_idx = col_map.get('Atom_ID')
                val_idx = col_map.get('Val')
                ambig_idx = col_map.get('Ambiguity_code')

                if any(idx is None for idx in [seq_idx, comp_idx, atom_idx, val_idx]):
                    continue

                for row_data in loop_data:
                    try:
                        seq_id = row_data[seq_idx]
                        comp_id = row_data[comp_idx]
                        atom_id = row_data[atom_idx]
                        val = row_data[val_idx]
                        ambig = row_data[ambig_idx] if ambig_idx is not None else None

                        if val in (None, '.', '') or seq_id in (None, '.', ''):
                            continue

                        rows.append({
                            'bmrb_id': str(bmrb_id),
                            'residue_id': int(seq_id),
                            'residue_code': str(comp_id).upper(),
                            'atom_id': str(atom_id).upper(),
                            'value': float(val),
                            'ambiguity_code': int(ambig) if ambig not in (None, '.', '') else None,
                        })
                    except (ValueError, IndexError, TypeError):
                        continue
    except Exception as e:
        print(f"  WARNING: Error parsing BMRB {bmrb_id}: {e}")

    return rows


def pivot_shift_data(flat_rows):
    """Pivot flat (one-row-per-atom) shift data into one-row-per-residue format.

    Columns: bmrb_id, residue_id, residue_code, {atom}_shift, {atom}_ambiguity_code, ...
    """
    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    df['atom_lower'] = df['atom_id'].str.lower()
    df = df[df['atom_lower'].isin(TARGET_ATOMS)].copy()

    if df.empty:
        return pd.DataFrame()

    shift_pivot = df.pivot_table(
        index=['bmrb_id', 'residue_id', 'residue_code'],
        columns='atom_lower',
        values='value',
        aggfunc='first',
    )
    shift_pivot.columns = [f'{c}_shift' for c in shift_pivot.columns]

    ambig_pivot = df.pivot_table(
        index=['bmrb_id', 'residue_id', 'residue_code'],
        columns='atom_lower',
        values='ambiguity_code',
        aggfunc='first',
    )
    ambig_pivot.columns = [f'{c}_ambiguity_code' for c in ambig_pivot.columns]

    result = pd.concat([shift_pivot, ambig_pivot], axis=1).reset_index()
    return result


def load_existing_shifts(input_path):
    """Load an existing chemical_shifts CSV file.

    Returns DataFrame with bmrb_id as string dtype, or None if not found.
    """
    if not os.path.exists(input_path):
        return None

    print(f"Loading existing shift data from {input_path}...")
    df = pd.read_csv(input_path, dtype={'bmrb_id': str})
    print(f"  Loaded {len(df):,} residues from {df['bmrb_id'].nunique():,} proteins")
    return df


def download_all_shifts(entry_ids, checkpoint_dir):
    """Download shift data for all BMRB entries with checkpoint/resume.

    Saves intermediate JSON per entry in checkpoint_dir so interrupted runs
    can resume without re-downloading.

    Args:
        entry_ids: list of BMRB entry ID strings
        checkpoint_dir: directory for per-entry JSON checkpoints

    Returns:
        list of flat row dicts (all entries concatenated)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_rows = []
    n_total = len(entry_ids)
    n_cached = 0
    n_downloaded = 0
    n_empty = 0
    n_failed = 0

    for i, eid in enumerate(entry_ids):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{n_total} "
                  f"(cached={n_cached}, downloaded={n_downloaded}, "
                  f"empty={n_empty}, failed={n_failed})")

        checkpoint_path = os.path.join(checkpoint_dir, f'{eid}.json')

        # Check for cached result
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    rows = json.load(f)
                all_rows.extend(rows)
                n_cached += 1
                continue
            except (json.JSONDecodeError, IOError):
                pass  # Re-download if corrupted

        # Download
        rows = download_bmrb_entry(eid)

        # Save checkpoint
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(rows, f)
        except IOError:
            pass

        if rows:
            all_rows.extend(rows)
            n_downloaded += 1
        else:
            n_empty += 1

        # Polite delay
        if n_downloaded > 0 and n_downloaded % 10 == 0:
            time.sleep(0.1)

    print(f"\n  Download complete:")
    print(f"    Cached:     {n_cached:,}")
    print(f"    Downloaded: {n_downloaded:,}")
    print(f"    Empty:      {n_empty:,}")
    print(f"    Failed:     {n_failed:,}")
    print(f"    Total rows: {len(all_rows):,}")

    return all_rows


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fetch and pivot BMRB chemical shift data (no filtering).'
    )
    parser.add_argument(
        '--input', default=None,
        help='Path to existing chemical shifts CSV (skip download if provided)'
    )
    parser.add_argument(
        '--output-dir', default=DATA_DIR,
        help=f'Output directory (default: {DATA_DIR})'
    )
    parser.add_argument(
        '--download', action='store_true', default=False,
        help='Download all entries from BMRB REST API'
    )
    parser.add_argument(
        '--checkpoint-dir', default='data/checkpoints',
        help='Directory for download checkpoints (default: data/checkpoints)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 00: FETCH BMRB CHEMICAL SHIFT DATA")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(path):
        return os.path.join(script_dir, path) if not os.path.isabs(path) else path

    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'chemical_shifts.csv')

    if args.input:
        # Load from existing file
        input_path = resolve(args.input)
        df = load_existing_shifts(input_path)
        if df is None:
            print(f"  ERROR: File not found: {input_path}")
            sys.exit(1)

    elif args.download:
        # Download all entries from BMRB
        checkpoint_dir = resolve(args.checkpoint_dir)

        print("\n[Step 1] Fetching BMRB entry list...")
        entry_ids = get_all_bmrb_entry_ids()
        if not entry_ids:
            print("  ERROR: Could not fetch BMRB entry list.")
            sys.exit(1)
        print(f"  Found {len(entry_ids):,} entries")

        print("\n[Step 2] Downloading shift data (with checkpoints)...")
        all_rows = download_all_shifts(entry_ids, checkpoint_dir)

        if not all_rows:
            print("  ERROR: No shift data downloaded.")
            sys.exit(1)

        print("\n[Step 3] Pivoting to one-row-per-residue format...")
        df = pivot_shift_data(all_rows)

    else:
        # Try default location, auto-download if missing
        default_input = os.path.join(output_dir, 'chemical_shifts.csv')
        df = load_existing_shifts(default_input)
        if df is None:
            print("  No existing shift data found. Auto-downloading from BMRB...")
            checkpoint_dir = resolve(args.checkpoint_dir)

            print("\n[Step 1] Fetching BMRB entry list...")
            entry_ids = get_all_bmrb_entry_ids()
            if not entry_ids:
                print("  ERROR: Could not fetch BMRB entry list.")
                sys.exit(1)
            print(f"  Found {len(entry_ids):,} entries")

            print("\n[Step 2] Downloading shift data (with checkpoints)...")
            all_rows = download_all_shifts(entry_ids, checkpoint_dir)

            if not all_rows:
                print("  ERROR: No shift data downloaded.")
                sys.exit(1)

            print("\n[Step 3] Pivoting to one-row-per-residue format...")
            df = pivot_shift_data(all_rows)

    # Ensure bmrb_id is string
    df['bmrb_id'] = df['bmrb_id'].astype(str)

    # Save
    print(f"\n[Saving] {output_path}")
    df.to_csv(output_path, index=False)

    n_proteins = df['bmrb_id'].nunique()
    n_residues = len(df)
    shift_cols = [c for c in df.columns if c.endswith('_shift')]

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Proteins: {n_proteins:,}")
    print(f"  Residues: {n_residues:,}")
    print(f"  Shift columns: {len(shift_cols)}")
    print(f"  Output: {output_path}")
    print()


if __name__ == '__main__':
    main()
