#!/usr/bin/env python3
"""
Step 00: Fetch and clean BMRB chemical shift data.

Reads the existing chemical_shifts_pivoted.csv (or downloads from BMRB REST API
if unavailable), applies data quality filters, and outputs a cleaned shift file
with full provenance tracking.

Filters applied:
1. Non-standard residue removal/mapping (SEC->CYS, MSE->MET, etc.)
2. Duplicate (bmrb_id, residue_id) removal
3. Physical range outlier detection (sets out-of-range values to NaN)

Outputs:
- data/chemical_shifts.csv       -- cleaned shift data
- data/shift_quality_log.csv     -- provenance log (every filtering decision)
"""

import argparse
import os
import sys
import time
import urllib.request
import json

import pandas as pd
import numpy as np

# Allow imports from the homologies_better_data package
sys.path.insert(0, os.path.dirname(__file__))
from config import SHIFT_RANGES, NONSTANDARD_MAP, STANDARD_RESIDUES, DATA_DIR
from data_quality import FilterLog, filter_shift_outliers, filter_standard_residues, remove_duplicates


# ============================================================================
# BMRB API Download (fallback if local file not found)
# ============================================================================

BMRB_SEARCH_URL = 'https://api.bmrb.io/v2/search/chemical_shifts'
BMRB_ENTRY_URL = 'https://api.bmrb.io/v2/entry'

# Atoms we want to pivot into columns (matches chemical_shifts_pivoted.csv)
TARGET_ATOMS = [
    'c', 'ca', 'cb', 'cd1', 'cd2', 'cd', 'ce1', 'ce2', 'ce',
    'cg1', 'cg2', 'cg', 'cz', 'h', 'ha2', 'ha3', 'ha',
    'hb1', 'hb2', 'hb3', 'hb', 'hd11', 'hd12', 'hd13', 'hd1',
    'hd21', 'hd22', 'hd23', 'hd2', 'hd3', 'he1', 'he21', 'he22',
    'he2', 'he3', 'he', 'hg11', 'hg12', 'hg13', 'hg21', 'hg22',
    'hg23', 'hg2', 'hg3', 'hg', 'hz', 'n', 'nd2', 'ne2',
]


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
        # Navigate NMR-STAR JSON structure
        for saveframe in data.get('data', []):
            if not isinstance(saveframe, dict):
                continue
            loops = saveframe.get('loops', [])
            for loop in loops:
                tags = loop.get('tags', [])
                loop_data = loop.get('data', [])

                # Look for chemical shift loop
                if 'Atom_chem_shift.Val' not in tags and 'Val' not in tags:
                    continue

                # Build column index
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

    Matches the column format of chemical_shifts_pivoted.csv:
    bmrb_id, residue_id, residue_code, split, {atom}_shift, {atom}_ambiguity_code, ...
    """
    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    df['atom_lower'] = df['atom_id'].str.lower()

    # Keep only atoms we care about
    df = df[df['atom_lower'].isin(TARGET_ATOMS)].copy()

    if df.empty:
        return pd.DataFrame()

    # Pivot shifts
    shift_pivot = df.pivot_table(
        index=['bmrb_id', 'residue_id', 'residue_code'],
        columns='atom_lower',
        values='value',
        aggfunc='first',
    )
    shift_pivot.columns = [f'{c}_shift' for c in shift_pivot.columns]

    # Pivot ambiguity codes
    ambig_pivot = df.pivot_table(
        index=['bmrb_id', 'residue_id', 'residue_code'],
        columns='atom_lower',
        values='ambiguity_code',
        aggfunc='first',
    )
    ambig_pivot.columns = [f'{c}_ambiguity_code' for c in ambig_pivot.columns]

    # Merge
    result = pd.concat([shift_pivot, ambig_pivot], axis=1).reset_index()
    result['split'] = 1  # Placeholder; will be reassigned later in pipeline

    return result


def load_existing_shifts(input_path):
    """Load the existing chemical_shifts_pivoted.csv file.

    Returns DataFrame with bmrb_id as string dtype.
    """
    if not os.path.exists(input_path):
        return None

    print(f"Loading existing shift data from {input_path}...")
    df = pd.read_csv(input_path, dtype={'bmrb_id': str})
    print(f"  Loaded {len(df):,} residues from {df['bmrb_id'].nunique():,} proteins")
    return df


# ============================================================================
# Summary Printing
# ============================================================================

def print_shift_coverage(df, label=""):
    """Print per-atom shift coverage statistics."""
    n_residues = len(df)
    if n_residues == 0:
        print(f"  {label}: empty DataFrame")
        return

    print(f"\n  {label} Backbone shift coverage ({n_residues:,} residues):")
    for col, (lo, hi) in SHIFT_RANGES.items():
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct = 100.0 * n_valid / n_residues
            n_in_range = ((df[col].dropna() >= lo) & (df[col].dropna() <= hi)).sum()
            n_outlier = n_valid - n_in_range
            print(f"    {col:12s}: {n_valid:>8,} ({pct:5.1f}%)  in-range: {n_in_range:,}  outliers: {n_outlier:,}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fetch and clean BMRB chemical shift data with provenance tracking.'
    )
    parser.add_argument(
        '--input', default='./data/chemical_shifts.csv',
        help='Path to existing chemical_shifts_pivoted.csv (default: ./data/chemical_shifts.csv)'
    )
    parser.add_argument(
        '--output-dir', default=DATA_DIR,
        help=f'Output directory for cleaned files (default: {DATA_DIR})'
    )
    parser.add_argument(
        '--max-outlier-pct', type=float, default=10.0,
        help='Flag proteins with more than this percentage of outlier shifts (default: 10.0)'
    )
    parser.add_argument(
        '--remove-bad-proteins', action='store_true', default=False,
        help='Remove proteins flagged as having systematic referencing errors'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 00: FETCH AND CLEAN BMRB CHEMICAL SHIFT DATA")
    print("=" * 70)

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    output_shifts = os.path.join(output_dir, 'chemical_shifts.csv')
    output_log = os.path.join(output_dir, 'shift_quality_log.csv')

    # ------------------------------------------------------------------
    # Step 1: Load or download shift data
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading chemical shift data...")

    df = load_existing_shifts(input_path)

    if df is None:
        print(f"  File not found: {input_path}")
        print("  BMRB API download is not implemented for bulk download.")
        print("  Please provide an existing chemical_shifts_pivoted.csv file.")
        print("  You can generate one using the lab_data/ pipeline or download")
        print("  from BMRB manually.")
        sys.exit(1)

    # Ensure bmrb_id is string
    df['bmrb_id'] = df['bmrb_id'].astype(str)

    # Record initial stats
    n_proteins_initial = df['bmrb_id'].nunique()
    n_residues_initial = len(df)

    shift_cols = sorted([c for c in df.columns if c.endswith('_shift')])
    backbone_shift_cols = [c for c in shift_cols if c in SHIFT_RANGES]

    n_total_shift_values_initial = sum(df[c].notna().sum() for c in shift_cols)
    n_backbone_shift_values_initial = sum(df[c].notna().sum() for c in backbone_shift_cols)

    print(f"  Proteins: {n_proteins_initial:,}")
    print(f"  Residues: {n_residues_initial:,}")
    print(f"  Total shift values (all atoms): {n_total_shift_values_initial:,}")
    print(f"  Backbone shift values (6 atoms): {n_backbone_shift_values_initial:,}")
    print(f"  Shift columns: {len(shift_cols)}")

    print_shift_coverage(df, "BEFORE filtering")

    # ------------------------------------------------------------------
    # Step 2: Apply quality filters with provenance
    # ------------------------------------------------------------------
    print("\n[Step 2] Applying data quality filters...")
    log = FilterLog()
    log.add_summary('initial', 'Starting proteins', n_proteins_initial)
    log.add_summary('initial', 'Starting residues', n_residues_initial)
    log.add_summary('initial', 'Starting shift values (all atoms)', n_total_shift_values_initial)

    # 2a: Flag bad proteins (systematic referencing errors)
    from data_quality import flag_bad_proteins
    bad_proteins, log = flag_bad_proteins(df, max_outlier_pct=args.max_outlier_pct, log=log)

    if bad_proteins:
        print(f"\n  Proteins with >{args.max_outlier_pct}% outlier shifts: {len(bad_proteins)}")
        if args.remove_bad_proteins:
            n_before = len(df)
            df = df[~df['bmrb_id'].isin(bad_proteins)].copy()
            n_removed = n_before - len(df)
            log.add_summary('bad_protein_removal', 'Residues removed (from bad proteins)', n_removed)
            print(f"  Removed {len(bad_proteins)} bad proteins ({n_removed:,} residues)")
        else:
            print(f"  (Not removing -- pass --remove-bad-proteins to remove them)")

    # 2b: Filter non-standard residues
    print("\n  Filtering non-standard residues...")
    df, log = filter_standard_residues(df, log=log)

    # 2c: Remove duplicates
    print("\n  Removing duplicate (bmrb_id, residue_id) pairs...")
    df, log = remove_duplicates(df, log=log)

    # 2d: Filter shift outliers (sets out-of-range values to NaN, does NOT remove rows)
    print("\n  Filtering shift outliers (setting out-of-range to NaN)...")
    df, log = filter_shift_outliers(df, log=log)

    # ------------------------------------------------------------------
    # Step 3: Compute final stats
    # ------------------------------------------------------------------
    n_proteins_final = df['bmrb_id'].nunique()
    n_residues_final = len(df)
    n_total_shift_values_final = sum(df[c].notna().sum() for c in shift_cols if c in df.columns)
    n_backbone_shift_values_final = sum(
        df[c].notna().sum() for c in backbone_shift_cols if c in df.columns
    )

    log.add_summary('final', 'Final proteins', n_proteins_final)
    log.add_summary('final', 'Final residues', n_residues_final)
    log.add_summary('final', 'Final shift values (all atoms)', n_total_shift_values_final)
    log.add_summary('final', 'Total rows removed', n_residues_initial - n_residues_final)
    log.add_summary(
        'final',
        'Total backbone shift values set to NaN',
        n_backbone_shift_values_initial - n_backbone_shift_values_final,
    )

    # ------------------------------------------------------------------
    # Step 4: Save outputs
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Saving outputs to {output_dir}/...")

    df.to_csv(output_shifts, index=False)
    print(f"  Saved cleaned shifts: {output_shifts}")
    print(f"    {n_proteins_final:,} proteins, {n_residues_final:,} residues")

    log.save(output_log)

    # ------------------------------------------------------------------
    # Step 5: Print comprehensive summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'':30s}  {'Before':>12s}  {'After':>12s}  {'Change':>12s}")
    print(f"  {'-' * 30}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
    print(f"  {'Proteins':30s}  {n_proteins_initial:>12,}  {n_proteins_final:>12,}  {n_proteins_final - n_proteins_initial:>+12,}")
    print(f"  {'Residues':30s}  {n_residues_initial:>12,}  {n_residues_final:>12,}  {n_residues_final - n_residues_initial:>+12,}")
    print(f"  {'Shift values (all atoms)':30s}  {n_total_shift_values_initial:>12,}  {n_total_shift_values_final:>12,}  {n_total_shift_values_final - n_total_shift_values_initial:>+12,}")
    print(f"  {'Backbone shift values':30s}  {n_backbone_shift_values_initial:>12,}  {n_backbone_shift_values_final:>12,}  {n_backbone_shift_values_final - n_backbone_shift_values_initial:>+12,}")

    if bad_proteins:
        print(f"\n  Bad proteins flagged (>{args.max_outlier_pct}% outlier shifts): {len(bad_proteins)}")
        if not args.remove_bad_proteins:
            print(f"    (kept in output -- use --remove-bad-proteins to exclude)")

    print_shift_coverage(df, "AFTER filtering")

    log.print_report()

    print(f"\nOutputs:")
    print(f"  {output_shifts}")
    print(f"  {output_log}")
    print()


if __name__ == '__main__':
    main()
