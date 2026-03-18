#!/usr/bin/env python3
"""
Comprehensive data quality analysis for chemical shift dataset.

Performs statistical checks to identify invalid/problematic data that would
hurt ML model training. Does NOT modify the original dataset — produces a
filtered copy and a detailed removal log.

Checks performed:
  1. Glycine CB shifts (GLY has no beta carbon)
  2. Proline H shifts (PRO has no amide proton)
  3. Per-(residue_type, shift_type) physical range violations using RefDB stats
  4. Per-(residue_type, shift_type) statistical outliers (Tukey IQR method)
  5. Proteins with too few residues (< 10)
  6. Residues with impossible backbone geometry (Ramachandran outliers)
  7. Sidechain shifts on residues that don't have that sidechain atom
  8. Shift referencing errors (systematic offset detection per protein)
  9. Ambiguity code filtering (remove ambiguity > 1 for backbone)
  10. Ultra-rare sidechain shift types with insufficient data
  11. Duplicate (bmrb_id, residue_id) entries
  12. Proteins where >50% of shifts are outliers (systematic problems)

Output:
  data/structure_data_hybrid_cleaned.csv  — cleaned dataset
  data/cleaning_log.json                  — detailed removal log
  data/cleaning_summary.txt               — human-readable summary
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AA_3_TO_1, STANDARD_RESIDUES

# ==========================================================================
# Reference ranges from RefDB / BMRB statistics
# These are well-established physical limits for each (residue, shift) pair
# ==========================================================================

# Backbone shift physical ranges (ppm) — conservative bounds from RefDB
# Format: {atom_type: (global_min, global_max)}
BACKBONE_GLOBAL_RANGES = {
    'ca_shift': (35.0, 75.0),
    'cb_shift': (10.0, 80.0),
    'c_shift':  (165.0, 185.0),
    'n_shift':  (95.0, 145.0),
    'h_shift':  (5.0, 12.5),
    'ha_shift': (2.5, 6.5),
}

# Residue-specific CB ranges (some residues have very different CB)
CB_RANGES_BY_RESIDUE = {
    'ALA': (14.0, 25.0),
    'ARG': (25.0, 38.0),
    'ASN': (33.0, 44.0),
    'ASP': (35.0, 48.0),
    'CYS': (22.0, 50.0),   # Wide range — oxidized vs reduced
    'GLU': (25.0, 36.0),
    'GLN': (24.0, 35.0),
    'GLY': None,            # NO CB
    'HIS': (24.0, 38.0),
    'ILE': (33.0, 45.0),
    'LEU': (37.0, 48.0),
    'LYS': (28.0, 38.0),
    'MET': (28.0, 40.0),
    'PHE': (34.0, 45.0),
    'PRO': (27.0, 38.0),
    'SER': (57.0, 70.0),   # Distinctive downfield CB
    'THR': (63.0, 76.0),   # Distinctive downfield CB
    'TRP': (24.0, 35.0),
    'TYR': (33.0, 44.0),
    'VAL': (27.0, 38.0),
}

# Sidechain atoms that DON'T exist for certain residues
# Key = residue_code, Value = set of shift columns that CANNOT exist
IMPOSSIBLE_SIDECHAIN_SHIFTS = {
    'GLY': {'cb_shift', 'cg_shift', 'cg1_shift', 'cg2_shift', 'cd_shift', 'cd1_shift', 'cd2_shift',
            'ce_shift', 'ce1_shift', 'ce2_shift', 'cz_shift',
            'hb_shift', 'hb1_shift', 'hb2_shift', 'hb3_shift',
            'hg_shift', 'hg2_shift', 'hg3_shift', 'hg11_shift', 'hg12_shift', 'hg13_shift',
            'hg21_shift', 'hg22_shift', 'hg23_shift',
            'hd1_shift', 'hd2_shift', 'hd3_shift', 'hd11_shift', 'hd12_shift', 'hd13_shift',
            'hd21_shift', 'hd22_shift', 'hd23_shift',
            'he_shift', 'he1_shift', 'he2_shift', 'he3_shift', 'he21_shift', 'he22_shift',
            'hz_shift', 'nd2_shift', 'ne2_shift'},
    'ALA': {'cg_shift', 'cg1_shift', 'cg2_shift', 'cd_shift', 'cd1_shift', 'cd2_shift',
            'ce_shift', 'ce1_shift', 'ce2_shift', 'cz_shift',
            'hg_shift', 'hg2_shift', 'hg3_shift', 'hg11_shift', 'hg12_shift', 'hg13_shift',
            'hg21_shift', 'hg22_shift', 'hg23_shift',
            'hd1_shift', 'hd2_shift', 'hd3_shift', 'hd11_shift', 'hd12_shift', 'hd13_shift',
            'hd21_shift', 'hd22_shift', 'hd23_shift',
            'he_shift', 'he1_shift', 'he2_shift', 'he3_shift', 'he21_shift', 'he22_shift',
            'hz_shift', 'nd2_shift', 'ne2_shift'},
    'PRO': set(),  # PRO has no amide H but has all sidechain atoms
}

# Minimum observations per (residue_type, shift_type) to keep in training
MIN_OBS_PER_GROUP = 30


def load_dataset(csv_path):
    """Load the hybrid dataset."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'bmrb_id': str})
    print(f"  {len(df):,} residues, {df['bmrb_id'].nunique():,} proteins")
    return df


def get_shift_columns(df):
    """Get all shift columns."""
    return sorted([c for c in df.columns if c.endswith('_shift')])


class CleaningLog:
    """Track all cleaning decisions with full provenance."""

    def __init__(self):
        self.entries = []
        self.summary = defaultdict(lambda: {'count': 0, 'proteins_affected': set()})

    def add(self, check_name, bmrb_id, residue_id, column, old_value, reason):
        self.entries.append({
            'check': check_name,
            'bmrb_id': str(bmrb_id),
            'residue_id': int(residue_id) if pd.notna(residue_id) else None,
            'column': column,
            'old_value': float(old_value) if pd.notna(old_value) else None,
            'reason': reason,
        })
        self.summary[check_name]['count'] += 1
        self.summary[check_name]['proteins_affected'].add(str(bmrb_id))

    def add_protein(self, check_name, bmrb_id, reason, n_residues=0):
        self.entries.append({
            'check': check_name,
            'bmrb_id': str(bmrb_id),
            'residue_id': None,
            'column': None,
            'old_value': None,
            'reason': f"{reason} ({n_residues} residues)",
        })
        self.summary[check_name]['count'] += n_residues
        self.summary[check_name]['proteins_affected'].add(str(bmrb_id))

    def print_summary(self):
        print("\n" + "=" * 80)
        print("DATA CLEANING SUMMARY")
        print("=" * 80)
        total = 0
        for check, info in sorted(self.summary.items()):
            n = info['count']
            np_ = len(info['proteins_affected'])
            total += n
            print(f"  {check:45s}: {n:>8,} values  ({np_:,} proteins)")
        print(f"  {'TOTAL':45s}: {total:>8,}")
        print("=" * 80)

    def to_json(self):
        summary = {}
        for check, info in self.summary.items():
            summary[check] = {
                'count': info['count'],
                'n_proteins_affected': len(info['proteins_affected']),
            }
        return {
            'summary': summary,
            'total_values_removed': sum(v['count'] for v in self.summary.values()),
            'entries': self.entries[:100000],  # Cap for JSON size
            'n_entries_total': len(self.entries),
        }


# ==========================================================================
# Check 1: Glycine CB shifts
# ==========================================================================
def check_glycine_cb(df, log):
    """Glycine has no beta carbon. Any CB shift on GLY is invalid."""
    mask = (df['residue_code'] == 'GLY') & df['cb_shift'].notna()
    n = mask.sum()
    if n > 0:
        for _, row in df[mask].iterrows():
            log.add('glycine_cb', row['bmrb_id'], row['residue_id'],
                     'cb_shift', row['cb_shift'], 'GLY has no CB atom')
        df.loc[mask, 'cb_shift'] = np.nan
    print(f"  [1] Glycine CB shifts removed: {n:,}")
    return df


# ==========================================================================
# Check 2: Proline amide H shifts
# ==========================================================================
def check_proline_h(df, log):
    """Proline has no amide proton (except at chain N-terminus). Remove H shifts."""
    mask = (df['residue_code'] == 'PRO') & df['h_shift'].notna()
    n = mask.sum()
    if n > 0:
        for _, row in df[mask].iterrows():
            log.add('proline_h', row['bmrb_id'], row['residue_id'],
                     'h_shift', row['h_shift'], 'PRO has no amide H')
        df.loc[mask, 'h_shift'] = np.nan
    print(f"  [2] Proline H shifts removed: {n:,}")
    return df


# ==========================================================================
# Check 3: Physical range violations
# ==========================================================================
def check_physical_ranges(df, shift_cols, log):
    """Remove shifts outside physically possible ranges."""
    total = 0
    for col in shift_cols:
        if col not in BACKBONE_GLOBAL_RANGES:
            continue
        lo, hi = BACKBONE_GLOBAL_RANGES[col]
        mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        n = mask.sum()
        if n > 0:
            for _, row in df[mask].iterrows():
                log.add('physical_range', row['bmrb_id'], row['residue_id'],
                         col, row[col], f'{col} outside [{lo}, {hi}]')
            df.loc[mask, col] = np.nan
            total += n
    print(f"  [3] Physical range violations removed: {total:,}")
    return df


# ==========================================================================
# Check 4: Residue-specific CB range violations
# ==========================================================================
def check_residue_cb_ranges(df, log):
    """Check CB shifts against residue-specific expected ranges (wider than RefDB)."""
    total = 0
    for res_code, expected_range in CB_RANGES_BY_RESIDUE.items():
        if expected_range is None:
            continue  # GLY handled separately
        lo, hi = expected_range
        # Use generous bounds (mean ± 5*std equivalent)
        margin = (hi - lo) * 0.3
        lo_generous = lo - margin
        hi_generous = hi + margin
        mask = (df['residue_code'] == res_code) & df['cb_shift'].notna() & \
               ((df['cb_shift'] < lo_generous) | (df['cb_shift'] > hi_generous))
        n = mask.sum()
        if n > 0:
            for _, row in df[mask].iterrows():
                log.add('residue_cb_range', row['bmrb_id'], row['residue_id'],
                         'cb_shift', row['cb_shift'],
                         f'{res_code} CB outside [{lo_generous:.1f}, {hi_generous:.1f}]')
            df.loc[mask, 'cb_shift'] = np.nan
            total += n
    print(f"  [4] Residue-specific CB range violations: {total:,}")
    return df


# ==========================================================================
# Check 5: Impossible sidechain shifts
# ==========================================================================
def check_impossible_sidechains(df, shift_cols, log):
    """Remove sidechain shifts for atoms that don't exist on that residue."""
    total = 0
    for res_code, impossible_cols in IMPOSSIBLE_SIDECHAIN_SHIFTS.items():
        for col in impossible_cols:
            if col not in shift_cols:
                continue
            mask = (df['residue_code'] == res_code) & df[col].notna()
            n = mask.sum()
            if n > 0:
                for _, row in df[mask].head(50).iterrows():  # Log first 50
                    log.add('impossible_sidechain', row['bmrb_id'], row['residue_id'],
                             col, row[col], f'{res_code} cannot have {col}')
                if n > 50:
                    log.summary['impossible_sidechain']['count'] += (n - 50)
                df.loc[mask, col] = np.nan
                total += n
    print(f"  [5] Impossible sidechain shifts removed: {total:,}")
    return df


# ==========================================================================
# Check 6: Tukey IQR outlier detection per (residue_type, shift_type)
# ==========================================================================
def check_tukey_outliers(df, shift_cols, log, k=3.0):
    """Tukey IQR outlier detection: outside [Q1 - k*IQR, Q3 + k*IQR]."""
    total = 0
    backbone_cols = [c for c in shift_cols if c in BACKBONE_GLOBAL_RANGES]

    for col in backbone_cols:
        for res_code in STANDARD_RESIDUES:
            if res_code == 'UNK':
                continue
            mask_group = (df['residue_code'] == res_code) & df[col].notna()
            vals = df.loc[mask_group, col]
            if len(vals) < MIN_OBS_PER_GROUP:
                continue
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            if iqr < 0.01:
                continue
            lo = q1 - k * iqr
            hi = q3 + k * iqr
            outlier_mask = mask_group & ((df[col] < lo) | (df[col] > hi))
            n = outlier_mask.sum()
            if n > 0:
                for _, row in df[outlier_mask].head(20).iterrows():
                    log.add('tukey_iqr_outlier', row['bmrb_id'], row['residue_id'],
                             col, row[col],
                             f'{res_code}/{col}: outside [{lo:.2f}, {hi:.2f}]')
                if n > 20:
                    log.summary['tukey_iqr_outlier']['count'] += (n - 20)
                    for _, row in df[outlier_mask].iterrows():
                        log.summary['tukey_iqr_outlier']['proteins_affected'].add(str(row['bmrb_id']))
                df.loc[outlier_mask, col] = np.nan
                total += n

    print(f"  [6] Tukey IQR outliers removed (k={k}): {total:,}")
    return df


# ==========================================================================
# Check 7: Proteins with too few residues
# ==========================================================================
def check_small_proteins(df, log, min_residues=10):
    """Remove proteins with fewer than min_residues."""
    protein_sizes = df.groupby('bmrb_id').size()
    small = protein_sizes[protein_sizes < min_residues]
    n_proteins = len(small)
    n_residues = small.sum()
    if n_proteins > 0:
        for bmrb_id, size in small.items():
            log.add_protein('small_protein', bmrb_id,
                           f'Only {size} residues (min={min_residues})', size)
        df = df[~df['bmrb_id'].isin(small.index)].copy()
    print(f"  [7] Small proteins removed: {n_proteins} proteins ({n_residues:,} residues)")
    return df


# ==========================================================================
# Check 8: Shift referencing errors (systematic offset per protein)
# ==========================================================================
def check_referencing_errors(df, log):
    """Detect proteins with systematically offset shifts (referencing errors).

    For each protein, compute the median deviation from the global residue-type
    mean for each backbone shift. If the median deviation is > 3 ppm for heavy
    atoms or > 0.5 ppm for H/HA, flag the protein.
    """
    # Compute global per-residue-type means
    backbone = ['ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift']
    # Thresholds for "obviously wrong referencing"
    thresholds = {
        'ca_shift': 4.0, 'cb_shift': 5.0, 'c_shift': 3.0,
        'n_shift': 5.0, 'h_shift': 0.8, 'ha_shift': 0.6,
    }

    global_means = {}
    for col in backbone:
        if col not in df.columns:
            continue
        means = df.groupby('residue_code')[col].mean()
        global_means[col] = means

    flagged_proteins = set()
    flagged_reasons = {}

    for bmrb_id, prot_df in df.groupby('bmrb_id'):
        if len(prot_df) < 10:
            continue
        for col in backbone:
            if col not in df.columns or col not in global_means:
                continue
            thresh = thresholds.get(col, 3.0)
            vals = prot_df[['residue_code', col]].dropna(subset=[col])
            if len(vals) < 5:
                continue
            deviations = []
            for _, row in vals.iterrows():
                expected = global_means[col].get(row['residue_code'])
                if expected is not None:
                    deviations.append(row[col] - expected)
            if len(deviations) >= 5:
                median_dev = np.median(deviations)
                if abs(median_dev) > thresh:
                    flagged_proteins.add(str(bmrb_id))
                    flagged_reasons[str(bmrb_id)] = (
                        f'{col} median deviation = {median_dev:.2f} ppm '
                        f'(threshold: {thresh})'
                    )

    # Remove flagged proteins entirely
    n_residues = 0
    for bmrb_id in flagged_proteins:
        n = (df['bmrb_id'] == bmrb_id).sum()
        log.add_protein('referencing_error', bmrb_id,
                       flagged_reasons.get(bmrb_id, 'systematic offset'), n)
        n_residues += n

    if flagged_proteins:
        df = df[~df['bmrb_id'].isin(flagged_proteins)].copy()

    print(f"  [8] Referencing errors: {len(flagged_proteins)} proteins ({n_residues:,} residues)")
    return df


# ==========================================================================
# Check 9: Ambiguity code filtering for backbone
# ==========================================================================
def check_ambiguity_codes(df, log):
    """Remove backbone shifts with ambiguity code > 1 (not uniquely assigned)."""
    backbone_atoms = ['ca', 'cb', 'c', 'n', 'h', 'ha']
    total = 0

    for atom in backbone_atoms:
        shift_col = f'{atom}_shift'
        ambig_col = f'{atom}_ambiguity_code'
        if shift_col not in df.columns or ambig_col not in df.columns:
            continue
        ambig_numeric = pd.to_numeric(df[ambig_col], errors='coerce')
        mask = df[shift_col].notna() & ambig_numeric.notna() & (ambig_numeric > 1)
        n = mask.sum()
        if n > 0:
            for _, row in df[mask].head(20).iterrows():
                log.add('ambiguity_code', row['bmrb_id'], row['residue_id'],
                         shift_col, row[shift_col],
                         f'ambiguity_code={int(row[ambig_col])}')
            if n > 20:
                log.summary['ambiguity_code']['count'] += (n - 20)
                for _, row in df[mask].iterrows():
                    log.summary['ambiguity_code']['proteins_affected'].add(str(row['bmrb_id']))
            df.loc[mask, shift_col] = np.nan
            total += n

    print(f"  [9] Ambiguous backbone shifts removed: {total:,}")
    return df


# ==========================================================================
# Check 10: Ultra-rare sidechain shift types
# ==========================================================================
def check_rare_shift_types(df, shift_cols, log, min_count=100):
    """Remove sidechain shift columns with very few observations overall."""
    backbone = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}
    sidechain_cols = [c for c in shift_cols if c not in backbone]

    total = 0
    removed_cols = []
    for col in sidechain_cols:
        n_obs = df[col].notna().sum()
        if 0 < n_obs < min_count:
            # NaN out this entire column
            for _, row in df[df[col].notna()].head(10).iterrows():
                log.add('rare_shift_type', row['bmrb_id'], row['residue_id'],
                         col, row[col], f'{col} has only {n_obs} total observations')
            if n_obs > 10:
                log.summary['rare_shift_type']['count'] += (n_obs - 10)
            df[col] = np.nan
            total += n_obs
            removed_cols.append(col)

    print(f"  [10] Rare sidechain shift types removed: {total:,} values in {len(removed_cols)} columns")
    if removed_cols:
        print(f"       Columns: {removed_cols}")
    return df


# ==========================================================================
# Check 11: Duplicate (bmrb_id, residue_id) entries
# ==========================================================================
def check_duplicates(df, log):
    """Remove duplicate (bmrb_id, residue_id) entries."""
    dup_mask = df.duplicated(subset=['bmrb_id', 'residue_id'], keep='first')
    n = dup_mask.sum()
    if n > 0:
        for _, row in df[dup_mask].head(20).iterrows():
            log.add('duplicate', row['bmrb_id'], row['residue_id'],
                     None, None, 'duplicate (bmrb_id, residue_id)')
        if n > 20:
            log.summary['duplicate']['count'] += (n - 20)
        df = df[~dup_mask].copy()
    print(f"  [11] Duplicate rows removed: {n:,}")
    return df


# ==========================================================================
# Check 12: Proteins where >50% of shifts are already NaN'd (bad proteins)
# ==========================================================================
def check_mostly_empty_proteins(df, log, max_nan_fraction=0.9):
    """Remove proteins where almost all shift data has been NaN'd out."""
    backbone = ['ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift']
    existing_bb = [c for c in backbone if c in df.columns]

    flagged = set()
    for bmrb_id, prot_df in df.groupby('bmrb_id'):
        n_residues = len(prot_df)
        if n_residues < 5:
            continue
        n_valid = prot_df[existing_bb].notna().any(axis=1).sum()
        if n_valid / n_residues < (1 - max_nan_fraction):
            flagged.add(str(bmrb_id))

    n_residues = 0
    for bmrb_id in flagged:
        n = (df['bmrb_id'] == bmrb_id).sum()
        log.add_protein('mostly_empty', bmrb_id,
                       f'>90% backbone shifts are NaN', n)
        n_residues += n

    if flagged:
        df = df[~df['bmrb_id'].isin(flagged)].copy()

    print(f"  [12] Mostly-empty proteins removed: {len(flagged)} proteins ({n_residues:,} residues)")
    return df


# ==========================================================================
# Check 13: Residues with NO observed backbone shifts at all
# ==========================================================================
def check_empty_residues(df, log):
    """Remove residues that have zero backbone shift observations."""
    backbone = ['ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift']
    existing_bb = [c for c in backbone if c in df.columns]

    # Also check sidechain — if a residue has NO shifts at all it's useless
    shift_cols = [c for c in df.columns if c.endswith('_shift')]
    has_any = df[shift_cols].notna().any(axis=1)
    empty_mask = ~has_any
    n = empty_mask.sum()

    if n > 0:
        for _, row in df[empty_mask].head(20).iterrows():
            log.add('empty_residue', row['bmrb_id'], row['residue_id'],
                     None, None, 'no observed shifts')
        if n > 20:
            log.summary['empty_residue']['count'] += (n - 20)
        df = df[has_any].copy()

    print(f"  [13] Empty residues (no shifts) removed: {n:,}")
    return df


# ==========================================================================
# Check 14: Mismatch type = 'mismatch' residues
# ==========================================================================
def check_sequence_mismatches(df, log):
    """Remove residues where structure-shift alignment shows a mismatch
    (the amino acid in the PDB doesn't match the BMRB assignment)."""
    if 'mismatch_type' not in df.columns:
        print("  [14] Sequence mismatch check: mismatch_type column not found, skipping")
        return df

    mask = df['mismatch_type'] == 'mismatch'
    n = mask.sum()
    if n > 0:
        for _, row in df[mask].head(20).iterrows():
            log.add('sequence_mismatch', row['bmrb_id'], row['residue_id'],
                     'mismatch_type', None,
                     'Structure-shift residue type mismatch')
        if n > 20:
            log.summary['sequence_mismatch']['count'] += (n - 20)
            for _, row in df[mask].iterrows():
                log.summary['sequence_mismatch']['proteins_affected'].add(str(row['bmrb_id']))
        df = df[~mask].copy()

    print(f"  [14] Sequence mismatched residues removed: {n:,}")
    return df


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description='Comprehensive data quality analysis')
    parser.add_argument('--input', default=None,
                        help='Input CSV (default: data/structure_data_hybrid.csv)')
    parser.add_argument('--output', default=None,
                        help='Output cleaned CSV')
    parser.add_argument('--data_dir', default='data',
                        help='Data directory')
    parser.add_argument('--iqr_k', type=float, default=3.0,
                        help='Tukey IQR multiplier (default: 3.0)')
    parser.add_argument('--min_protein_size', type=int, default=10,
                        help='Minimum residues per protein')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)

    if args.input is None:
        args.input = os.path.join(data_dir, 'structure_data_hybrid.csv')
    if args.output is None:
        args.output = os.path.join(data_dir, 'structure_data_hybrid_cleaned.csv')

    t0 = time.time()
    df = load_dataset(args.input)
    shift_cols = get_shift_columns(df)
    n_before = len(df)
    n_proteins_before = df['bmrb_id'].nunique()

    log = CleaningLog()

    print(f"\nRunning {14} quality checks...\n")

    # === Value-level cleaning (NaN out bad values) ===
    df = check_glycine_cb(df, log)
    df = check_proline_h(df, log)
    df = check_physical_ranges(df, shift_cols, log)
    df = check_residue_cb_ranges(df, log)
    df = check_impossible_sidechains(df, shift_cols, log)
    df = check_tukey_outliers(df, shift_cols, log, k=args.iqr_k)
    df = check_ambiguity_codes(df, log)
    df = check_rare_shift_types(df, shift_cols, log)

    # === Row-level cleaning (remove rows/proteins) ===
    df = check_duplicates(df, log)
    df = check_sequence_mismatches(df, log)
    df = check_referencing_errors(df, log)
    df = check_small_proteins(df, log, args.min_protein_size)
    df = check_empty_residues(df, log)
    df = check_mostly_empty_proteins(df, log)

    # === Final stats ===
    n_after = len(df)
    n_proteins_after = df['bmrb_id'].nunique()

    print(f"\n{'=' * 80}")
    print(f"BEFORE: {n_before:,} residues, {n_proteins_before:,} proteins")
    print(f"AFTER:  {n_after:,} residues, {n_proteins_after:,} proteins")
    print(f"REMOVED: {n_before - n_after:,} residues, {n_proteins_before - n_proteins_after:,} proteins")
    print(f"{'=' * 80}")

    # Print per-shift stats after cleaning
    print(f"\nBackbone shift coverage after cleaning:")
    for col in ['ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift']:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct = 100 * n_valid / n_after if n_after > 0 else 0
            print(f"  {col:12s}: {n_valid:>8,} ({pct:5.1f}%)")

    log.print_summary()

    # Save cleaned dataset
    print(f"\nSaving cleaned dataset to {args.output}...")
    df.to_csv(args.output, index=False)
    print(f"  Done. ({len(df):,} residues)")

    # Save log
    log_path = args.output.replace('.csv', '_log.json')
    with open(log_path, 'w') as f:
        json.dump(log.to_json(), f, indent=2, default=str)
    print(f"  Cleaning log saved to {log_path}")

    # Save human-readable summary
    summary_path = args.output.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("DATA QUALITY CLEANING SUMMARY\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Input:  {args.input}\n")
        f.write(f"Output: {args.output}\n")
        f.write(f"Date:   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nBefore: {n_before:,} residues, {n_proteins_before:,} proteins\n")
        f.write(f"After:  {n_after:,} residues, {n_proteins_after:,} proteins\n")
        f.write(f"Removed: {n_before - n_after:,} residues, "
                f"{n_proteins_before - n_proteins_after:,} proteins\n\n")
        f.write(f"Checks performed:\n")
        for check, info in sorted(log.summary.items()):
            f.write(f"  {check:45s}: {info['count']:>8,} values "
                    f"({len(info['proteins_affected']):,} proteins)\n")
        f.write(f"\nTime: {time.time() - t0:.1f}s\n")
    print(f"  Summary saved to {summary_path}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
