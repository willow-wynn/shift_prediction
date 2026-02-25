"""
Data quality filtering with full provenance tracking.

Every filtering step documents:
- What was removed (specific bmrb_ids, residue_ids, values)
- Why it was removed (which rule triggered)
- Exact counts before and after

Returns a FilterLog object that can be saved to CSV/JSON for audit.

Note: The old range-based outlier filtering (SHIFT_RANGES) and nonstandard
residue mapping (NONSTANDARD_MAP) have been removed. Quality filtering is
now handled in 01_build_datasets.py with per-(atom_type, secondary_structure)
outlier detection and whole-protein nonstandard residue exclusion.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from config import AA_3_TO_1, STANDARD_RESIDUES, MIN_SEQ_SEPARATION


class FilterLog:
    """Accumulates filtering decisions for full provenance tracking."""

    def __init__(self):
        self.entries = []  # List of {step, reason, bmrb_id, residue_id, column, value, action}
        self.summaries = []  # List of {step, reason, count, details}

    def add(self, step, reason, bmrb_id=None, residue_id=None, column=None, value=None, action='removed'):
        self.entries.append({
            'step': step,
            'reason': reason,
            'bmrb_id': bmrb_id,
            'residue_id': residue_id,
            'column': column,
            'value': value,
            'action': action,
        })

    def add_summary(self, step, reason, count, details=None):
        self.summaries.append({
            'step': step,
            'reason': reason,
            'count': count,
            'details': details or '',
        })
        print(f"  [{step}] {reason}: {count:,} items")

    def to_dataframe(self):
        return pd.DataFrame(self.entries)

    def save(self, path):
        """Save provenance log to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        # Also save summaries
        summary_df = pd.DataFrame(self.summaries)
        summary_path = path.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"  Provenance log saved: {path} ({len(df):,} entries)")
        print(f"  Summary saved: {summary_path} ({len(summary_df)} steps)")

    def print_report(self):
        """Print human-readable filtering report."""
        print("\n" + "=" * 70)
        print("DATA QUALITY FILTERING REPORT")
        print("=" * 70)
        for s in self.summaries:
            print(f"  {s['step']:30s} | {s['reason']:40s} | {s['count']:>8,}")
            if s['details']:
                print(f"  {'':30s} | {s['details']}")
        print("=" * 70)


def remove_duplicates(df, log=None):
    """Remove duplicate (bmrb_id, residue_id) pairs, keeping first occurrence.

    Returns:
        (deduplicated_df, log)
    """
    if log is None:
        log = FilterLog()

    n_before = len(df)
    dup_mask = df.duplicated(subset=['bmrb_id', 'residue_id'], keep='first')
    n_dups = dup_mask.sum()

    if n_dups > 0:
        dup_rows = df[dup_mask]
        for _, row in dup_rows.iterrows():
            log.add(
                step='deduplication',
                reason='duplicate (bmrb_id, residue_id) pair',
                bmrb_id=row.get('bmrb_id'),
                residue_id=row.get('residue_id'),
                action='removed',
            )
        df = df[~dup_mask].copy()

    log.add_summary('deduplication', 'Duplicate rows removed', n_dups, f'{n_before} -> {len(df)} rows')
    return df, log


def validate_spatial_neighbors(df, min_sep=None, log=None):
    """Validate spatial neighbor columns: enforce |seq_sep| >= min_sep.

    Sets invalid neighbors to -1 (missing).

    Returns:
        (validated_df, log)
    """
    if log is None:
        log = FilterLog()
    if min_sep is None:
        min_sep = MIN_SEQ_SEPARATION

    df = df.copy()
    total_fixed = 0

    for k in range(5):
        sep_col = f'spatial_neighbor_{k}_seq_sep'
        id_col = f'spatial_neighbor_{k}_id'
        dist_col = f'spatial_neighbor_{k}_dist'

        if sep_col not in df.columns:
            continue

        bad = df[sep_col].notna() & (df[sep_col].abs() < min_sep) & (df[sep_col] != -1)
        n_bad = bad.sum()

        if n_bad > 0:
            if id_col in df.columns:
                df.loc[bad, id_col] = -1
            if dist_col in df.columns:
                df.loc[bad, dist_col] = -1.0
            df.loc[bad, sep_col] = -1
            total_fixed += n_bad

    log.add_summary('spatial_neighbor_validation', f'Neighbors with |seq_sep| < {min_sep} invalidated', total_fixed)
    return df, log


def report_quality(df, log=None):
    """Print comprehensive quality summary of a dataset.

    Returns:
        Dict of quality metrics
    """
    if log is None:
        log = FilterLog()

    n_proteins = df['bmrb_id'].nunique() if 'bmrb_id' in df.columns else 0
    n_residues = len(df)

    print(f"\n{'=' * 60}")
    print(f"DATASET QUALITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Proteins: {n_proteins:,}")
    print(f"  Residues: {n_residues:,}")

    # Shift coverage
    shift_cols = [c for c in df.columns if c.endswith('_shift')]
    if shift_cols:
        print(f"\n  Shift coverage:")
        for col in sorted(shift_cols):
            n_valid = df[col].notna().sum()
            pct = 100.0 * n_valid / n_residues if n_residues > 0 else 0
            print(f"    {col:20s}: {n_valid:>8,} ({pct:5.1f}%)")

    # Distance coverage
    dist_cols = [c for c in df.columns if c.startswith('dist_')]
    if dist_cols:
        nan_fracs = [df[c].isna().mean() for c in dist_cols]
        print(f"\n  Distance columns: {len(dist_cols)}")
        print(f"    Mean NaN fraction: {np.mean(nan_fracs):.1%}")
        print(f"    Columns with <10% NaN: {sum(1 for f in nan_fracs if f < 0.1)}")
        print(f"    Columns with <50% NaN: {sum(1 for f in nan_fracs if f < 0.5)}")

    # Residue composition
    if 'residue_code' in df.columns:
        print(f"\n  Residue types: {df['residue_code'].nunique()}")
        non_standard = set(df['residue_code'].unique()) - set(STANDARD_RESIDUES)
        if non_standard:
            print(f"    Non-standard: {non_standard}")

    print(f"{'=' * 60}\n")

    return {
        'n_proteins': n_proteins,
        'n_residues': n_residues,
        'n_distance_cols': len(dist_cols) if dist_cols else 0,
    }
