"""
Data quality filtering with full provenance tracking.

Every filtering step documents:
- What was removed (specific bmrb_ids, residue_ids, values)
- Why it was removed (which rule triggered)
- Exact counts before and after

Returns a FilterLog object that can be saved to CSV/JSON for audit.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from config import SHIFT_RANGES, AA_3_TO_1, NONSTANDARD_MAP, STANDARD_RESIDUES, MIN_SEQ_SEPARATION


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


def filter_shift_outliers(df, log=None):
    """Apply physical range filters to backbone chemical shifts.

    Sets out-of-range values to NaN (does NOT remove rows).
    Uses ranges from config.SHIFT_RANGES.

    Args:
        df: DataFrame with shift columns
        log: FilterLog instance (created if None)

    Returns:
        (filtered_df, log) -- df with outliers set to NaN, provenance log
    """
    if log is None:
        log = FilterLog()

    df = df.copy()
    total_outliers = 0

    for col, (lo, hi) in SHIFT_RANGES.items():
        if col not in df.columns:
            continue

        valid_mask = df[col].notna()
        below = valid_mask & (df[col] < lo)
        above = valid_mask & (df[col] > hi)
        outlier_mask = below | above
        n_outliers = outlier_mask.sum()

        if n_outliers > 0:
            # Log each outlier
            outlier_rows = df[outlier_mask]
            for _, row in outlier_rows.iterrows():
                log.add(
                    step='shift_outlier_filter',
                    reason=f'{col} outside [{lo}, {hi}] ppm',
                    bmrb_id=row.get('bmrb_id'),
                    residue_id=row.get('residue_id'),
                    column=col,
                    value=row[col],
                    action='set_to_nan',
                )

            df.loc[outlier_mask, col] = np.nan
            total_outliers += n_outliers

            n_below = below.sum()
            n_above = above.sum()
            log.add_summary(
                'shift_outlier_filter',
                f'{col}: {n_below} below {lo}, {n_above} above {hi}',
                n_outliers,
                f'range=[{lo}, {hi}] ppm'
            )

    log.add_summary('shift_outlier_filter', 'TOTAL shift values set to NaN', total_outliers)
    return df, log


def flag_bad_proteins(df, max_outlier_pct=10.0, log=None):
    """Flag proteins with systematically bad shift data.

    A protein is flagged if >max_outlier_pct% of its shifts fall outside
    physical ranges (before filtering). These likely have referencing errors.

    Args:
        df: DataFrame (before outlier filtering)
        max_outlier_pct: Threshold percentage
        log: FilterLog instance

    Returns:
        (set of bad bmrb_ids, log)
    """
    if log is None:
        log = FilterLog()

    bad_proteins = set()

    for pid, prot_df in df.groupby('bmrb_id'):
        total_shifts = 0
        total_outliers = 0

        for col, (lo, hi) in SHIFT_RANGES.items():
            if col not in prot_df.columns:
                continue
            valid = prot_df[col].dropna()
            total_shifts += len(valid)
            total_outliers += ((valid < lo) | (valid > hi)).sum()

        if total_shifts > 0:
            pct = 100.0 * total_outliers / total_shifts
            if pct > max_outlier_pct:
                bad_proteins.add(pid)
                log.add(
                    step='bad_protein_flag',
                    reason=f'{pct:.1f}% outliers (>{max_outlier_pct}%)',
                    bmrb_id=pid,
                    action='flagged',
                )

    log.add_summary('bad_protein_flag', f'Proteins with >{max_outlier_pct}% outlier shifts', len(bad_proteins))
    return bad_proteins, log


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


def filter_standard_residues(df, log=None):
    """Keep only standard 20 amino acids, mapping non-standard where possible.

    Maps: SEC->CYS, MSE->MET, PYL->LYS, ASX->ASN, GLX->GLN
    Removes: anything not in STANDARD_RESIDUES after mapping

    Returns:
        (filtered_df, log)
    """
    if log is None:
        log = FilterLog()

    n_before = len(df)
    df = df.copy()

    # Map non-standard residues
    standard_set = set(STANDARD_RESIDUES) - {'UNK'}
    mapped_count = 0

    for old_name, new_name in NONSTANDARD_MAP.items():
        if old_name == new_name:
            continue
        mask = df['residue_code'].str.upper() == old_name
        n_mapped = mask.sum()
        if n_mapped > 0:
            mapped_count += n_mapped
            log.add_summary(
                'residue_mapping',
                f'Mapped {old_name} -> {new_name}',
                n_mapped,
            )
            df.loc[mask, 'residue_code'] = new_name

    # Remove residues not in standard set
    valid = df['residue_code'].str.upper().isin(standard_set)
    removed = df[~valid]
    n_removed = len(removed)

    if n_removed > 0:
        # Log what was removed (by residue type)
        removed_counts = removed['residue_code'].value_counts()
        for res_code, count in removed_counts.items():
            log.add_summary(
                'nonstandard_residue_filter',
                f'Removed non-standard residue: {res_code}',
                count,
            )

        # Log individual removals (cap at 1000 to avoid huge logs)
        for _, row in removed.head(1000).iterrows():
            log.add(
                step='nonstandard_residue_filter',
                reason=f'non-standard residue: {row["residue_code"]}',
                bmrb_id=row.get('bmrb_id'),
                residue_id=row.get('residue_id'),
                column='residue_code',
                value=row['residue_code'],
                action='removed',
            )

        df = df[valid].copy()

    log.add_summary('nonstandard_residue_filter', 'TOTAL non-standard residues removed', n_removed,
                    f'{n_before} -> {len(df)} rows, {mapped_count} mapped')
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
    print(f"\n  Backbone shift coverage:")
    for col, (lo, hi) in SHIFT_RANGES.items():
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct = 100.0 * n_valid / n_residues if n_residues > 0 else 0
            n_outlier = ((df[col].dropna() < lo) | (df[col].dropna() > hi)).sum()
            print(f"    {col:12s}: {n_valid:>8,} ({pct:5.1f}%)  outliers: {n_outlier}")

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


def run_full_quality_pipeline(df, remove_bad_proteins=True, max_outlier_pct=10.0):
    """Run all quality filters in sequence with full provenance.

    Steps:
    1. Flag bad proteins (systematic referencing errors)
    2. Remove bad proteins (optional)
    3. Filter non-standard residues (map or remove)
    4. Remove duplicates
    5. Filter shift outliers (set to NaN)
    6. Validate spatial neighbors

    Returns:
        (cleaned_df, log)
    """
    log = FilterLog()

    print("Running full data quality pipeline...")
    n_initial = len(df)
    log.add_summary('initial', 'Starting residues', n_initial)
    log.add_summary('initial', 'Starting proteins', df['bmrb_id'].nunique() if 'bmrb_id' in df.columns else 0)

    # Step 1-2: Flag and optionally remove bad proteins
    bad_proteins, log = flag_bad_proteins(df, max_outlier_pct=max_outlier_pct, log=log)
    if remove_bad_proteins and bad_proteins:
        n_before = len(df)
        df = df[~df['bmrb_id'].isin(bad_proteins)].copy()
        log.add_summary('bad_protein_removal', 'Residues removed (from bad proteins)', n_before - len(df))

    # Step 3: Filter non-standard residues
    df, log = filter_standard_residues(df, log=log)

    # Step 4: Remove duplicates
    df, log = remove_duplicates(df, log=log)

    # Step 5: Filter shift outliers
    df, log = filter_shift_outliers(df, log=log)

    # Step 6: Validate spatial neighbors
    df, log = validate_spatial_neighbors(df, log=log)

    log.add_summary('final', 'Final residues', len(df))
    log.add_summary('final', 'Final proteins', df['bmrb_id'].nunique() if 'bmrb_id' in df.columns else 0)
    log.add_summary('final', 'Total rows removed', n_initial - len(df))

    log.print_report()

    return df, log
