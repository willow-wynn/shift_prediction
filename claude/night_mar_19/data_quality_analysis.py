#!/usr/bin/env python3
"""
Data Quality Analysis: Does cleaner but less data help?

Analyzes:
1. Data quality distributions across hybrid dataset
2. Shift coverage per protein (how complete are the observations?)
3. Alphafold vs experimental structure comparison
4. Outlier analysis
5. Per-fold data consistency
6. Retrieval quality distributions
7. Data volume vs quality tradeoffs

Outputs charts and a summary to claude/night_mar_19/
"""

import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(PLOT_DIR, exist_ok=True)

BACKBONE_SHIFTS = ['ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift']


def load_light_data(data_dir='data'):
    """Load lightweight columns from the dataset."""
    data_file = os.path.join(data_dir, 'structure_data_hybrid.csv')

    # Read header
    all_cols = pd.read_csv(data_file, nrows=0).columns.tolist()
    shift_cols = sorted([c for c in all_cols if c.endswith('_shift')])

    # Load only what we need
    light_cols = ['bmrb_id', 'residue_id', 'residue_code', 'split',
                  'secondary_structure', 'structure_source',
                  'pdb_id', 'chain_id']
    light_cols += shift_cols

    # Add DSSP and angle columns if available
    for c in ['phi', 'psi', 'rel_acc']:
        if c in all_cols:
            light_cols.append(c)

    # Add quality indicators
    for c in ['alignment_score', 'alignment_type', 'sequence_identity',
              'resolution', 'r_value', 'structure_method']:
        if c in all_cols:
            light_cols.append(c)

    light_cols = [c for c in light_cols if c in all_cols]

    # Use per-fold files if available
    fold_files = [os.path.join(data_dir, f'structure_data_hybrid_fold_{f}.csv')
                  for f in range(1, 6)]
    if all(os.path.exists(f) for f in fold_files):
        print("Loading from per-fold files...")
        parts = []
        for ff in fold_files:
            avail = [c for c in light_cols if c in pd.read_csv(ff, nrows=0).columns]
            parts.append(pd.read_csv(ff, usecols=avail, dtype={'bmrb_id': str}, low_memory=False))
        df = pd.concat(parts, ignore_index=True)
        del parts
    else:
        print(f"Loading from {data_file}...")
        df = pd.read_csv(data_file, usecols=light_cols, dtype={'bmrb_id': str}, low_memory=False)

    return df, shift_cols


def analyze_shift_coverage(df, shift_cols):
    """Analyze shift coverage patterns."""
    print("\n=== Shift Coverage Analysis ===")

    # Overall coverage per shift type
    coverage = {}
    total = len(df)
    for col in shift_cols:
        n_obs = df[col].notna().sum()
        coverage[col] = {
            'n_observed': int(n_obs),
            'pct_observed': float(n_obs / total * 100),
            'is_backbone': col in BACKBONE_SHIFTS,
        }
        print(f"  {col:20s}: {n_obs:>8,} / {total:,} ({n_obs/total*100:.1f}%)")

    # Per-protein shift completeness
    backbone_cols = [c for c in shift_cols if c in BACKBONE_SHIFTS]
    per_protein = df.groupby('bmrb_id')[backbone_cols].apply(
        lambda x: (x.notna().sum() / len(x)).mean()
    )

    print(f"\n  Per-protein backbone shift completeness:")
    print(f"    Mean:   {per_protein.mean():.3f}")
    print(f"    Median: {per_protein.median():.3f}")
    print(f"    Std:    {per_protein.std():.3f}")
    print(f"    <25%:   {(per_protein < 0.25).sum()} proteins")
    print(f"    <50%:   {(per_protein < 0.50).sum()} proteins")
    print(f"    >90%:   {(per_protein > 0.90).sum()} proteins")

    return coverage, per_protein


def analyze_per_protein_quality(df, shift_cols):
    """Analyze quality metrics per protein."""
    print("\n=== Per-Protein Quality Analysis ===")

    backbone_cols = [c for c in shift_cols if c in BACKBONE_SHIFTS]

    per_protein = df.groupby('bmrb_id').agg(
        n_residues=('residue_id', 'count'),
        n_unique_residues=('residue_id', 'nunique'),
        fold=('split', 'first'),
    )

    # Count observed backbone shifts per protein
    for col in backbone_cols:
        per_protein[f'{col}_observed'] = df.groupby('bmrb_id')[col].apply(lambda x: x.notna().sum())

    per_protein['backbone_coverage'] = per_protein[[f'{c}_observed' for c in backbone_cols]].sum(axis=1) / (per_protein['n_residues'] * len(backbone_cols))

    # Add structure source if available
    if 'structure_source' in df.columns:
        per_protein['structure_source'] = df.groupby('bmrb_id')['structure_source'].first()

    # Add resolution if available
    if 'resolution' in df.columns:
        per_protein['resolution'] = df.groupby('bmrb_id')['resolution'].first()

    print(f"  Total proteins: {len(per_protein):,}")
    print(f"  Protein sizes:")
    print(f"    Mean residues:   {per_protein['n_residues'].mean():.0f}")
    print(f"    Median residues: {per_protein['n_residues'].median():.0f}")
    print(f"    Min:             {per_protein['n_residues'].min()}")
    print(f"    Max:             {per_protein['n_residues'].max()}")

    if 'structure_source' in per_protein.columns:
        print(f"\n  Structure sources:")
        src_counts = per_protein['structure_source'].value_counts()
        for src, count in src_counts.items():
            print(f"    {src}: {count:,} proteins")

    return per_protein


def analyze_shift_distributions(df, shift_cols):
    """Analyze shift value distributions and outliers."""
    print("\n=== Shift Distribution Analysis ===")

    distributions = {}
    for col in shift_cols:
        if col not in BACKBONE_SHIFTS:
            continue
        vals = df[col].dropna()
        if len(vals) < 100:
            continue

        mean = float(vals.mean())
        std = float(vals.std())
        q01 = float(vals.quantile(0.01))
        q99 = float(vals.quantile(0.99))

        # Count outliers at various thresholds
        n_4std = int(((vals < mean - 4*std) | (vals > mean + 4*std)).sum())
        n_3std = int(((vals < mean - 3*std) | (vals > mean + 3*std)).sum())

        distributions[col] = {
            'count': len(vals),
            'mean': mean,
            'std': std,
            'min': float(vals.min()),
            'max': float(vals.max()),
            'q01': q01,
            'q99': q99,
            'n_outliers_3std': n_3std,
            'n_outliers_4std': n_4std,
            'pct_outliers_3std': float(n_3std / len(vals) * 100),
            'pct_outliers_4std': float(n_4std / len(vals) * 100),
        }

        print(f"  {col:15s}: mean={mean:8.2f}, std={std:6.2f}, "
              f"range=[{vals.min():.1f}, {vals.max():.1f}], "
              f"outliers(3σ)={n_3std:,} ({n_3std/len(vals)*100:.2f}%)")

    return distributions


def analyze_fold_balance(df, shift_cols):
    """Analyze fold balance."""
    print("\n=== Fold Balance Analysis ===")

    backbone_cols = [c for c in shift_cols if c in BACKBONE_SHIFTS]

    fold_stats = {}
    for fold in sorted(df['split'].unique()):
        fold_df = df[df['split'] == fold]
        n_proteins = fold_df['bmrb_id'].nunique()
        n_residues = len(fold_df)

        bb_coverage = {}
        for col in backbone_cols:
            bb_coverage[col] = float(fold_df[col].notna().mean() * 100)

        fold_stats[int(fold)] = {
            'n_proteins': n_proteins,
            'n_residues': n_residues,
            'backbone_coverage': bb_coverage,
        }

        print(f"  Fold {fold}: {n_proteins:,} proteins, {n_residues:,} residues")
        for col in backbone_cols:
            print(f"    {col}: {bb_coverage[col]:.1f}% observed")

    return fold_stats


def analyze_residue_distribution(df):
    """Analyze amino acid distribution."""
    print("\n=== Residue Distribution ===")

    if 'residue_code' not in df.columns:
        return {}

    counts = df['residue_code'].value_counts()
    total = len(df)

    distribution = {}
    for aa, count in counts.items():
        distribution[aa] = {
            'count': int(count),
            'pct': float(count / total * 100),
        }
        print(f"  {aa}: {count:>8,} ({count/total*100:.1f}%)")

    return distribution


def compare_alphafold_vs_experimental(data_dir='data'):
    """Compare alphafold and experimental structure datasets."""
    print("\n=== AlphaFold vs Experimental Comparison ===")

    af_file = os.path.join(data_dir, 'structure_data_alphafold.csv')
    exp_file = os.path.join(data_dir, 'structure_data_experimental.csv')

    comparison = {}

    for name, fpath in [('alphafold', af_file), ('experimental', exp_file)]:
        if not os.path.exists(fpath):
            print(f"  {name}: file not found")
            continue

        # Read just header + light cols
        all_cols = pd.read_csv(fpath, nrows=0).columns.tolist()
        shift_cols = sorted([c for c in all_cols if c.endswith('_shift')])
        light_cols = ['bmrb_id', 'residue_id', 'split'] + [c for c in shift_cols if c in BACKBONE_SHIFTS]
        light_cols = [c for c in light_cols if c in all_cols]

        # Sample to avoid memory issues - read first 500k rows
        df = pd.read_csv(fpath, usecols=light_cols, dtype={'bmrb_id': str},
                         low_memory=False, nrows=500000)

        bb_cols = [c for c in BACKBONE_SHIFTS if c in df.columns]

        stats = {
            'n_rows': len(df),
            'n_proteins': int(df['bmrb_id'].nunique()),
            'folds': sorted(df['split'].unique().tolist()) if 'split' in df.columns else [],
        }

        for col in bb_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                stats[f'{col}_mean'] = float(vals.mean())
                stats[f'{col}_std'] = float(vals.std())
                stats[f'{col}_coverage'] = float(vals.count() / len(df) * 100)

        comparison[name] = stats
        print(f"  {name}: {stats['n_rows']:,} rows, {stats['n_proteins']:,} proteins, folds={stats['folds']}")
        for col in bb_cols:
            cov_key = f'{col}_coverage'
            if cov_key in stats:
                print(f"    {col}: coverage={stats[cov_key]:.1f}%, mean={stats.get(f'{col}_mean', 0):.2f}")

        del df
        gc.collect()

    return comparison


# ============================================================================
# Plotting
# ============================================================================

def plot_shift_distributions(df, shift_cols, plot_dir):
    """Plot shift value distributions."""
    backbone = [c for c in shift_cols if c in BACKBONE_SHIFTS]
    n_plots = min(6, len(backbone))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(backbone[:n_plots]):
        ax = axes[idx]
        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        ax.hist(vals, bins=100, alpha=0.7, color='steelblue', edgecolor='black', density=True)
        mean = vals.mean()
        std = vals.std()
        ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.1f}')
        ax.axvline(mean - 4*std, color='orange', linestyle=':', label=f'±4σ')
        ax.axvline(mean + 4*std, color='orange', linestyle=':')

        n_outliers = ((vals < mean - 4*std) | (vals > mean + 4*std)).sum()
        name = col.replace('_shift', '').upper()
        ax.set_title(f'{name} (n={len(vals):,}, outliers={n_outliers})', fontsize=11)
        ax.set_xlabel('ppm')
        ax.legend(fontsize=8)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Backbone Shift Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'shift_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_coverage_per_protein(per_protein_coverage, plot_dir):
    """Plot shift coverage distribution across proteins."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(per_protein_coverage, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(per_protein_coverage.mean(), color='red', linestyle='--',
               label=f'Mean: {per_protein_coverage.mean():.2f}')
    ax.axvline(per_protein_coverage.median(), color='green', linestyle='--',
               label=f'Median: {per_protein_coverage.median():.2f}')
    ax.set_xlabel('Backbone Shift Coverage (fraction)')
    ax.set_ylabel('Number of Proteins')
    ax.set_title('Per-Protein Backbone Shift Coverage')
    ax.legend()

    ax = axes[1]
    # CDF
    sorted_cov = np.sort(per_protein_coverage)
    cdf = np.arange(1, len(sorted_cov) + 1) / len(sorted_cov)
    ax.plot(sorted_cov, cdf, color='steelblue', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(per_protein_coverage.median(), color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Backbone Shift Coverage (fraction)')
    ax.set_ylabel('Cumulative Fraction of Proteins')
    ax.set_title('Coverage CDF')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'coverage_per_protein.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_protein_sizes(per_protein, plot_dir):
    """Plot protein size distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = per_protein['n_residues']

    ax = axes[0]
    ax.hist(sizes, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(sizes.mean(), color='red', linestyle='--', label=f'Mean: {sizes.mean():.0f}')
    ax.axvline(sizes.median(), color='green', linestyle='--', label=f'Median: {sizes.median():.0f}')
    ax.set_xlabel('Residues per Protein')
    ax.set_ylabel('Count')
    ax.set_title('Protein Size Distribution')
    ax.legend()
    ax.set_xlim(0, np.percentile(sizes, 99))

    ax = axes[1]
    coverage = per_protein['backbone_coverage']
    ax.scatter(sizes, coverage, alpha=0.3, s=5, rasterized=True)
    ax.set_xlabel('Protein Size (residues)')
    ax.set_ylabel('Backbone Shift Coverage')
    ax.set_title('Size vs Coverage')
    ax.set_xlim(0, np.percentile(sizes, 99))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'protein_sizes.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_fold_comparison(fold_stats, plot_dir):
    """Plot fold balance."""
    folds = sorted(fold_stats.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Proteins and residues per fold
    ax = axes[0]
    proteins = [fold_stats[f]['n_proteins'] for f in folds]
    residues = [fold_stats[f]['n_residues'] for f in folds]
    x = np.arange(len(folds))
    width = 0.35
    ax.bar(x - width/2, proteins, width, label='Proteins', color='steelblue', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, residues, width, label='Residues', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylabel('Proteins', color='steelblue')
    ax2.set_ylabel('Residues', color='coral')
    ax.set_title('Fold Balance')

    # Coverage per fold
    ax = axes[1]
    bb_cols = sorted(fold_stats[folds[0]]['backbone_coverage'].keys())
    for col in bb_cols:
        coverages = [fold_stats[f]['backbone_coverage'][col] for f in folds]
        label = col.replace('_shift', '').upper()
        ax.plot(folds, coverages, 'o-', label=label)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Backbone Shift Coverage by Fold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'fold_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_residue_distribution(aa_dist, plot_dir):
    """Plot amino acid distribution."""
    if not aa_dist:
        return

    sorted_aa = sorted(aa_dist.items(), key=lambda x: -x[1]['count'])
    names = [aa for aa, _ in sorted_aa]
    counts = [v['count'] for _, v in sorted_aa]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(names)), counts, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Count')
    ax.set_title('Amino Acid Distribution in Training Data')

    # Add percentage labels
    total = sum(counts)
    for i, (name, count) in enumerate(zip(names, counts)):
        ax.text(i, count + total * 0.002, f'{count/total*100:.1f}%',
                ha='center', fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'residue_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("DATA QUALITY ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df, shift_cols = load_light_data('data')
    print(f"  Loaded {len(df):,} residues from {df['bmrb_id'].nunique():,} proteins")

    # Run analyses
    coverage, per_protein_coverage = analyze_shift_coverage(df, shift_cols)
    per_protein = analyze_per_protein_quality(df, shift_cols)
    distributions = analyze_shift_distributions(df, shift_cols)
    fold_stats = analyze_fold_balance(df, shift_cols)
    aa_dist = analyze_residue_distribution(df)

    # AlphaFold vs Experimental comparison
    af_vs_exp = compare_alphafold_vs_experimental('data')

    # Generate plots
    print("\nGenerating plots...")
    plot_shift_distributions(df, shift_cols, PLOT_DIR)
    print("  - shift_distributions.png")

    plot_coverage_per_protein(per_protein_coverage, PLOT_DIR)
    print("  - coverage_per_protein.png")

    plot_protein_sizes(per_protein, PLOT_DIR)
    print("  - protein_sizes.png")

    plot_fold_comparison(fold_stats, PLOT_DIR)
    print("  - fold_comparison.png")

    plot_residue_distribution(aa_dist, PLOT_DIR)
    print("  - residue_distribution.png")

    # Save results
    all_results = {
        'shift_coverage': coverage,
        'shift_distributions': distributions,
        'fold_stats': fold_stats,
        'residue_distribution': aa_dist,
        'alphafold_vs_experimental': af_vs_exp,
        'per_protein_summary': {
            'total_proteins': len(per_protein),
            'mean_size': float(per_protein['n_residues'].mean()),
            'median_size': float(per_protein['n_residues'].median()),
            'mean_backbone_coverage': float(per_protein['backbone_coverage'].mean()),
            'median_backbone_coverage': float(per_protein['backbone_coverage'].median()),
            'proteins_below_50pct_coverage': int((per_protein['backbone_coverage'] < 0.50).sum()),
            'proteins_above_90pct_coverage': int((per_protein['backbone_coverage'] > 0.90).sum()),
        },
    }

    json_path = os.path.join(RESULTS_DIR, 'data_quality_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"  Total proteins: {len(per_protein):,}")
    print(f"  Total residues: {len(df):,}")
    print(f"  Mean protein size: {per_protein['n_residues'].mean():.0f} residues")
    print(f"  Mean backbone coverage: {per_protein['backbone_coverage'].mean():.1%}")
    print(f"  Proteins with <50% backbone coverage: {(per_protein['backbone_coverage'] < 0.50).sum():,}")
    print(f"  Proteins with >90% backbone coverage: {(per_protein['backbone_coverage'] > 0.90).sum():,}")

    # Data quality recommendation
    low_cov = (per_protein['backbone_coverage'] < 0.25).sum()
    total = len(per_protein)
    print(f"\n  Very low coverage (<25%): {low_cov:,} / {total:,} ({low_cov/total*100:.1f}%)")
    if low_cov > total * 0.1:
        print(f"  >> {low_cov/total*100:.0f}% of proteins have very sparse shift data.")
        print(f"  >> Filtering these could improve data quality with minimal volume loss.")

    print("=" * 70)


if __name__ == '__main__':
    main()
