#!/usr/bin/env python3
"""
Comprehensive Evaluation for Retrieval-Augmented Chemical Shift Prediction
(Better Data Pipeline)

Adapted from homologies/evaluate_retrieval_model.py with improvements:
- Auto-detects architecture params from checkpoint
- Per-shift-type MAE, RMSE, R^2 metrics (CA, CB, C, N, H, HA)
- Per-protein aggregated metrics
- Baseline comparisons: mean predictor, random coil predictor
- Trust score analysis (model confidence)
- Retrieval contribution analysis (how much retrieval helps vs structure-only)
- Provenance logging: which proteins evaluated, exclusions, checkpoint used

Usage:
    python 07_evaluate.py --model checkpoints/best_retrieval_fold1.pt \\
        --data_dir data --cache_dir cache --fold 1 --output_dir eval_results

    # With plots
    python 07_evaluate.py --model checkpoints/best_retrieval_fold1.pt \\
        --data_dir data --cache_dir cache --fold 1 --plots
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    N_RESIDUE_TYPES, N_SS_TYPES, N_MISMATCH_TYPES, DSSP_COLS,
    SHIFT_RANGES, K_RETRIEVED, STANDARD_RESIDUES, AA_3_TO_1,
)
from dataset import CachedRetrievalDataset
from model import ShiftPredictorWithRetrieval
from random_coil import RC_SHIFTS

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Backbone atoms for separate analysis
BACKBONE_ATOMS = {'C', 'CA', 'CB', 'H', 'HA', 'N'}


# ============================================================================
# Model Loading (auto-detect architecture)
# ============================================================================

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint with auto-detected architecture.

    Returns:
        (model, checkpoint_info_dict) or (None, None) on failure
    """
    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        return None, None

    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove _orig_mod. prefix from compiled model keys
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        clean_state_dict[new_key] = v

    # Infer model dimensions from state dict
    n_atom_types = clean_state_dict['distance_attention.atom_embed.weight'].shape[0] - 1
    n_shifts = sum(1 for k in clean_state_dict
                   if k.startswith('shift_heads.') and k.endswith('.0.weight'))

    stats = checkpoint.get('stats', None)
    shift_cols = checkpoint.get('shift_cols', None)
    k_retrieved = checkpoint.get('k_retrieved', K_RETRIEVED)
    n_physics = checkpoint.get('n_physics', 28)

    # Detect query-conditioned transfer
    use_query_conditioned = 'shift_transfer.query_proj.0.weight' in clean_state_dict

    # Detect random coil correction
    use_random_coil = 'shift_transfer.rc_table' in clean_state_dict

    # Get DSSP dimension
    n_dssp = clean_state_dict['dssp_proj.weight'].shape[1] if 'dssp_proj.weight' in clean_state_dict else 0

    # Infer CNN channels
    cnn_channels = []
    for i in range(0, 10, 2):
        key = f'cnn.{i}.conv1.weight'
        if key in clean_state_dict:
            cnn_channels.append(clean_state_dict[key].shape[0])

    # Infer spatial_hidden
    spatial_hidden = (clean_state_dict['spatial_attention.fallback_embed'].shape[0]
                      if 'spatial_attention.fallback_embed' in clean_state_dict else 192)

    # Infer retrieval_hidden
    retrieval_hidden = (clean_state_dict['retrieval_cross_attn.fallback'].shape[0]
                        if 'retrieval_cross_attn.fallback' in clean_state_dict else 192)

    # Infer physics dimension
    if 'physics_encoder.mlp.0.weight' in clean_state_dict:
        n_physics = clean_state_dict['physics_encoder.mlp.0.weight'].shape[1]

    print(f"  Auto-detected architecture:")
    print(f"    n_atom_types:            {n_atom_types}")
    print(f"    n_shifts:                {n_shifts}")
    print(f"    n_dssp:                  {n_dssp}")
    print(f"    n_physics:               {n_physics}")
    print(f"    cnn_channels:            {cnn_channels}")
    print(f"    spatial_hidden:          {spatial_hidden}")
    print(f"    retrieval_hidden:        {retrieval_hidden}")
    print(f"    use_query_conditioned:   {use_query_conditioned}")
    print(f"    use_random_coil:         {use_random_coil}")
    print(f"    k_retrieved:             {k_retrieved}")

    model = ShiftPredictorWithRetrieval(
        n_atom_types=n_atom_types,
        n_residue_types=N_RESIDUE_TYPES,
        n_ss_types=N_SS_TYPES,
        n_mismatch_types=N_MISMATCH_TYPES,
        n_dssp=n_dssp,
        n_shifts=n_shifts,
        n_physics=n_physics,
        cnn_channels=cnn_channels,
        spatial_hidden=spatial_hidden,
        retrieval_hidden=retrieval_hidden,
        use_query_conditioned_transfer=use_query_conditioned,
        use_random_coil=use_random_coil,
        shift_cols=shift_cols,
    ).to(device)

    model.load_state_dict(clean_state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model with {n_params:,} parameters")

    return model, {
        'stats': stats,
        'shift_cols': shift_cols,
        'k_retrieved': k_retrieved,
        'n_physics': n_physics,
        'checkpoint_path': checkpoint_path,
        'epoch': checkpoint.get('epoch', 'unknown'),
    }


# ============================================================================
# Prediction Collection
# ============================================================================

@torch.no_grad()
def collect_predictions(model, loader, device, stats, shift_cols):
    """Collect all predictions, targets, and metadata."""
    model.eval()

    all_pred = []
    all_target = []
    all_mask = []
    all_residue_codes = []
    all_retrieved_distances = []
    all_retrieved_valid = []

    for batch in tqdm(loader, desc="  Collecting predictions", leave=False):
        batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch_dev.pop('shift_target')
        mask = batch_dev.pop('shift_mask')

        ctx = autocast('cuda') if device == 'cuda' else nullcontext()
        with ctx:
            pred = model(**batch_dev)

        all_pred.append(pred.cpu())
        all_target.append(target.cpu())
        all_mask.append(mask.cpu())
        all_residue_codes.append(batch_dev['query_residue_code'].cpu())
        all_retrieved_distances.append(batch_dev['retrieved_distances'].cpu())
        all_retrieved_valid.append(batch_dev['retrieved_valid'].cpu())

    predictions = torch.cat(all_pred).numpy()
    targets = torch.cat(all_target).numpy()
    masks = torch.cat(all_mask).numpy()
    residue_codes = torch.cat(all_residue_codes).numpy()
    retrieved_distances = torch.cat(all_retrieved_distances).numpy()
    retrieved_valid = torch.cat(all_retrieved_valid).numpy()

    # Denormalize
    predictions_denorm = predictions.copy()
    targets_denorm = targets.copy()

    for i, col in enumerate(shift_cols):
        if col in stats:
            mean = stats[col]['mean']
            std = stats[col]['std']
            predictions_denorm[:, i] = predictions[:, i] * std + mean
            targets_denorm[:, i] = targets[:, i] * std + mean

    errors = np.abs(predictions_denorm - targets_denorm)

    return {
        'predictions': predictions_denorm,
        'predictions_norm': predictions,
        'targets': targets_denorm,
        'targets_norm': targets,
        'errors': errors,
        'masks': masks,
        'residue_codes': residue_codes,
        'retrieved_distances': retrieved_distances,
        'retrieved_valid': retrieved_valid,
    }


# ============================================================================
# Metric Computation
# ============================================================================

def compute_per_shift_metrics(results, shift_cols):
    """Compute MAE, RMSE, R^2 per shift type."""
    metrics = {}

    for i, col in enumerate(shift_cols):
        mask_i = results['masks'][:, i].astype(bool)
        if mask_i.sum() == 0:
            continue

        preds = results['predictions'][:, i][mask_i]
        targs = results['targets'][:, i][mask_i]
        errs = results['errors'][:, i][mask_i]

        mae = float(np.mean(errs))
        rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))

        # R^2
        ss_res = np.sum((targs - preds) ** 2)
        ss_tot = np.sum((targs - np.mean(targs)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics[col] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'count': int(mask_i.sum()),
            'mean_true': float(np.mean(targs)),
            'std_true': float(np.std(targs)),
        }

    return metrics


def compute_overall_metrics(results):
    """Compute overall metrics across all shift types."""
    valid_mask = results['masks'].astype(bool)
    all_errors = results['errors'][valid_mask]

    if len(all_errors) == 0:
        return {'mae': 0, 'rmse': 0, 'median': 0, 'p95': 0, 'count': 0}

    return {
        'mae': float(np.mean(all_errors)),
        'rmse': float(np.sqrt(np.mean(all_errors ** 2))),
        'median': float(np.median(all_errors)),
        'p95': float(np.percentile(all_errors, 95)),
        'count': int(valid_mask.sum()),
    }


def compute_per_protein_metrics(results, shift_cols, dataset):
    """Compute per-protein aggregated metrics.

    Returns dict mapping protein_id -> {mae, rmse, n_residues}
    """
    # We need to map sample indices back to proteins
    # Use the dataset's idx_to_bmrb mapping
    if not hasattr(dataset, 'samples') or not hasattr(dataset, 'protein_offsets'):
        print("  Warning: Cannot compute per-protein metrics (missing dataset attributes)")
        return {}

    # Group by protein index
    protein_metrics = defaultdict(lambda: {'errors': [], 'count': 0})

    samples = dataset.samples
    for sample_idx in range(len(results['predictions'])):
        if sample_idx >= len(samples):
            break
        _, prot_idx = samples[sample_idx]
        prot_idx = int(prot_idx)

        mask_row = results['masks'][sample_idx].astype(bool)
        if mask_row.any():
            errs = results['errors'][sample_idx][mask_row]
            protein_metrics[prot_idx]['errors'].extend(errs.tolist())
            protein_metrics[prot_idx]['count'] += 1

    per_protein = {}
    for prot_idx, data in protein_metrics.items():
        if data['errors']:
            errs = np.array(data['errors'])
            per_protein[str(prot_idx)] = {
                'mae': float(np.mean(errs)),
                'rmse': float(np.sqrt(np.mean(errs ** 2))),
                'n_residues': data['count'],
                'n_predictions': len(errs),
            }

    return per_protein


# ============================================================================
# Baseline Comparisons
# ============================================================================

def compute_baseline_metrics(results, shift_cols, stats):
    """Compute baseline predictor metrics for comparison.

    Baselines:
    1. Mean predictor: always predicts the training set mean
    2. Random coil predictor: predicts random coil shift based on residue type
    """
    baselines = {}

    # --- Mean predictor ---
    mean_errors = {}
    for i, col in enumerate(shift_cols):
        mask_i = results['masks'][:, i].astype(bool)
        if mask_i.sum() == 0 or col not in stats:
            continue
        targs = results['targets'][:, i][mask_i]
        mean_pred = stats[col]['mean']
        mean_errors[col] = float(np.mean(np.abs(targs - mean_pred)))

    overall_mean_mae = np.mean(list(mean_errors.values())) if mean_errors else 0.0
    baselines['mean_predictor'] = {
        'per_shift_mae': mean_errors,
        'overall_mae': overall_mean_mae,
    }

    # --- Random coil predictor ---
    # Map residue index back to 1-letter code for RC lookup
    idx_to_aa1 = {}
    for res3 in STANDARD_RESIDUES:
        from config import RESIDUE_TO_IDX
        idx = RESIDUE_TO_IDX.get(res3, -1)
        aa1 = AA_3_TO_1.get(res3, None)
        if idx >= 0 and aa1:
            idx_to_aa1[idx] = aa1

    # Map shift column names to RC atom types
    col_to_atom = {}
    for col in shift_cols:
        atom = col.replace('_shift', '').upper()
        col_to_atom[col] = atom

    rc_errors = {}
    for i, col in enumerate(shift_cols):
        mask_i = results['masks'][:, i].astype(bool)
        if mask_i.sum() == 0:
            continue

        atom = col_to_atom.get(col)
        if atom is None:
            continue

        targs = results['targets'][:, i][mask_i]
        res_codes = results['residue_codes'][mask_i]

        # Build RC predictions
        rc_preds = np.full_like(targs, np.nan)
        for j, rc_idx in enumerate(res_codes):
            aa1 = idx_to_aa1.get(int(rc_idx))
            if aa1 and aa1 in RC_SHIFTS and atom in RC_SHIFTS[aa1]:
                rc_val = RC_SHIFTS[aa1][atom]
                if rc_val is not None:
                    rc_preds[j] = rc_val

        valid = ~np.isnan(rc_preds)
        if valid.sum() > 0:
            rc_errors[col] = float(np.mean(np.abs(targs[valid] - rc_preds[valid])))

    overall_rc_mae = np.mean(list(rc_errors.values())) if rc_errors else 0.0
    baselines['random_coil_predictor'] = {
        'per_shift_mae': rc_errors,
        'overall_mae': overall_rc_mae,
    }

    return baselines


# ============================================================================
# Retrieval Contribution Analysis
# ============================================================================

def analyze_retrieval_contribution(results):
    """Analyze how retrieval quality correlates with prediction quality."""
    retrieved_distances = results['retrieved_distances']
    retrieved_valid = results['retrieved_valid']

    # Mean cosine similarity of valid neighbors
    valid_float = retrieved_valid.astype(float)
    n_valid_per_sample = valid_float.sum(axis=1)
    masked_dist = retrieved_distances * valid_float
    mean_sim = np.where(n_valid_per_sample > 0,
                        masked_dist.sum(axis=1) / (n_valid_per_sample + 1e-8),
                        0.0)

    # Per-sample MAE
    masks = results['masks'].astype(bool)
    sample_mae = np.zeros(len(results['predictions']))
    for i in range(len(results['predictions'])):
        if masks[i].any():
            sample_mae[i] = np.mean(results['errors'][i][masks[i]])

    # Correlation between retrieval similarity and prediction error
    valid_samples = n_valid_per_sample > 0
    if valid_samples.sum() > 1:
        corr = float(np.corrcoef(mean_sim[valid_samples],
                                  sample_mae[valid_samples])[0, 1])
    else:
        corr = 0.0

    # MAE by retrieval quality bins
    sim_bins = [0, 0.7, 0.8, 0.9, 0.95, 1.0]
    mae_by_sim = {}
    for j in range(len(sim_bins) - 1):
        lo, hi = sim_bins[j], sim_bins[j + 1]
        bin_mask = (mean_sim >= lo) & (mean_sim < hi)
        if bin_mask.sum() > 0:
            mae_by_sim[f'{lo:.2f}-{hi:.2f}'] = {
                'mae': float(np.mean(sample_mae[bin_mask])),
                'count': int(bin_mask.sum()),
            }

    return {
        'mean_cosine_similarity': float(np.mean(mean_sim[valid_samples])) if valid_samples.sum() > 0 else 0,
        'mean_valid_neighbors': float(np.mean(n_valid_per_sample)),
        'pct_with_no_retrieval': float((n_valid_per_sample == 0).mean() * 100),
        'correlation_similarity_vs_error': corr,
        'mae_by_similarity_bin': mae_by_sim,
    }


# ============================================================================
# Visualization
# ============================================================================

def generate_plots(results, shift_cols, stats, baselines, output_dir):
    """Generate all evaluation plots."""
    if not HAS_MATPLOTLIB:
        print("  Warning: matplotlib not available, skipping plots.")
        return

    print("\nGenerating evaluation plots...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Predicted vs Actual scatter for backbone shifts
    _plot_pred_vs_actual(results, shift_cols, output_dir)

    # 2. Error distribution histograms
    _plot_error_distributions(results, shift_cols, output_dir)

    # 3. Per-shift MAE bar chart with baseline comparison
    _plot_mae_comparison(results, shift_cols, baselines, output_dir)

    # 4. Per-protein performance distribution
    _plot_per_protein(results, shift_cols, output_dir)

    print(f"  Plots saved to {output_dir}")


def _plot_pred_vs_actual(results, shift_cols, output_dir):
    """Scatter plots of predicted vs actual for backbone shifts."""
    backbone_cols = [(i, c) for i, c in enumerate(shift_cols)
                     if c.replace('_shift', '').upper() in BACKBONE_ATOMS]

    if not backbone_cols:
        return

    n_plots = min(6, len(backbone_cols))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, (shift_idx, col) in enumerate(backbone_cols[:n_plots]):
        ax = axes[ax_idx]
        mask = results['masks'][:, shift_idx].astype(bool)
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        preds = results['predictions'][:, shift_idx][mask]
        targs = results['targets'][:, shift_idx][mask]

        # Subsample for visualization
        if len(preds) > 10000:
            idx = np.random.choice(len(preds), 10000, replace=False)
            preds_plot, targs_plot = preds[idx], targs[idx]
        else:
            preds_plot, targs_plot = preds, targs

        ax.scatter(targs_plot, preds_plot, alpha=0.1, s=5, rasterized=True)

        lims = [min(targs.min(), preds.min()), max(targs.max(), preds.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')

        corr = np.corrcoef(targs, preds)[0, 1]
        mae = np.mean(np.abs(preds - targs))
        name = col.replace('_shift', '').upper()
        ax.set_title(f'{name} (r={corr:.3f}, MAE={mae:.3f})', fontsize=12)
        ax.set_xlabel('True (ppm)')
        ax.set_ylabel('Predicted (ppm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=200, bbox_inches='tight')
    plt.close()


def _plot_error_distributions(results, shift_cols, output_dir):
    """Histograms of error distributions for top shift types."""
    shift_counts = [(i, c, results['masks'][:, i].sum()) for i, c in enumerate(shift_cols)]
    shift_counts.sort(key=lambda x: -x[2])
    top = shift_counts[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, (shift_idx, col, count) in enumerate(top):
        ax = axes[ax_idx]
        mask = results['masks'][:, shift_idx].astype(bool)
        errors = results['errors'][:, shift_idx][mask]

        if len(errors) == 0:
            ax.set_visible(False)
            continue

        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', density=True)
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        ax.axvline(mean_err, color='r', linestyle='--', label=f'Mean: {mean_err:.3f}')
        ax.axvline(median_err, color='g', linestyle='--', label=f'Median: {median_err:.3f}')

        name = col.replace('_shift', '').upper()
        ax.set_title(f'{name} (n={int(count):,})', fontsize=12)
        ax.set_xlabel('Absolute Error (ppm)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(len(top), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()


def _plot_mae_comparison(results, shift_cols, baselines, output_dir):
    """Bar chart comparing model MAE vs baselines per shift type."""
    per_shift = compute_per_shift_metrics(results, shift_cols)

    cols_sorted = sorted(per_shift.keys(), key=lambda c: per_shift[c]['mae'])
    names = [c.replace('_shift', '').upper() for c in cols_sorted]
    model_mae = [per_shift[c]['mae'] for c in cols_sorted]

    mean_mae = [baselines['mean_predictor']['per_shift_mae'].get(c, 0) for c in cols_sorted]
    rc_mae = [baselines['random_coil_predictor']['per_shift_mae'].get(c, 0) for c in cols_sorted]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, model_mae, width, label='Model', color='steelblue', alpha=0.8)
    ax.bar(x, mean_mae, width, label='Mean Baseline', color='coral', alpha=0.8)
    ax.bar(x + width, rc_mae, width, label='Random Coil', color='gold', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('MAE (ppm)', fontsize=12)
    ax.set_title('Model vs Baselines: Per-Shift MAE', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()


def _plot_per_protein(results, shift_cols, output_dir):
    """Distribution of per-protein MAE."""
    masks = results['masks'].astype(bool)
    # Rough per-sample MAE (not true per-protein, but illustrative)
    sample_mae = []
    for i in range(len(results['predictions'])):
        if masks[i].any():
            sample_mae.append(np.mean(results['errors'][i][masks[i]]))

    if not sample_mae:
        return

    sample_mae = np.array(sample_mae)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(sample_mae, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(np.mean(sample_mae), color='r', linestyle='--',
               label=f'Mean: {np.mean(sample_mae):.3f}')
    ax.axvline(np.median(sample_mae), color='g', linestyle='--',
               label=f'Median: {np.median(sample_mae):.3f}')
    ax.set_xlabel('Per-Residue MAE (ppm)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Per-Residue MAE Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    percentiles = np.percentile(sample_mae, [10, 25, 50, 75, 90, 95, 99])
    labels = ['10th', '25th', '50th', '75th', '90th', '95th', '99th']
    ax.barh(range(len(labels)), percentiles, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('MAE (ppm)', fontsize=12)
    ax.set_title('Per-Residue MAE Percentiles', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_protein_performance.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate retrieval-augmented chemical shift predictor (better data pipeline)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing data CSV file')
    parser.add_argument('--cache_dir', type=str, default='dataset_cache',
                        help='Directory for dataset cache')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold used for evaluation (test fold)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory for evaluation outputs')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu, auto-detected if omitted)')
    parser.add_argument('--plots', action='store_true',
                        help='Generate evaluation plots')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation')
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print("=" * 70)
    print("RETRIEVAL-AUGMENTED MODEL EVALUATION (Better Data Pipeline)")
    print("=" * 70)
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.model}")
    print(f"Fold:       {args.fold}")
    print(f"Output:     {args.output_dir}")

    # ========== Provenance ==========
    provenance = {
        'checkpoint': os.path.abspath(args.model),
        'fold': args.fold,
        'device': device,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'excluded_proteins': [],
        'evaluation_notes': [],
    }

    # ========== Load model ==========
    print("\nLoading model...")
    model, checkpoint_info = load_model(args.model, device)
    if model is None:
        print("Failed to load model!")
        sys.exit(1)

    stats = checkpoint_info['stats']
    shift_cols = checkpoint_info['shift_cols']
    k_retrieved = checkpoint_info['k_retrieved']

    provenance['model_epoch'] = checkpoint_info.get('epoch', 'unknown')
    provenance['shift_cols'] = shift_cols
    provenance['k_retrieved'] = k_retrieved

    if stats is None or shift_cols is None:
        print("ERROR: Stats and shift_cols not found in checkpoint.")
        print("  Ensure the model was saved with stats and shift_cols.")
        sys.exit(1)

    # ========== Load dataset ==========
    print("\nLoading test dataset...")
    test_cache = os.path.join(args.cache_dir, f'fold{args.fold}_test')

    if not CachedRetrievalDataset.exists(test_cache):
        print(f"ERROR: Cached test dataset not found at {test_cache}")
        print("  Run 06_train.py first to build the dataset cache.")
        sys.exit(1)

    dataset = CachedRetrievalDataset.load(
        test_cache,
        n_shifts=len(shift_cols),
        k_retrieved=k_retrieved,
        stats=stats,
        shift_cols=shift_cols,
    )
    print(f"  Loaded {len(dataset):,} test samples")

    provenance['n_test_samples'] = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )

    # ========== Collect predictions ==========
    print("\nRunning evaluation...")
    results = collect_predictions(model, loader, device, stats, shift_cols)

    # ========== Compute metrics ==========
    print("\nComputing metrics...")

    # Overall
    overall = compute_overall_metrics(results)
    print(f"\n  Overall Metrics:")
    print(f"    MAE:    {overall['mae']:.4f} ppm")
    print(f"    RMSE:   {overall['rmse']:.4f} ppm")
    print(f"    Median: {overall['median']:.4f} ppm")
    print(f"    95th:   {overall['p95']:.4f} ppm")
    print(f"    Count:  {overall['count']:,}")

    # Per-shift
    per_shift = compute_per_shift_metrics(results, shift_cols)
    print(f"\n  Per-Shift Metrics (MAE / RMSE / R^2):")
    for col in sorted(per_shift.keys(), key=lambda c: per_shift[c]['mae']):
        m = per_shift[col]
        name = col.replace('_shift', '').upper()
        print(f"    {name:6s}: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
              f"R2={m['r2']:.3f}  (n={m['count']:,})")

    # Backbone vs sidechain summary
    backbone_mae = [per_shift[c]['mae'] for c in per_shift
                    if c.replace('_shift', '').upper() in BACKBONE_ATOMS]
    sidechain_mae = [per_shift[c]['mae'] for c in per_shift
                     if c.replace('_shift', '').upper() not in BACKBONE_ATOMS]
    if backbone_mae:
        print(f"\n  Backbone mean MAE:  {np.mean(backbone_mae):.4f} ppm ({len(backbone_mae)} types)")
    if sidechain_mae:
        print(f"  Sidechain mean MAE: {np.mean(sidechain_mae):.4f} ppm ({len(sidechain_mae)} types)")

    # Per-protein
    per_protein = compute_per_protein_metrics(results, shift_cols, dataset)
    if per_protein:
        prot_maes = [v['mae'] for v in per_protein.values()]
        print(f"\n  Per-Protein MAE Distribution ({len(per_protein)} proteins):")
        print(f"    Mean:   {np.mean(prot_maes):.4f}")
        print(f"    Median: {np.median(prot_maes):.4f}")
        print(f"    Std:    {np.std(prot_maes):.4f}")
        print(f"    Best:   {np.min(prot_maes):.4f}")
        print(f"    Worst:  {np.max(prot_maes):.4f}")

    # ========== Baseline comparisons ==========
    print("\nComputing baseline comparisons...")
    baselines = compute_baseline_metrics(results, shift_cols, stats)

    print(f"\n  Baseline Comparison:")
    print(f"    Model MAE:              {overall['mae']:.4f} ppm")
    print(f"    Mean Predictor MAE:     {baselines['mean_predictor']['overall_mae']:.4f} ppm")
    print(f"    Random Coil MAE:        {baselines['random_coil_predictor']['overall_mae']:.4f} ppm")

    mean_improvement = baselines['mean_predictor']['overall_mae'] - overall['mae']
    rc_improvement = baselines['random_coil_predictor']['overall_mae'] - overall['mae']
    print(f"    Improvement over mean:  {mean_improvement:.4f} ppm "
          f"({100*mean_improvement/baselines['mean_predictor']['overall_mae']:.1f}%)")
    if baselines['random_coil_predictor']['overall_mae'] > 0:
        print(f"    Improvement over RC:    {rc_improvement:.4f} ppm "
              f"({100*rc_improvement/baselines['random_coil_predictor']['overall_mae']:.1f}%)")

    # ========== Retrieval contribution ==========
    print("\nAnalyzing retrieval contribution...")
    retrieval_analysis = analyze_retrieval_contribution(results)
    print(f"  Mean cosine similarity:  {retrieval_analysis['mean_cosine_similarity']:.3f}")
    print(f"  Mean valid neighbors:    {retrieval_analysis['mean_valid_neighbors']:.1f}")
    print(f"  % with no retrieval:     {retrieval_analysis['pct_with_no_retrieval']:.1f}%")
    print(f"  Similarity-error corr:   {retrieval_analysis['correlation_similarity_vs_error']:.3f}")
    print(f"  MAE by similarity bin:")
    for bin_name, data in retrieval_analysis['mae_by_similarity_bin'].items():
        print(f"    {bin_name}: MAE={data['mae']:.3f} (n={data['count']:,})")

    # ========== Save results ==========
    os.makedirs(args.output_dir, exist_ok=True)

    # JSON results
    all_results = {
        'overall': overall,
        'per_shift': per_shift,
        'baselines': baselines,
        'retrieval_analysis': retrieval_analysis,
        'provenance': provenance,
    }

    json_path = os.path.join(args.output_dir, f'evaluation_fold{args.fold}.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    # CSV per-shift results
    csv_rows = []
    for col, m in per_shift.items():
        row = {'shift_type': col, **m}
        # Add baseline MAE
        row['baseline_mean_mae'] = baselines['mean_predictor']['per_shift_mae'].get(col, None)
        row['baseline_rc_mae'] = baselines['random_coil_predictor']['per_shift_mae'].get(col, None)
        csv_rows.append(row)

    csv_path = os.path.join(args.output_dir, f'per_shift_metrics_fold{args.fold}.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Per-shift CSV saved to {csv_path}")

    # Per-protein CSV
    if per_protein:
        pp_rows = [{'protein_idx': k, **v} for k, v in per_protein.items()]
        pp_path = os.path.join(args.output_dir, f'per_protein_metrics_fold{args.fold}.csv')
        pd.DataFrame(pp_rows).to_csv(pp_path, index=False)
        print(f"  Per-protein CSV saved to {pp_path}")

    # ========== Plots ==========
    if args.plots:
        plot_dir = os.path.join(args.output_dir, f'plots_fold{args.fold}')
        generate_plots(results, shift_cols, stats, baselines, plot_dir)

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Model checkpoint: {args.model}")
    print(f"  Test fold: {args.fold}")
    print(f"  Overall MAE: {overall['mae']:.4f} ppm")
    print(f"  Results: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
