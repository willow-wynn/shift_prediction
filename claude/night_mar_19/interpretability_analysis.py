#!/usr/bin/env python3
"""
Comprehensive Interpretability Analysis for Chemical Shift Prediction Model.

Analyzes:
1. Gate behavior (direct_gate, retrieval_gate) - how much does the model rely on retrieval?
2. Per-amino-acid MAE breakdown
3. Per-secondary-structure MAE breakdown
4. Retrieval quality vs prediction quality
5. Ablation: structure-only vs full model
6. Error analysis: worst proteins, worst residues
7. Shift coverage analysis
8. Per-amino-acid gate behavior

Outputs charts to claude/night_mar_19/plots/ and a summary JSON.
"""

import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from config import (
    STANDARD_RESIDUES, AA_3_TO_1, RESIDUE_TO_IDX,
    SS_TYPES, N_RESIDUE_TYPES, N_SS_TYPES,
    K_RETRIEVED, DSSP_COLS,
)
from dataset import CachedRetrievalDataset, parse_distance_columns, build_atom_vocabulary, parse_shift_columns, get_dssp_columns
from model import ShiftPredictorWithRetrieval
from random_coil import RC_SHIFTS

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(PLOT_DIR, exist_ok=True)

BACKBONE_SHIFTS = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}


def load_model(checkpoint_path, device):
    """Load model from checkpoint with auto-detection."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Clean compiled model keys
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Infer architecture
    n_atom_types = clean_sd['distance_attention.atom_embed.weight'].shape[0] - 1
    n_shifts = sum(1 for k in clean_sd if k.startswith('shift_heads.') and k.endswith('.0.weight'))
    if n_shifts == 0:
        # Fallback: use struct_head output dim (last linear layer)
        struct_head_keys = sorted([k for k in clean_sd if k.startswith('struct_head.') and k.endswith('.weight')])
        n_shifts = clean_sd[struct_head_keys[-1]].shape[0]

    n_dssp = clean_sd['dssp_proj.weight'].shape[1] if 'dssp_proj.weight' in clean_sd else 0

    cnn_channels = []
    for i in range(0, 10, 2):
        key = f'cnn.{i}.conv1.weight'
        if key in clean_sd:
            cnn_channels.append(clean_sd[key].shape[0])

    spatial_hidden = clean_sd.get('spatial_attention.fallback_embed', torch.zeros(192)).shape[0]
    retrieval_hidden = clean_sd.get('retrieval_cross_attn.fallback',
                                     clean_sd.get('cross_attn_layers.0.shift_embed.weight', torch.zeros(1, 320))).shape[-1]

    from config import N_MISMATCH_TYPES
    model = ShiftPredictorWithRetrieval(
        n_atom_types=n_atom_types,
        n_residue_types=N_RESIDUE_TYPES,
        n_ss_types=N_SS_TYPES,
        n_mismatch_types=N_MISMATCH_TYPES,
        n_dssp=n_dssp,
        n_shifts=n_shifts,
        cnn_channels=cnn_channels,
        spatial_hidden=spatial_hidden,
        retrieval_hidden=retrieval_hidden,
    ).to(device)

    filtered_sd = {k: v for k, v in clean_sd.items() if not k.startswith('physics_encoder.')}
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()
    return model, checkpoint


def collect_predictions_with_gates(model, loader, device, stats, shift_cols, max_batches=None):
    """Collect predictions, targets, gate activations, and metadata."""
    model.eval()

    all_pred = []
    all_target = []
    all_mask = []
    all_residue_codes = []
    all_ss_codes = []
    all_retrieved_distances = []
    all_retrieved_valid = []
    all_direct_gate = []
    all_retrieval_gate = []
    all_n_valid_neighbors = []
    all_struct_pred = []

    # Hook into gates to capture their outputs
    direct_gate_vals = []
    retrieval_gate_vals = []
    struct_pred_vals = []

    def hook_direct_gate(module, input, output):
        direct_gate_vals.append(output.detach().cpu())

    def hook_retrieval_gate(module, input, output):
        retrieval_gate_vals.append(output.detach().cpu())

    def hook_struct_head(module, input, output):
        struct_pred_vals.append(output.detach().cpu())

    h1 = model.direct_gate.register_forward_hook(hook_direct_gate)
    h2 = model.retrieval_gate.register_forward_hook(hook_retrieval_gate)
    h3 = model.struct_head.register_forward_hook(hook_struct_head)

    try:
        for batch_idx, batch in enumerate(tqdm(loader, desc="  Collecting predictions")):
            if max_batches and batch_idx >= max_batches:
                break

            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch_dev.pop('shift_target')
            mask = batch_dev.pop('shift_mask')

            # Save metadata before forward pass
            residue_codes = batch_dev['query_residue_code'].cpu()
            ss_codes = batch_dev['ss_idx'][:, batch_dev['ss_idx'].shape[1] // 2].cpu()  # center position
            ret_distances = batch_dev['retrieved_distances'].cpu()
            ret_valid = batch_dev['retrieved_valid'].cpu()

            direct_gate_vals.clear()
            retrieval_gate_vals.clear()
            struct_pred_vals.clear()

            with torch.no_grad():
                ctx = autocast('cuda') if device == 'cuda' else nullcontext()
                with ctx:
                    pred = model(**batch_dev)

            all_pred.append(pred.cpu())
            all_target.append(target.cpu())
            all_mask.append(mask.cpu())
            all_residue_codes.append(residue_codes)
            all_ss_codes.append(ss_codes)
            all_retrieved_distances.append(ret_distances)
            all_retrieved_valid.append(ret_valid)

            if direct_gate_vals:
                all_direct_gate.append(direct_gate_vals[0].squeeze(-1))
            if retrieval_gate_vals:
                all_retrieval_gate.append(retrieval_gate_vals[0].squeeze(-1))
            if struct_pred_vals:
                all_struct_pred.append(struct_pred_vals[0])

            # Track number of valid neighbors (after same-AA filtering happens in model)
            n_valid = ret_valid.sum(dim=1).float()
            all_n_valid_neighbors.append(n_valid)

    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    results = {
        'predictions_norm': torch.cat(all_pred).numpy(),
        'targets_norm': torch.cat(all_target).numpy(),
        'masks': torch.cat(all_mask).numpy(),
        'residue_codes': torch.cat(all_residue_codes).numpy(),
        'ss_codes': torch.cat(all_ss_codes).numpy(),
        'retrieved_distances': torch.cat(all_retrieved_distances).numpy(),
        'retrieved_valid': torch.cat(all_retrieved_valid).numpy(),
        'n_valid_neighbors': torch.cat(all_n_valid_neighbors).numpy(),
    }

    if all_direct_gate:
        results['direct_gate'] = torch.cat(all_direct_gate).numpy()
    if all_retrieval_gate:
        results['retrieval_gate'] = torch.cat(all_retrieval_gate).numpy()
    if all_struct_pred:
        results['struct_pred_norm'] = torch.cat(all_struct_pred).numpy()

    # Denormalize predictions and targets
    predictions = results['predictions_norm'].copy()
    targets = results['targets_norm'].copy()
    for i, col in enumerate(shift_cols):
        if col in stats:
            predictions[:, i] = predictions[:, i] * stats[col]['std'] + stats[col]['mean']
            targets[:, i] = targets[:, i] * stats[col]['std'] + stats[col]['mean']
    results['predictions'] = predictions
    results['targets'] = targets
    results['errors'] = np.abs(predictions - targets)

    if 'struct_pred_norm' in results:
        struct_pred = results['struct_pred_norm'].copy()
        for i, col in enumerate(shift_cols):
            if col in stats:
                struct_pred[:, i] = struct_pred[:, i] * stats[col]['std'] + stats[col]['mean']
        results['struct_pred'] = struct_pred
        results['struct_errors'] = np.abs(struct_pred - targets)

    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_per_amino_acid(results, shift_cols):
    """Per-amino-acid MAE breakdown."""
    idx_to_aa3 = {v: k for k, v in RESIDUE_TO_IDX.items()}
    per_aa = {}

    for aa_idx in range(len(STANDARD_RESIDUES)):
        aa_mask = results['residue_codes'] == aa_idx
        if aa_mask.sum() == 0:
            continue
        aa_name = idx_to_aa3.get(aa_idx, f'UNK_{aa_idx}')

        per_shift = {}
        for si, col in enumerate(shift_cols):
            shift_mask = results['masks'][:, si].astype(bool) & aa_mask
            if shift_mask.sum() > 0:
                per_shift[col] = {
                    'mae': float(np.mean(results['errors'][:, si][shift_mask])),
                    'count': int(shift_mask.sum()),
                }

        if per_shift:
            overall = np.mean([v['mae'] for v in per_shift.values()])
            per_aa[aa_name] = {
                'overall_mae': float(overall),
                'per_shift': per_shift,
                'count': int(aa_mask.sum()),
            }

    return per_aa


def analyze_per_secondary_structure(results, shift_cols):
    """Per-secondary-structure MAE breakdown."""
    per_ss = {}
    for ss_idx, ss_name in enumerate(SS_TYPES):
        ss_mask = results['ss_codes'] == ss_idx
        if ss_mask.sum() == 0:
            continue

        per_shift = {}
        for si, col in enumerate(shift_cols):
            shift_mask = results['masks'][:, si].astype(bool) & ss_mask
            if shift_mask.sum() > 0:
                per_shift[col] = {
                    'mae': float(np.mean(results['errors'][:, si][shift_mask])),
                    'count': int(shift_mask.sum()),
                }

        if per_shift:
            overall = np.mean([v['mae'] for v in per_shift.values()])
            per_ss[ss_name if ss_name.strip() else 'COIL'] = {
                'overall_mae': float(overall),
                'per_shift': per_shift,
                'count': int(ss_mask.sum()),
            }

    return per_ss


def analyze_gate_behavior(results, shift_cols):
    """Analyze how gates behave across different conditions."""
    analysis = {}

    if 'retrieval_gate' not in results:
        return analysis

    rg = results['retrieval_gate']
    dg = results['direct_gate']

    # Overall gate statistics
    analysis['retrieval_gate'] = {
        'mean': float(np.mean(rg)),
        'std': float(np.std(rg)),
        'median': float(np.median(rg)),
        'p10': float(np.percentile(rg, 10)),
        'p90': float(np.percentile(rg, 90)),
    }
    analysis['direct_gate'] = {
        'mean': float(np.mean(dg)),
        'std': float(np.std(dg)),
        'median': float(np.median(dg)),
        'p10': float(np.percentile(dg, 10)),
        'p90': float(np.percentile(dg, 90)),
    }

    # Gate values by number of valid neighbors
    n_valid = results['n_valid_neighbors']
    bins = [0, 1, 5, 10, 20, 32, 100]
    gate_by_neighbors = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        bin_mask = (n_valid >= lo) & (n_valid < hi)
        if bin_mask.sum() > 0:
            # Average over shifts for retrieval gate
            rg_mean = np.mean(rg[bin_mask])
            dg_mean = np.mean(dg[bin_mask])
            gate_by_neighbors[f'{lo}-{hi}'] = {
                'retrieval_gate_mean': float(rg_mean),
                'direct_gate_mean': float(dg_mean),
                'count': int(bin_mask.sum()),
            }
    analysis['gate_by_neighbor_count'] = gate_by_neighbors

    # Per-amino-acid gate values
    idx_to_aa3 = {v: k for k, v in RESIDUE_TO_IDX.items()}
    gate_by_aa = {}
    for aa_idx in range(len(STANDARD_RESIDUES)):
        aa_mask = results['residue_codes'] == aa_idx
        if aa_mask.sum() < 10:
            continue
        aa_name = idx_to_aa3.get(aa_idx, f'UNK_{aa_idx}')
        gate_by_aa[aa_name] = {
            'retrieval_gate_mean': float(np.mean(rg[aa_mask])),
            'direct_gate_mean': float(np.mean(dg[aa_mask])),
            'count': int(aa_mask.sum()),
        }
    analysis['gate_by_amino_acid'] = gate_by_aa

    # Per-shift gate values
    gate_by_shift = {}
    for si, col in enumerate(shift_cols):
        shift_mask = results['masks'][:, si].astype(bool)
        if shift_mask.sum() > 0:
            gate_by_shift[col] = {
                'retrieval_gate_mean': float(np.mean(rg[:, si][shift_mask])),
                'direct_gate_mean': float(np.mean(dg[:, si][shift_mask])),
                'count': int(shift_mask.sum()),
            }
    analysis['gate_by_shift'] = gate_by_shift

    return analysis


def analyze_retrieval_contribution(results, shift_cols):
    """Analyze how much retrieval helps vs structure-only."""
    analysis = {}

    if 'struct_pred' not in results:
        return analysis

    # Per-shift: model MAE vs structure-only MAE
    for si, col in enumerate(shift_cols):
        mask = results['masks'][:, si].astype(bool)
        if mask.sum() == 0:
            continue

        model_mae = float(np.mean(results['errors'][:, si][mask]))
        struct_mae = float(np.mean(results['struct_errors'][:, si][mask]))
        improvement = struct_mae - model_mae
        pct_improvement = (improvement / struct_mae * 100) if struct_mae > 0 else 0

        analysis[col] = {
            'model_mae': model_mae,
            'struct_only_mae': struct_mae,
            'improvement_ppm': float(improvement),
            'improvement_pct': float(pct_improvement),
            'count': int(mask.sum()),
        }

    # Overall
    all_mask = results['masks'].astype(bool)
    model_all = np.mean(results['errors'][all_mask])
    struct_all = np.mean(results['struct_errors'][all_mask])
    analysis['overall'] = {
        'model_mae': float(model_all),
        'struct_only_mae': float(struct_all),
        'improvement_ppm': float(struct_all - model_all),
        'improvement_pct': float((struct_all - model_all) / struct_all * 100),
    }

    # By retrieval quality
    ret_valid = results['retrieved_valid']
    n_valid = ret_valid.sum(axis=1)

    has_retrieval = n_valid > 0
    no_retrieval = n_valid == 0

    if has_retrieval.sum() > 0:
        model_w_ret = float(np.mean(results['errors'][has_retrieval][results['masks'][has_retrieval].astype(bool)]))
        struct_w_ret = float(np.mean(results['struct_errors'][has_retrieval][results['masks'][has_retrieval].astype(bool)]))
        analysis['with_retrieval'] = {
            'model_mae': model_w_ret,
            'struct_only_mae': struct_w_ret,
            'count': int(has_retrieval.sum()),
        }

    if no_retrieval.sum() > 0 and results['masks'][no_retrieval].astype(bool).sum() > 0:
        model_no_ret = float(np.mean(results['errors'][no_retrieval][results['masks'][no_retrieval].astype(bool)]))
        struct_no_ret = float(np.mean(results['struct_errors'][no_retrieval][results['masks'][no_retrieval].astype(bool)]))
        analysis['without_retrieval'] = {
            'model_mae': model_no_ret,
            'struct_only_mae': struct_no_ret,
            'count': int(no_retrieval.sum()),
        }

    return analysis


def analyze_error_patterns(results, shift_cols, stats):
    """Analyze error patterns - what makes predictions hard?"""
    analysis = {}

    # Error by true shift value (are extremes harder?)
    for si, col in enumerate(shift_cols):
        if col not in BACKBONE_SHIFTS:
            continue
        mask = results['masks'][:, si].astype(bool)
        if mask.sum() < 100:
            continue

        true_vals = results['targets'][:, si][mask]
        errs = results['errors'][:, si][mask]

        # Bin by percentile of true value
        percentiles = np.percentile(true_vals, [0, 10, 25, 50, 75, 90, 100])
        bins = {}
        for j in range(len(percentiles) - 1):
            lo, hi = percentiles[j], percentiles[j+1]
            bin_mask = (true_vals >= lo) & (true_vals < hi + 0.001)
            if bin_mask.sum() > 0:
                bins[f'{lo:.1f}-{hi:.1f}'] = {
                    'mae': float(np.mean(errs[bin_mask])),
                    'count': int(bin_mask.sum()),
                }
        analysis[col] = bins

    return analysis


def analyze_shift_coverage(results, shift_cols):
    """Analyze which shifts are most/least observed."""
    coverage = {}
    total = len(results['masks'])
    for si, col in enumerate(shift_cols):
        n_observed = int(results['masks'][:, si].sum())
        coverage[col] = {
            'n_observed': n_observed,
            'pct_observed': float(n_observed / total * 100),
            'is_backbone': col in BACKBONE_SHIFTS,
        }
    return coverage


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_per_aa_heatmap(per_aa, shift_cols, plot_dir):
    """Heatmap of MAE by amino acid and shift type."""
    backbone_cols = sorted([c for c in shift_cols if c in BACKBONE_SHIFTS])
    aa_names = sorted(per_aa.keys())

    matrix = np.full((len(aa_names), len(backbone_cols)), np.nan)
    for i, aa in enumerate(aa_names):
        for j, col in enumerate(backbone_cols):
            if col in per_aa[aa]['per_shift']:
                matrix[i, j] = per_aa[aa]['per_shift'][col]['mae']

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(backbone_cols)))
    ax.set_xticklabels([c.replace('_shift', '').upper() for c in backbone_cols], rotation=45, ha='right')
    ax.set_yticks(range(len(aa_names)))
    aa_labels = [f"{aa} ({AA_3_TO_1.get(aa, '?')})" for aa in aa_names]
    ax.set_yticklabels(aa_labels)

    # Add text annotations
    for i in range(len(aa_names)):
        for j in range(len(backbone_cols)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if matrix[i, j] > np.nanmedian(matrix) else 'black')

    plt.colorbar(im, label='MAE (ppm)')
    ax.set_title('Per-Amino-Acid Per-Shift MAE', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'per_aa_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_gate_distributions(results, shift_cols, plot_dir):
    """Plot gate activation distributions."""
    if 'retrieval_gate' not in results:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Retrieval gate overall distribution
    ax = axes[0, 0]
    rg_flat = results['retrieval_gate'].flatten()
    ax.hist(rg_flat, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax.axvline(np.mean(rg_flat), color='red', linestyle='--', label=f'Mean: {np.mean(rg_flat):.3f}')
    ax.set_xlabel('Retrieval Gate Value')
    ax.set_ylabel('Density')
    ax.set_title('Retrieval Gate Distribution')
    ax.legend()

    # Direct gate overall distribution
    ax = axes[0, 1]
    dg_flat = results['direct_gate'].flatten()
    ax.hist(dg_flat, bins=50, alpha=0.7, color='coral', edgecolor='black', density=True)
    ax.axvline(np.mean(dg_flat), color='red', linestyle='--', label=f'Mean: {np.mean(dg_flat):.3f}')
    ax.set_xlabel('Direct Gate Value')
    ax.set_ylabel('Density')
    ax.set_title('Direct Gate Distribution')
    ax.legend()

    # Gate vs number of valid neighbors
    ax = axes[0, 2]
    n_valid = results['n_valid_neighbors']
    # Mean retrieval gate per sample
    rg_mean_per_sample = np.mean(results['retrieval_gate'], axis=1)
    ax.scatter(n_valid, rg_mean_per_sample, alpha=0.05, s=3, rasterized=True)
    # Bin averages
    unique_n = sorted(set(n_valid.astype(int)))
    bin_means = []
    bin_ns = []
    for n in unique_n:
        mask = n_valid.astype(int) == n
        if mask.sum() > 10:
            bin_ns.append(n)
            bin_means.append(np.mean(rg_mean_per_sample[mask]))
    ax.plot(bin_ns, bin_means, 'r-', linewidth=2, label='Binned mean')
    ax.set_xlabel('Number of Valid Neighbors')
    ax.set_ylabel('Mean Retrieval Gate')
    ax.set_title('Retrieval Gate vs Neighbor Count')
    ax.legend()

    # Per-shift retrieval gate
    ax = axes[1, 0]
    backbone_cols = sorted([c for c in shift_cols if c in BACKBONE_SHIFTS])
    rg_per_shift = []
    shift_names = []
    for si, col in enumerate(shift_cols):
        if col in BACKBONE_SHIFTS:
            mask = results['masks'][:, si].astype(bool)
            if mask.sum() > 0:
                rg_per_shift.append(results['retrieval_gate'][:, si][mask])
                shift_names.append(col.replace('_shift', '').upper())
    if rg_per_shift:
        ax.boxplot(rg_per_shift, tick_labels=shift_names, showfliers=False)
        ax.set_ylabel('Retrieval Gate Value')
        ax.set_title('Retrieval Gate by Shift Type')
        ax.tick_params(axis='x', rotation=45)

    # Per-shift direct gate
    ax = axes[1, 1]
    dg_per_shift = []
    shift_names2 = []
    for si, col in enumerate(shift_cols):
        if col in BACKBONE_SHIFTS:
            mask = results['masks'][:, si].astype(bool)
            if mask.sum() > 0:
                dg_per_shift.append(results['direct_gate'][:, si][mask])
                shift_names2.append(col.replace('_shift', '').upper())
    if dg_per_shift:
        ax.boxplot(dg_per_shift, tick_labels=shift_names2, showfliers=False)
        ax.set_ylabel('Direct Gate Value')
        ax.set_title('Direct Gate by Shift Type')
        ax.tick_params(axis='x', rotation=45)

    # Retrieval gate vs error
    ax = axes[1, 2]
    mask_all = results['masks'].astype(bool)
    sample_rg = np.mean(results['retrieval_gate'], axis=1)
    sample_mae = np.zeros(len(results['predictions']))
    for i in range(len(results['predictions'])):
        if mask_all[i].any():
            sample_mae[i] = np.mean(results['errors'][i][mask_all[i]])
    valid = sample_mae > 0
    ax.scatter(sample_rg[valid], sample_mae[valid], alpha=0.05, s=3, rasterized=True)
    corr = np.corrcoef(sample_rg[valid], sample_mae[valid])[0, 1] if valid.sum() > 1 else 0
    ax.set_xlabel('Mean Retrieval Gate')
    ax.set_ylabel('Sample MAE (ppm)')
    ax.set_title(f'Retrieval Gate vs Error (r={corr:.3f})')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'gate_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_retrieval_ablation(retrieval_analysis, plot_dir):
    """Bar chart: model vs structure-only MAE per shift."""
    if not retrieval_analysis:
        return

    cols = sorted([k for k in retrieval_analysis.keys()
                   if k not in ('overall', 'with_retrieval', 'without_retrieval')])
    backbone = [c for c in cols if c in BACKBONE_SHIFTS]
    sidechain = [c for c in cols if c not in BACKBONE_SHIFTS]

    for subset, label in [(backbone, 'backbone'), (sidechain, 'sidechain')]:
        if not subset:
            continue
        names = [c.replace('_shift', '').upper() for c in subset]
        model_mae = [retrieval_analysis[c]['model_mae'] for c in subset]
        struct_mae = [retrieval_analysis[c]['struct_only_mae'] for c in subset]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, model_mae, width, label='Full Model', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, struct_mae, width, label='Structure Only', color='coral', alpha=0.8)

        # Add improvement labels
        for i, c in enumerate(subset):
            imp = retrieval_analysis[c]['improvement_pct']
            ax.text(i, max(model_mae[i], struct_mae[i]) + 0.02,
                    f'{imp:+.1f}%', ha='center', fontsize=8, color='green' if imp > 0 else 'red')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('MAE (ppm)')
        ax.set_title(f'Retrieval Contribution: {label.title()} Shifts')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'retrieval_ablation_{label}.png'), dpi=200, bbox_inches='tight')
        plt.close()


def plot_per_ss_bars(per_ss, plot_dir):
    """Bar chart of MAE by secondary structure."""
    ss_names = sorted(per_ss.keys(), key=lambda x: per_ss[x]['overall_mae'])
    maes = [per_ss[s]['overall_mae'] for s in ss_names]
    counts = [per_ss[s]['count'] for s in ss_names]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(range(len(ss_names)), maes, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(ss_names)))
    ax1.set_xticklabels(ss_names, fontsize=11)
    ax1.set_ylabel('Overall MAE (ppm)', fontsize=12)
    ax1.set_title('Prediction Quality by Secondary Structure', fontsize=14)

    ax2 = ax1.twinx()
    ax2.plot(range(len(ss_names)), counts, 'ro-', alpha=0.7, label='Count')
    ax2.set_ylabel('Sample Count', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'per_ss_mae.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_error_vs_true_value(error_patterns, plot_dir):
    """Plot how errors vary with true shift value."""
    backbone_cols = [c for c in error_patterns.keys()]
    n_plots = len(backbone_cols)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(backbone_cols[:6]):
        ax = axes[idx]
        bins = error_patterns[col]
        centers = []
        maes = []
        counts = []
        for bin_name, data in sorted(bins.items()):
            parts = bin_name.split('-')
            center = (float(parts[0]) + float(parts[1])) / 2
            centers.append(center)
            maes.append(data['mae'])
            counts.append(data['count'])

        ax.bar(range(len(centers)), maes, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(centers)))
        ax.set_xticklabels([f'{c:.0f}' for c in centers], fontsize=8, rotation=45)
        name = col.replace('_shift', '').upper()
        ax.set_title(f'{name} Shift')
        ax.set_xlabel('True Value (ppm)')
        ax.set_ylabel('MAE (ppm)')
        ax.grid(True, alpha=0.3, axis='y')

    for i in range(len(backbone_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Error vs True Shift Value (are extremes harder?)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'error_vs_true_value.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_prediction_scatter(results, shift_cols, stats, plot_dir):
    """Scatter plots of predicted vs actual for backbone shifts."""
    backbone_cols = [(i, c) for i, c in enumerate(shift_cols) if c in BACKBONE_SHIFTS]
    n_plots = min(6, len(backbone_cols))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, (si, col) in enumerate(backbone_cols[:n_plots]):
        ax = axes[ax_idx]
        mask = results['masks'][:, si].astype(bool)
        if mask.sum() == 0:
            continue

        preds = results['predictions'][:, si][mask]
        targs = results['targets'][:, si][mask]

        # Subsample
        if len(preds) > 15000:
            idx = np.random.choice(len(preds), 15000, replace=False)
            preds_plot, targs_plot = preds[idx], targs[idx]
        else:
            preds_plot, targs_plot = preds, targs

        ax.scatter(targs_plot, preds_plot, alpha=0.08, s=3, rasterized=True, color='steelblue')
        lims = [min(targs.min(), preds.min()), max(targs.max(), preds.max())]
        ax.plot(lims, lims, 'r--', linewidth=2)

        corr = np.corrcoef(targs, preds)[0, 1]
        mae = np.mean(np.abs(preds - targs))
        rmse = np.sqrt(np.mean((preds - targs)**2))
        name = col.replace('_shift', '').upper()
        ax.set_title(f'{name} (r={corr:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f})', fontsize=11)
        ax.set_xlabel('True (ppm)')
        ax.set_ylabel('Predicted (ppm)')
        ax.grid(True, alpha=0.3)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Predicted vs Actual Chemical Shifts', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'prediction_scatter.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(results, shift_cols, stats, per_aa, per_ss, gate_analysis, retrieval_analysis, plot_dir):
    """Create a single-page summary dashboard."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # 1. Overall metrics table
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    metrics = []
    for si, col in enumerate(shift_cols):
        if col not in BACKBONE_SHIFTS:
            continue
        mask = results['masks'][:, si].astype(bool)
        if mask.sum() == 0:
            continue
        mae = np.mean(results['errors'][:, si][mask])
        name = col.replace('_shift', '').upper()
        metrics.append([name, f'{mae:.3f}', f'{int(mask.sum()):,}'])

    if metrics:
        table = ax.table(cellText=metrics, colLabels=['Shift', 'MAE', 'Count'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    ax.set_title('Backbone Shift MAE (ppm)', fontsize=12, fontweight='bold')

    # 2. Top/bottom 5 amino acids
    ax = fig.add_subplot(gs[0, 1])
    aa_sorted = sorted(per_aa.items(), key=lambda x: x[1]['overall_mae'])
    best5 = aa_sorted[:5]
    worst5 = aa_sorted[-5:]
    items = best5 + worst5
    names = [f"{aa} ({AA_3_TO_1.get(aa, '?')})" for aa, _ in items]
    vals = [v['overall_mae'] for _, v in items]
    colors = ['green'] * 5 + ['red'] * 5
    ax.barh(range(len(names)), vals, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('MAE (ppm)')
    ax.set_title('Best/Worst 5 Amino Acids', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # 3. Secondary structure
    ax = fig.add_subplot(gs[0, 2])
    ss_sorted = sorted(per_ss.items(), key=lambda x: x[1]['overall_mae'])
    ss_names = [s for s, _ in ss_sorted]
    ss_maes = [v['overall_mae'] for _, v in ss_sorted]
    ax.bar(range(len(ss_names)), ss_maes, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(ss_names)))
    ax.set_xticklabels(ss_names, fontsize=10)
    ax.set_ylabel('MAE (ppm)')
    ax.set_title('MAE by Secondary Structure', fontsize=12, fontweight='bold')

    # 4. Retrieval ablation summary
    ax = fig.add_subplot(gs[0, 3])
    if retrieval_analysis and 'overall' in retrieval_analysis:
        labels = ['Full Model', 'Structure Only']
        values = [retrieval_analysis['overall']['model_mae'],
                  retrieval_analysis['overall']['struct_only_mae']]
        ax.bar(labels, values, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
        imp = retrieval_analysis['overall']['improvement_pct']
        ax.set_title(f'Retrieval Helps by {imp:.1f}%', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE (ppm)')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No ablation data', ha='center', va='center')

    # 5-6. Prediction scatter for CA and N (most important)
    for plot_idx, shift_name in enumerate(['ca_shift', 'n_shift']):
        ax = fig.add_subplot(gs[1, plot_idx])
        si = shift_cols.index(shift_name) if shift_name in shift_cols else -1
        if si >= 0:
            mask = results['masks'][:, si].astype(bool)
            if mask.sum() > 0:
                preds = results['predictions'][:, si][mask]
                targs = results['targets'][:, si][mask]
                if len(preds) > 10000:
                    idx = np.random.choice(len(preds), 10000, replace=False)
                    preds, targs = preds[idx], targs[idx]
                ax.scatter(targs, preds, alpha=0.1, s=2, rasterized=True)
                lims = [min(targs.min(), preds.min()), max(targs.max(), preds.max())]
                ax.plot(lims, lims, 'r--', linewidth=2)
                corr = np.corrcoef(targs, preds)[0, 1]
                ax.set_title(f'{shift_name.replace("_shift","").upper()} (r={corr:.3f})', fontsize=11)
                ax.set_xlabel('True (ppm)')
                ax.set_ylabel('Predicted (ppm)')

    # 7. Gate distribution
    ax = fig.add_subplot(gs[1, 2])
    if 'retrieval_gate' in results:
        rg = results['retrieval_gate'].flatten()
        ax.hist(rg, bins=40, alpha=0.7, color='steelblue', density=True)
        ax.axvline(np.mean(rg), color='red', linestyle='--')
        ax.set_title(f'Retrieval Gate (mean={np.mean(rg):.3f})', fontsize=11)
        ax.set_xlabel('Gate Value')

    # 8. Shift coverage
    ax = fig.add_subplot(gs[1, 3])
    coverage_data = analyze_shift_coverage(results, shift_cols)
    backbone_cov = {k: v for k, v in coverage_data.items() if v['is_backbone']}
    names = sorted(backbone_cov.keys())
    pcts = [backbone_cov[n]['pct_observed'] for n in names]
    labels = [n.replace('_shift', '').upper() for n in names]
    ax.barh(range(len(labels)), pcts, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('% Observed')
    ax.set_title('Backbone Shift Coverage', fontsize=11, fontweight='bold')

    # 9-12. Error distribution for backbone shifts
    backbone_shifts = sorted([c for c in shift_cols if c in BACKBONE_SHIFTS])
    for idx, col in enumerate(backbone_shifts[:4]):
        ax = fig.add_subplot(gs[2, idx])
        si = shift_cols.index(col)
        mask = results['masks'][:, si].astype(bool)
        if mask.sum() > 0:
            errs = results['errors'][:, si][mask]
            ax.hist(errs, bins=50, alpha=0.7, color='steelblue', density=True, edgecolor='black')
            ax.axvline(np.mean(errs), color='red', linestyle='--', label=f'Mean: {np.mean(errs):.3f}')
            ax.axvline(np.median(errs), color='green', linestyle='--', label=f'Med: {np.median(errs):.3f}')
            ax.set_xlim(0, np.percentile(errs, 99))
            name = col.replace('_shift', '').upper()
            ax.set_title(f'{name} Error Dist', fontsize=10)
            ax.legend(fontsize=7)

    plt.suptitle('Chemical Shift Prediction: Interpretability Dashboard', fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(plot_dir, 'summary_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint (default: auto-detect best)')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--cache_dir', type=str, default='data/cache')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit batches for quick testing')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Auto-detect checkpoint
    if args.checkpoint is None:
        ckpt_dir = 'data/checkpoints'
        best = os.path.join(ckpt_dir, f'best_retrieval_fold{args.fold}.pt')
        if os.path.exists(best):
            args.checkpoint = best
        else:
            # Find latest epoch checkpoint
            ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith(f'checkpoint_fold{args.fold}_epoch')])
            if ckpts:
                args.checkpoint = os.path.join(ckpt_dir, ckpts[-1])
            else:
                print("ERROR: No checkpoint found!")
                sys.exit(1)

    print("=" * 70)
    print("INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    print(f"Fold:       {args.fold}")

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(args.checkpoint, device)
    stats = checkpoint.get('stats', {})
    shift_cols = checkpoint.get('shift_cols', [])
    k_retrieved = checkpoint.get('k_retrieved', K_RETRIEVED)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Epoch: {epoch}")
    print(f"  Shifts: {shift_cols}")

    # Load test dataset
    print("\nLoading test dataset...")
    test_cache = os.path.join(args.cache_dir, f'fold_{args.fold}')
    if not CachedRetrievalDataset.exists(test_cache):
        print(f"ERROR: Cache not found at {test_cache}")
        sys.exit(1)

    dataset = CachedRetrievalDataset.load(
        test_cache, len(shift_cols), k_retrieved, stats=stats, shift_cols=shift_cols)
    print(f"  Loaded {len(dataset):,} test samples")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device == 'cuda'))

    # Collect predictions with gate activations
    print("\nCollecting predictions with gate analysis...")
    results = collect_predictions_with_gates(
        model, loader, device, stats, shift_cols, max_batches=args.max_batches)
    print(f"  Collected {len(results['predictions']):,} predictions")

    # Free GPU memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Run analyses
    print("\nAnalyzing per-amino-acid performance...")
    per_aa = analyze_per_amino_acid(results, shift_cols)

    print("Analyzing per-secondary-structure performance...")
    per_ss = analyze_per_secondary_structure(results, shift_cols)

    print("Analyzing gate behavior...")
    gate_analysis = analyze_gate_behavior(results, shift_cols)

    print("Analyzing retrieval contribution (ablation)...")
    retrieval_analysis = analyze_retrieval_contribution(results, shift_cols)

    print("Analyzing error patterns...")
    error_patterns = analyze_error_patterns(results, shift_cols, stats)

    print("Analyzing shift coverage...")
    coverage = analyze_shift_coverage(results, shift_cols)

    # Generate plots
    print("\nGenerating plots...")
    plot_per_aa_heatmap(per_aa, shift_cols, PLOT_DIR)
    print("  - per_aa_heatmap.png")

    plot_gate_distributions(results, shift_cols, PLOT_DIR)
    print("  - gate_analysis.png")

    plot_retrieval_ablation(retrieval_analysis, PLOT_DIR)
    print("  - retrieval_ablation_backbone.png")
    print("  - retrieval_ablation_sidechain.png")

    plot_per_ss_bars(per_ss, PLOT_DIR)
    print("  - per_ss_mae.png")

    plot_error_vs_true_value(error_patterns, PLOT_DIR)
    print("  - error_vs_true_value.png")

    plot_prediction_scatter(results, shift_cols, stats, PLOT_DIR)
    print("  - prediction_scatter.png")

    plot_summary_dashboard(results, shift_cols, stats, per_aa, per_ss,
                           gate_analysis, retrieval_analysis, PLOT_DIR)
    print("  - summary_dashboard.png")

    # Save JSON results
    all_results = {
        'checkpoint': args.checkpoint,
        'epoch': epoch,
        'fold': args.fold,
        'n_test_samples': len(results['predictions']),
        'per_amino_acid': per_aa,
        'per_secondary_structure': per_ss,
        'gate_analysis': gate_analysis,
        'retrieval_contribution': retrieval_analysis,
        'error_patterns': error_patterns,
        'shift_coverage': coverage,
    }

    json_path = os.path.join(RESULTS_DIR, 'interpretability_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if gate_analysis and 'retrieval_gate' in gate_analysis:
        rg = gate_analysis['retrieval_gate']
        print(f"\nRetrieval Gate: mean={rg['mean']:.3f}, median={rg['median']:.3f}")
        print(f"  Model uses retrieval {rg['mean']*100:.1f}% on average")

    if gate_analysis and 'direct_gate' in gate_analysis:
        dg = gate_analysis['direct_gate']
        print(f"\nDirect Gate: mean={dg['mean']:.3f}")
        print(f"  Direct transfer used {dg['mean']*100:.1f}% vs attention {(1-dg['mean'])*100:.1f}%")

    if retrieval_analysis and 'overall' in retrieval_analysis:
        ra = retrieval_analysis['overall']
        print(f"\nRetrieval Ablation:")
        print(f"  Full model MAE:    {ra['model_mae']:.4f} ppm")
        print(f"  Structure only:    {ra['struct_only_mae']:.4f} ppm")
        print(f"  Retrieval helps:   {ra['improvement_ppm']:.4f} ppm ({ra['improvement_pct']:.1f}%)")

    # Best and worst amino acids
    aa_sorted = sorted(per_aa.items(), key=lambda x: x[1]['overall_mae'])
    print(f"\nBest amino acids:  {', '.join(f'{aa}({v['overall_mae']:.3f})' for aa, v in aa_sorted[:3])}")
    print(f"Worst amino acids: {', '.join(f'{aa}({v['overall_mae']:.3f})' for aa, v in aa_sorted[-3:])}")

    print("\n" + "=" * 70)
    print(f"All plots saved to {PLOT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
