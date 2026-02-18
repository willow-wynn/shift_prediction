#!/usr/bin/env python3
"""
Evaluation Script for Shift Imputation Model.

Computes per-shift MAE, RMSE, R^2 and compares against baselines:
1. Structure-only model (existing 06_train.py model)
2. Mean predictor
3. Random coil predictor

Usage:
    python 09_eval_imputation.py --model checkpoints/best_imputation_fold1.pt \\
        --data_dir data --fold 1 --output_dir eval_imputation_results

    # With structure-only baseline comparison
    python 09_eval_imputation.py --model checkpoints/best_imputation_fold1.pt \\
        --data_dir data --fold 1 \\
        --structure_model checkpoints/best_retrieval_fold1.pt
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

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
    N_RESIDUE_TYPES, N_SS_TYPES, N_MISMATCH_TYPES,
    DSSP_COLS, K_RETRIEVED, STANDARD_RESIDUES, AA_3_TO_1,
    OUTLIER_STD_THRESHOLD, CONTEXT_WINDOW, K_SPATIAL_NEIGHBORS,
)
from dataset import (
    CachedRetrievalDataset,
    parse_shift_columns,
    get_dssp_columns,
)
from imputation_model import ShiftImputationModel
from imputation_dataset import load_imputation_dataset
from random_coil import RC_SHIFTS

BACKBONE_ATOMS = {'C', 'CA', 'CB', 'H', 'HA', 'N'}


# ============================================================================
# Model Loading
# ============================================================================

def load_imputation_model(checkpoint_path, device):
    """Load trained imputation model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        return None, None

    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Remove _orig_mod. prefix from compiled model keys
    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.replace('_orig_mod.', '')] = v

    stats = checkpoint.get('stats', None)
    shift_cols = checkpoint.get('shift_cols', None)
    k_retrieved = checkpoint.get('k_retrieved', K_RETRIEVED)
    n_physics = checkpoint.get('n_physics', 28)
    n_atom_types = checkpoint.get('n_atom_types', None)
    dssp_cols = checkpoint.get('dssp_cols', [])

    # Auto-detect from state dict if not in checkpoint
    if n_atom_types is None:
        n_atom_types = clean_sd['distance_attention.atom_embed.weight'].shape[0] - 1

    n_shifts = clean_sd['shift_type_proj.0.weight'].shape[1]
    n_dssp = clean_sd['dssp_proj.weight'].shape[1] if 'dssp_proj.weight' in clean_sd else 0

    if 'physics_encoder.mlp.0.weight' in clean_sd:
        n_physics = clean_sd['physics_encoder.mlp.0.weight'].shape[1]

    # Detect CNN channels
    struct_cnn_channels = []
    for i in range(0, 20, 2):
        key = f'struct_cnn.{i}.conv1.weight'
        if key in clean_sd:
            struct_cnn_channels.append(clean_sd[key].shape[0])

    # Detect shift context channels
    shift_context_channels = []
    for i in range(0, 20, 2):
        key = f'shift_context_encoder.cnn.{i}.conv1.weight'
        if key in clean_sd:
            shift_context_channels.append(clean_sd[key].shape[0])

    spatial_hidden = clean_sd.get(
        'spatial_attention.fallback_embed', torch.zeros(192)).shape[0]
    retrieval_hidden = clean_sd.get(
        'retrieval.fallback_context', torch.zeros(192)).shape[0]

    use_random_coil = 'retrieval.rc_table' in clean_sd

    print(f"  Auto-detected: n_atoms={n_atom_types}, n_shifts={n_shifts}, "
          f"n_dssp={n_dssp}, n_physics={n_physics}")
    print(f"  struct_cnn={struct_cnn_channels}, shift_ctx={shift_context_channels}")
    print(f"  spatial={spatial_hidden}, retrieval={retrieval_hidden}, RC={use_random_coil}")

    model = ShiftImputationModel(
        n_atom_types=n_atom_types,
        n_shifts=n_shifts,
        n_physics=n_physics,
        n_dssp=n_dssp,
        struct_cnn_channels=struct_cnn_channels or None,
        shift_context_channels=shift_context_channels or None,
        spatial_hidden=spatial_hidden,
        retrieval_hidden=retrieval_hidden,
        use_random_coil=use_random_coil,
        shift_cols=shift_cols,
    ).to(device)

    model.load_state_dict(clean_sd)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model: {n_params:,} parameters")

    return model, {
        'stats': stats,
        'shift_cols': shift_cols,
        'k_retrieved': k_retrieved,
        'n_physics': n_physics,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'dssp_cols': dssp_cols,
    }


# ============================================================================
# Prediction Collection
# ============================================================================

@torch.no_grad()
def collect_predictions(model, loader, device, stats, shift_cols):
    """Collect all predictions grouped by shift type."""
    model.eval()

    per_shift_preds = {col: [] for col in shift_cols}
    per_shift_targets = {col: [] for col in shift_cols}
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="  Collecting predictions", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch.pop('target_value')
        target_shift_idx = batch.pop('target_shift_idx')

        ctx = autocast('cuda') if device == 'cuda' else nullcontext()
        with ctx:
            pred = model(**batch)

        loss = F.huber_loss(pred, target, reduction='mean', delta=0.5)
        bs = target.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        idx_np = target_shift_idx.cpu().numpy()

        for i in range(len(idx_np)):
            col = shift_cols[idx_np[i]]
            per_shift_preds[col].append(pred_np[i])
            per_shift_targets[col].append(target_np[i])

    avg_loss = total_loss / total_count if total_count > 0 else 0.0

    # Denormalize
    per_shift_results = {}
    for col in shift_cols:
        if col in stats and per_shift_preds[col]:
            preds = np.array(per_shift_preds[col])
            targs = np.array(per_shift_targets[col])
            pred_denorm = preds * stats[col]['std'] + stats[col]['mean']
            targ_denorm = targs * stats[col]['std'] + stats[col]['mean']
            errors = np.abs(pred_denorm - targ_denorm)

            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean((pred_denorm - targ_denorm) ** 2)))
            ss_res = np.sum((targ_denorm - pred_denorm) ** 2)
            ss_tot = np.sum((targ_denorm - np.mean(targ_denorm)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            per_shift_results[col] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'count': len(preds),
                'predictions_denorm': pred_denorm,
                'targets_denorm': targ_denorm,
            }

    return {
        'loss': avg_loss,
        'per_shift': per_shift_results,
        'total_count': total_count,
    }


# ============================================================================
# Baseline Computations
# ============================================================================

def compute_baselines(per_shift_results, stats, shift_cols):
    """Compute mean predictor and random coil baseline metrics."""
    baselines = {}

    # Mean predictor
    mean_mae = {}
    for col in shift_cols:
        if col in per_shift_results and col in stats:
            targs = per_shift_results[col]['targets_denorm']
            mean_pred = stats[col]['mean']
            mean_mae[col] = float(np.mean(np.abs(targs - mean_pred)))
    baselines['mean_predictor'] = {
        'per_shift_mae': mean_mae,
        'overall_mae': float(np.mean(list(mean_mae.values()))) if mean_mae else 0.0,
    }

    # Random coil predictor
    # We don't have per-sample residue codes in the imputation dataset results,
    # so we use population-level RC MAE from the dataset stats
    # This is approximate but sufficient for comparison
    rc_mae = {}
    idx_to_aa1 = {}
    from config import RESIDUE_TO_IDX
    for res3 in STANDARD_RESIDUES:
        idx = RESIDUE_TO_IDX.get(res3, -1)
        aa1 = AA_3_TO_1.get(res3, None)
        if idx >= 0 and aa1:
            idx_to_aa1[idx] = aa1

    # For each shift, compute average |true - RC| using population stats
    for col in shift_cols:
        if col not in per_shift_results:
            continue
        atom = col.replace('_shift', '').upper()
        # Compute weighted RC shift across all residue types
        rc_vals = []
        for aa1, shifts in RC_SHIFTS.items():
            if atom in shifts and shifts[atom] is not None:
                rc_vals.append(shifts[atom])
        if rc_vals and col in stats:
            # Use training mean as proxy for population mean RC
            targs = per_shift_results[col]['targets_denorm']
            mean_rc = np.mean(rc_vals)
            rc_mae[col] = float(np.mean(np.abs(targs - mean_rc)))

    baselines['random_coil_predictor'] = {
        'per_shift_mae': rc_mae,
        'overall_mae': float(np.mean(list(rc_mae.values()))) if rc_mae else 0.0,
    }

    return baselines


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate shift imputation model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to imputation model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='eval_imputation_results')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--structure_model', type=str, default=None,
                        help='Path to structure-only model for comparison')
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.data_dir, 'cache')

    # Device
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
    print("SHIFT IMPUTATION MODEL EVALUATION")
    print("=" * 70)
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.model}")
    print(f"Fold:       {args.fold}")

    # ========== Load model ==========
    print("\nLoading imputation model...")
    model, ckpt_info = load_imputation_model(args.model, device)
    if model is None:
        sys.exit(1)

    stats = ckpt_info['stats']
    shift_cols = ckpt_info['shift_cols']
    k_retrieved = ckpt_info['k_retrieved']
    dssp_cols = ckpt_info.get('dssp_cols', [])

    if stats is None or shift_cols is None:
        print("ERROR: stats/shift_cols not in checkpoint")
        sys.exit(1)

    n_shifts = len(shift_cols)

    # ========== Load dataset ==========
    print("\nLoading test dataset...")
    test_base_cache = os.path.join(args.cache_dir, f'fold{args.fold}_test')
    if not CachedRetrievalDataset.exists(test_base_cache):
        print(f"ERROR: Base test cache not found at {test_base_cache}")
        sys.exit(1)

    test_base = CachedRetrievalDataset.load(
        test_base_cache, n_shifts, k_retrieved,
        stats=stats, shift_cols=shift_cols,
    )

    test_imp_cache = os.path.join(args.cache_dir, f'fold{args.fold}_test', 'imputation')
    test_dataset = load_imputation_dataset(
        test_base, test_imp_cache, n_shifts, context_window=CONTEXT_WINDOW,
    )
    print(f"  Test samples: {len(test_dataset):,}")

    loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == 'cuda'),
    )

    # ========== Collect predictions ==========
    print("\nRunning evaluation...")
    results = collect_predictions(model, loader, device, stats, shift_cols)

    # ========== Print results ==========
    print("\n" + "=" * 70)
    print("IMPUTATION MODEL RESULTS")
    print("=" * 70)
    print(f"Huber Loss: {results['loss']:.4f}")

    per_shift = results['per_shift']
    all_mae = [m['mae'] for m in per_shift.values()]
    overall_mae = float(np.mean(all_mae)) if all_mae else 0.0
    print(f"Overall MAE: {overall_mae:.4f} ppm")

    print(f"\nPer-shift (MAE / RMSE / R^2):")
    for col in sorted(per_shift.keys(), key=lambda c: per_shift[c]['mae']):
        m = per_shift[col]
        name = col.replace('_shift', '').upper()
        is_bb = "BB" if name in BACKBONE_ATOMS else "SC"
        print(f"  {name:6s} [{is_bb}]: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
              f"R2={m['r2']:.3f}  (n={m['count']:,})")

    # Backbone vs sidechain
    bb_mae = [per_shift[c]['mae'] for c in per_shift
              if c.replace('_shift', '').upper() in BACKBONE_ATOMS]
    sc_mae = [per_shift[c]['mae'] for c in per_shift
              if c.replace('_shift', '').upper() not in BACKBONE_ATOMS]
    if bb_mae:
        print(f"\nBackbone mean MAE:  {np.mean(bb_mae):.4f} ppm ({len(bb_mae)} types)")
    if sc_mae:
        print(f"Sidechain mean MAE: {np.mean(sc_mae):.4f} ppm ({len(sc_mae)} types)")

    # ========== Baselines ==========
    print("\nBaseline comparisons...")
    baselines = compute_baselines(per_shift, stats, shift_cols)

    print(f"\n  Imputation Model MAE:   {overall_mae:.4f} ppm")
    print(f"  Mean Predictor MAE:     {baselines['mean_predictor']['overall_mae']:.4f} ppm")
    print(f"  Random Coil MAE:        {baselines['random_coil_predictor']['overall_mae']:.4f} ppm")

    if baselines['mean_predictor']['overall_mae'] > 0:
        imp = baselines['mean_predictor']['overall_mae'] - overall_mae
        print(f"  Improvement over mean:  {imp:.4f} ppm "
              f"({100*imp/baselines['mean_predictor']['overall_mae']:.1f}%)")

    # ========== Structure-only comparison ==========
    if args.structure_model:
        print(f"\nLoading structure-only model for comparison: {args.structure_model}")
        try:
            from model import ShiftPredictorWithRetrieval
            struct_ckpt = torch.load(args.structure_model, map_location=device, weights_only=False)
            struct_mae = struct_ckpt.get('per_shift_mae', struct_ckpt.get('mae', None))
            if isinstance(struct_mae, dict):
                struct_overall = float(np.mean(list(struct_mae.values())))
                print(f"\n  Structure-only model MAE: {struct_overall:.4f} ppm")
                improvement = struct_overall - overall_mae
                print(f"  Imputation improvement:   {improvement:.4f} ppm "
                      f"({100*improvement/struct_overall:.1f}%)")

                print(f"\n  Per-shift comparison:")
                for col in sorted(shift_cols):
                    if col in per_shift and col in struct_mae:
                        imp_mae = per_shift[col]['mae']
                        str_mae = struct_mae[col]
                        delta = str_mae - imp_mae
                        print(f"    {col:20s}: struct={str_mae:.3f}  "
                              f"imput={imp_mae:.3f}  delta={delta:+.3f}")
            elif isinstance(struct_mae, (int, float)):
                print(f"  Structure-only MAE: {struct_mae:.4f}")
                print(f"  Imputation MAE:     {overall_mae:.4f}")
                print(f"  Improvement:        {struct_mae - overall_mae:.4f}")
        except Exception as e:
            print(f"  Warning: Could not load structure model: {e}")

    # ========== Save results ==========
    os.makedirs(args.output_dir, exist_ok=True)

    # Strip numpy arrays for JSON serialization
    json_per_shift = {}
    for col, m in per_shift.items():
        json_per_shift[col] = {
            'mae': m['mae'], 'rmse': m['rmse'], 'r2': m['r2'], 'count': m['count'],
        }

    all_results = {
        'overall_mae': overall_mae,
        'loss': results['loss'],
        'per_shift': json_per_shift,
        'baselines': baselines,
        'provenance': {
            'checkpoint': os.path.abspath(args.model),
            'fold': args.fold,
            'epoch': ckpt_info.get('epoch', 'unknown'),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_test_samples': results['total_count'],
        },
    }

    json_path = os.path.join(args.output_dir, f'imputation_eval_fold{args.fold}.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    # CSV
    csv_rows = []
    for col, m in json_per_shift.items():
        row = {'shift_type': col, **m}
        row['baseline_mean_mae'] = baselines['mean_predictor']['per_shift_mae'].get(col)
        row['baseline_rc_mae'] = baselines['random_coil_predictor']['per_shift_mae'].get(col)
        csv_rows.append(row)

    csv_path = os.path.join(args.output_dir, f'imputation_per_shift_fold{args.fold}.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Per-shift CSV saved to {csv_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Overall MAE: {overall_mae:.4f} ppm")
    print(f"  Results: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
