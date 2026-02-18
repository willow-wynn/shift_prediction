#!/usr/bin/env python3
"""
Training Script for Shift Imputation Model (Structure + Shifts + Retrieval).

Follows the same conventions as 06_train.py but trains the ShiftImputationModel
which combines structural features, observed shift context, and retrieval.

Key differences from 06_train.py:
- Per-(residue, shift_type) samples instead of per-residue
- Curriculum masking: target-only -> random additional masking
- Single scalar prediction per sample (not all 6 shifts)
- Huber loss on z-normalized targets, equal weights

Usage:
    # Build imputation cache only (no training)
    python 08_train_imputation.py --data_dir data --fold 1 --build_cache_only

    # Train fold 1
    python 08_train_imputation.py --data_dir data --fold 1 --epochs 350

    # Resume from checkpoint
    python 08_train_imputation.py --data_dir data --fold 1 --checkpoint checkpoints/...pt
"""

import argparse
import gc
import json
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from config import (
    LEARNING_RATE, BATCH_SIZE, EPOCHS, HUBER_DELTA,
    WEIGHT_DECAY, OUTLIER_STD_THRESHOLD,
    CONTEXT_WINDOW, K_SPATIAL_NEIGHBORS, K_RETRIEVED,
    MAX_VALID_DISTANCES, SHIFT_RANGES,
)
from dataset import (
    CachedRetrievalDataset,
    parse_distance_columns,
    build_atom_vocabulary,
    parse_shift_columns,
    get_dssp_columns,
)
from imputation_model import create_imputation_model
from imputation_dataset import load_imputation_dataset

GRAD_CLIP = 1.0
WANDB_PROJECT = "shift-imputation-v2"
NUM_WORKERS = 4
PREFETCH = 2


# ============================================================================
# Outlier Handling (same as 06_train.py)
# ============================================================================

def mask_cs_outliers(df, stats, shift_cols):
    """Mask chemical shift outliers as NaN."""
    df = df.copy()
    total_outliers = 0
    for col in shift_cols:
        if col in stats and col in df.columns:
            lower_bound = stats[col]['lower_bound']
            upper_bound = stats[col]['upper_bound']
            valid_mask = df[col].notna()
            outlier_mask = valid_mask & ((df[col] < lower_bound) | (df[col] > upper_bound))
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                df.loc[outlier_mask, col] = np.nan
                total_outliers += outlier_count
    print(f"  Outliers masked: {total_outliers:,}")
    return df, total_outliers


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, device, scaler=None, delta=0.5,
                clear_cache_every=50):
    """Training epoch. Each sample predicts one shift."""
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)
    nan_batches = 0

    pbar = tqdm(loader, desc="  Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        if device == 'cuda' and batch_idx > 0 and batch_idx % clear_cache_every == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch.pop('target_value')
        target_shift_idx = batch.pop('target_shift_idx')

        optimizer.zero_grad(set_to_none=True)

        ctx = autocast('cuda') if scaler else nullcontext()
        with ctx:
            pred = model(**batch)  # (B,)
            loss = F.huber_loss(pred, target, reduction='mean', delta=delta)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        bs = target.size(0)
        total_loss = total_loss + loss.detach() * bs
        total_count = total_count + bs

        if batch_idx % 10 == 0:
            avg = (total_loss / total_count).item() if total_count > 0 else 0.0
            pbar.set_postfix(loss=f"{avg:.4f}", nan=nan_batches)

    if nan_batches > 0:
        print(f"  Warning: {nan_batches} NaN batches skipped")

    count_val = total_count.item()
    return (total_loss / total_count).item() if count_val > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device, stats, shift_cols, delta=0.5):
    """Evaluate model: per-shift MAE (denormalized)."""
    model.eval()

    # Collect per-shift predictions
    per_shift_preds = {col: [] for col in shift_cols}
    per_shift_targets = {col: [] for col in shift_cols}
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch.pop('target_value')
        target_shift_idx = batch.pop('target_shift_idx')

        ctx = autocast('cuda') if device == 'cuda' else nullcontext()
        with ctx:
            pred = model(**batch)

        loss = F.huber_loss(pred, target, reduction='mean', delta=delta)
        bs = target.size(0)
        if bs > 0:
            total_loss += loss.item() * bs
            total_count += bs

        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        shift_idx_np = target_shift_idx.cpu().numpy()

        for i in range(len(shift_idx_np)):
            col = shift_cols[shift_idx_np[i]]
            per_shift_preds[col].append(pred_np[i])
            per_shift_targets[col].append(target_np[i])

    avg_loss = total_loss / total_count if total_count > 0 else 0.0

    # Denormalize and compute MAE
    per_shift_mae = {}
    for col in shift_cols:
        if col in stats and len(per_shift_preds[col]) > 0:
            preds = np.array(per_shift_preds[col])
            targs = np.array(per_shift_targets[col])
            pred_denorm = preds * stats[col]['std'] + stats[col]['mean']
            targ_denorm = targs * stats[col]['std'] + stats[col]['mean']
            per_shift_mae[col] = float(np.mean(np.abs(pred_denorm - targ_denorm)))

    overall_mae = np.mean(list(per_shift_mae.values())) if per_shift_mae else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Imputation Model)")
    print("=" * 70)
    print(f"Huber Loss: {avg_loss:.4f}")
    print(f"Overall MAE: {overall_mae:.4f} ppm")
    print(f"\nPer-shift MAE (ppm):")
    for col, mae in sorted(per_shift_mae.items(), key=lambda x: x[1]):
        n = len(per_shift_preds[col])
        print(f"  {col:20s}: {mae:.3f}  (n={n:,})")
    print("=" * 70 + "\n")

    return {
        'loss': avg_loss,
        'mae': overall_mae,
        'per_shift_mae': per_shift_mae,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train shift imputation model (structure + shifts + retrieval)')
    parser.add_argument('--fold', type=int, default=1, help='Fold to hold out (1-5)')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--huber_delta', type=float, default=HUBER_DELTA)
    parser.add_argument('--k_retrieved', type=int, default=K_RETRIEVED)
    parser.add_argument('--save_every', type=int, default=25)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--no_random_coil', action='store_true')
    parser.add_argument('--build_cache_only', action='store_true',
                        help='Build imputation cache and exit')
    parser.add_argument('--rebuild_cache', action='store_true')
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.data_dir, 'cache')
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'checkpoints')

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

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    num_workers = NUM_WORKERS if device == 'cuda' else 0

    print("=" * 80)
    print("SHIFT IMPUTATION MODEL (Structure + Observed Shifts + Retrieval)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Load base datasets ==========
    # We need the existing cached datasets from 06_train.py
    train_base_cache = os.path.join(args.cache_dir, f'fold{args.fold}_train')
    test_base_cache = os.path.join(args.cache_dir, f'fold{args.fold}_test')

    if not CachedRetrievalDataset.exists(train_base_cache):
        print(f"ERROR: Base training cache not found at {train_base_cache}")
        print("  Run 06_train.py first to build the structural + retrieval cache.")
        sys.exit(1)
    if not CachedRetrievalDataset.exists(test_base_cache):
        print(f"ERROR: Base test cache not found at {test_base_cache}")
        sys.exit(1)

    # Load data file for stats computation
    print("\nLoading data for statistics...")
    data_file = os.path.join(args.data_dir, 'structure_data.csv')
    if not os.path.exists(data_file):
        for c in ['small_structure_data.csv', 'sidechain_structure_data.csv']:
            p = os.path.join(args.data_dir, c)
            if os.path.exists(p):
                data_file = p
                break
        else:
            print(f"ERROR: No data file found in {args.data_dir}")
            sys.exit(1)

    df = pd.read_csv(data_file, dtype={'bmrb_id': str})
    shift_cols = parse_shift_columns(df.columns)
    dssp_cols = get_dssp_columns(df.columns)
    n_shifts = len(shift_cols)

    train_df = df[df['split'] != args.fold]
    print(f"  Shift columns: {shift_cols}")

    # Compute stats from training data
    stats = {}
    for col in shift_cols:
        vals = train_df[col].dropna()
        if len(vals) > 10:
            mean = float(vals.mean())
            std = float(vals.std())
            if not np.isnan(mean) and not np.isnan(std) and std > 1e-6:
                stats[col] = {
                    'mean': mean, 'std': std, 'count': len(vals),
                    'lower_bound': mean - OUTLIER_STD_THRESHOLD * std,
                    'upper_bound': mean + OUTLIER_STD_THRESHOLD * std,
                }

    stats['dssp'] = {}
    for col in dssp_cols:
        vals = train_df[col].dropna()
        if len(vals) > 10:
            mean = float(vals.mean())
            std = float(vals.std())
            if not np.isnan(mean) and not np.isnan(std) and std > 1e-6:
                stats['dssp'][col] = {'mean': mean, 'std': std}

    del df, train_df
    gc.collect()

    # Load base datasets
    print(f"\nLoading base training dataset from {train_base_cache}...")
    train_base = CachedRetrievalDataset.load(
        train_base_cache, n_shifts, args.k_retrieved,
        stats=stats, shift_cols=shift_cols,
    )
    print(f"  Base samples: {len(train_base):,}, Residues: {train_base.total_residues:,}")

    print(f"\nLoading base test dataset from {test_base_cache}...")
    test_base = CachedRetrievalDataset.load(
        test_base_cache, n_shifts, args.k_retrieved,
        stats=stats, shift_cols=shift_cols,
    )
    print(f"  Base samples: {len(test_base):,}, Residues: {test_base.total_residues:,}")

    # ========== Build/Load imputation datasets ==========
    train_imp_cache = os.path.join(args.cache_dir, f'fold{args.fold}_train', 'imputation')
    test_imp_cache = os.path.join(args.cache_dir, f'fold{args.fold}_test', 'imputation')

    if args.rebuild_cache:
        import shutil
        for p in [train_imp_cache, test_imp_cache]:
            if os.path.exists(p):
                shutil.rmtree(p)

    print("\nLoading/building imputation training dataset...")
    train_dataset = load_imputation_dataset(
        train_base, train_imp_cache, n_shifts, context_window=CONTEXT_WINDOW,
    )
    print(f"  Imputation samples: {len(train_dataset):,}")

    print("\nLoading/building imputation test dataset...")
    test_dataset = load_imputation_dataset(
        test_base, test_imp_cache, n_shifts, context_window=CONTEXT_WINDOW,
    )
    print(f"  Imputation samples: {len(test_dataset):,}")

    if args.build_cache_only:
        print("\n  --build_cache_only: done.")
        return

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == 'cuda'),
        prefetch_factor=PREFETCH if num_workers > 0 else None,
        persistent_workers=num_workers > 0, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == 'cuda'),
        prefetch_factor=PREFETCH if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # ========== Create model ==========
    print("\nCreating imputation model...")

    # Get n_atom_types from base dataset config
    n_atom_types = train_base.n_atom_types
    n_physics = getattr(train_base, 'n_physics', 28)

    model = create_imputation_model(
        n_atom_types=n_atom_types,
        n_shifts=n_shifts,
        n_physics=n_physics,
        shift_cols=shift_cols,
        use_random_coil=not args.no_random_coil,
        n_dssp=len(dssp_cols),
        k_spatial=K_SPATIAL_NEIGHBORS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Resume
    start_epoch = 1
    resume_checkpoint = None
    if args.checkpoint:
        print(f"\nResuming from: {args.checkpoint}")
        resume_checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        start_epoch = resume_checkpoint.get('epoch', 0) + 1
        print(f"  Resuming from epoch {start_epoch}")

    # ========== Test forward pass ==========
    print("\nTesting forward pass...")
    sample_batch = next(iter(train_loader))
    sample_gpu = {k: v.to(device) for k, v in sample_batch.items()}
    target = sample_gpu.pop('target_value')
    _ = sample_gpu.pop('target_shift_idx')

    with torch.no_grad():
        pred = model(**sample_gpu)
        print(f"  pred shape: {pred.shape}")
        print(f"  pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        loss = F.huber_loss(pred, target, reduction='mean', delta=args.huber_delta)
        print(f"  initial loss: {loss.item():.4f}")

    del sample_batch, sample_gpu, target, pred
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # ========== Optimizer ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=args.lr * 0.01,
    )
    scaler = GradScaler('cuda') if device == 'cuda' else None

    if resume_checkpoint is not None:
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])

    # WandB
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"imputation-fold{args.fold}-lr{args.lr}",
            config={
                **vars(args),
                'weight_decay': WEIGHT_DECAY,
                'grad_clip': GRAD_CLIP,
                'n_params': n_params,
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
            },
        )

    # ========== Training loop ==========
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    best_mae = float('inf')
    epoch_times = []

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler,
            delta=args.huber_delta,
        )
        scheduler.step()

        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"  Train loss: {train_loss:.4f}, LR: {current_lr:.2e}, "
              f"Time: {epoch_time:.1f}s")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_dir, f'imputation_fold{args.fold}_epoch{epoch}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'stats': stats,
                'shift_cols': shift_cols,
                'dssp_cols': dssp_cols,
                'n_atom_types': n_atom_types,
                'n_physics': n_physics,
                'k_retrieved': args.k_retrieved,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Evaluate periodically
        eval_results = None
        if epoch % 5 == 0 or epoch == args.epochs:
            eval_results = evaluate(
                model, test_loader, device, stats, shift_cols,
                delta=args.huber_delta,
            )

            if use_wandb:
                log = {
                    'train/loss': train_loss,
                    'test/loss': eval_results['loss'],
                    'test/mae': eval_results['mae'],
                    'lr': current_lr,
                }
                for col, mae in eval_results['per_shift_mae'].items():
                    log[f'test/mae_{col}'] = mae
                wandb.log(log, step=epoch)

            if eval_results['mae'] < best_mae:
                best_mae = eval_results['mae']
                best_path = os.path.join(
                    args.output_dir, f'best_imputation_fold{args.fold}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'mae': eval_results['mae'],
                    'per_shift_mae': eval_results['per_shift_mae'],
                    'stats': stats,
                    'shift_cols': shift_cols,
                    'dssp_cols': dssp_cols,
                    'n_atom_types': n_atom_types,
                    'n_physics': n_physics,
                    'k_retrieved': args.k_retrieved,
                }, best_path)
                print(f"  *** New best model saved (MAE: {best_mae:.4f} ppm) ***")

        # WandB for non-eval epochs
        if use_wandb and eval_results is None:
            wandb.log({
                'train/loss': train_loss,
                'lr': current_lr,
            }, step=epoch)

    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"  Best MAE: {best_mae:.4f} ppm")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Mean epoch time: {np.mean(epoch_times):.1f}s")
    print(f"  Best model: {os.path.join(args.output_dir, f'best_imputation_fold{args.fold}.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
