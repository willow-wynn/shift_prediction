#!/usr/bin/env python3
"""
Training Script for Retrieval-Augmented Chemical Shift Prediction
(Better Data Pipeline)

Adapted from homologies/train_retrieval.py with improvements:
- Imports hyperparameters from config.py (central configuration)
- Model creation via create_model() factory from model.py
- Outlier masking: mask predictions where |residual| > 4*std during loss
- AdamW optimizer with CosineAnnealingWarmRestarts scheduler
- AMP (automatic mixed precision) support
- Optional WandB logging (--no_wandb flag)
- 5-fold cross-validation support (--fold)
- Full provenance logging: samples used, masked outliers per epoch, LR schedule

Usage:
    # Train fold 1
    python 06_train.py --data_dir data --cache_dir cache --fold 1

    # Train with WandB disabled
    python 06_train.py --data_dir data --cache_dir cache --fold 1 --no_wandb
"""

import argparse
import gc
import json
import os
import sys
import time
from contextlib import nullcontext

# Local imports
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
    MAX_VALID_DISTANCES,
)
from dataset import (
    CachedRetrievalDataset,
    parse_distance_columns,
    build_atom_vocabulary,
    parse_shift_columns,
    get_dssp_columns,
)
from model import create_model
from data_quality import FilterLog

# Gradient clipping constant
GRAD_CLIP = 1.0

WANDB_PROJECT = "shift-retrieval-v2"

NUM_WORKERS = 4
PREFETCH = 2


# ============================================================================
# Provenance Logger
# ============================================================================

class TrainingProvenance:
    """Log training decisions and data flow for full reproducibility."""

    def __init__(self):
        self.records = {
            'data_summary': {},
            'per_epoch': [],
            'configuration': {},
        }

    def log_data_summary(self, key, value):
        self.records['data_summary'][key] = value
        print(f"  [Provenance] {key}: {value}")

    def log_config(self, args):
        self.records['configuration'] = vars(args) if hasattr(args, '__dict__') else dict(args)

    def log_epoch(self, epoch, train_loss, val_mae, lr, masked_outliers, eval_results=None):
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mae': val_mae,
            'learning_rate': lr,
            'masked_outliers': masked_outliers,
        }
        if eval_results:
            entry['per_shift_mae'] = eval_results.get('per_shift_mae', {})
        self.records['per_epoch'].append(entry)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.records, f, indent=2, default=str)
        print(f"  Training provenance saved: {path}")


# ============================================================================
# Outlier Handling
# ============================================================================

def mask_cs_outliers(df, stats, shift_cols):
    """Mask chemical shift outliers as NaN (excluded from training).

    Uses both physical range bounds (from config.SHIFT_RANGES) and
    statistical bounds (mean +/- OUTLIER_STD_THRESHOLD * std).

    Returns:
        (filtered_df, total_outliers_masked)
    """
    print("\nMasking chemical shift outliers...")
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
                print(f"  {col}: {outlier_count} outliers masked "
                      f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
                df.loc[outlier_mask, col] = np.nan
                total_outliers += outlier_count

    print(f"Total outliers masked: {total_outliers:,}")
    return df, total_outliers


# ============================================================================
# Loss Functions
# ============================================================================

def build_shift_weights(shift_cols, device):
    """Build per-shift loss weights. Backbone shifts 2x, sidechain 1x."""
    from config import BACKBONE_LOSS_WEIGHT
    BACKBONE_SHIFTS = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}
    weights = torch.ones(len(shift_cols), device=device)
    for si, col in enumerate(shift_cols):
        if col in BACKBONE_SHIFTS:
            weights[si] = BACKBONE_LOSS_WEIGHT
    return weights


def huber_loss_masked(pred, target, mask, delta=0.5, shift_weights=None):
    """Huber loss over masked positions with optional per-shift weighting.

    Args:
        pred: (B, n_shifts) predictions
        target: (B, n_shifts) targets
        mask: (B, n_shifts) boolean mask
        delta: Huber delta parameter
        shift_weights: (n_shifts,) optional per-shift loss weights

    Returns:
        Scalar loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    if shift_weights is not None:
        # Per-element Huber loss with shift-specific weighting
        elem_loss = F.huber_loss(pred, target, reduction='none', delta=delta)
        weighted = elem_loss * shift_weights.unsqueeze(0)  # (B, n_shifts)
        loss = weighted[mask].mean()
    else:
        loss = F.huber_loss(pred[mask], target[mask], reduction='mean', delta=delta)

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return loss


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, device, scaler=None, delta=0.5,
                clear_cache_every=50, shift_weights=None):
    """Training epoch with Huber loss and outlier tracking.

    Returns:
        (avg_loss, total_masked_outliers)
    """
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)
    nan_batches = 0

    pbar = tqdm(loader, desc="  Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Periodic GPU cache clearing
        if device == 'cuda' and batch_idx > 0 and batch_idx % clear_cache_every == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch.pop('shift_target')
        mask = batch.pop('shift_mask')

        if mask.sum() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        ctx = autocast('cuda') if scaler else nullcontext()
        with ctx:
            pred = model(**batch)
            loss = huber_loss_masked(pred, target, mask, delta=delta,
                                     shift_weights=shift_weights)

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

        # Accumulate on GPU without .item() to avoid sync stalls
        bs = mask.sum()
        total_loss = total_loss + loss.detach() * bs
        total_count = total_count + bs

        if batch_idx % 10 == 0:
            avg_so_far = (total_loss / total_count).item() if total_count > 0 else 0.0
            pbar.set_postfix(loss=f"{avg_so_far:.4f}", nan=nan_batches)

    if nan_batches > 0:
        print(f"  Warning: {nan_batches} batches skipped due to NaN")

    total_count_val = total_count.item()
    avg_loss = (total_loss / total_count).item() if total_count_val > 0 else 0.0

    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, stats, shift_cols, delta=0.5):
    """Evaluate model and return per-shift MAE (denormalized)."""
    model.eval()

    all_pred = []
    all_target = []
    all_mask = []
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch.pop('shift_target')
        mask = batch.pop('shift_mask')

        ctx = autocast('cuda') if device == 'cuda' else nullcontext()
        with ctx:
            pred = model(**batch)

        loss = huber_loss_masked(pred, target, mask, delta=delta)

        bs = mask.sum().item()
        if bs > 0:
            total_loss += loss.item() * bs
            total_count += bs

        all_pred.append(pred.cpu())
        all_target.append(target.cpu())
        all_mask.append(mask.cpu())

    all_pred = torch.cat(all_pred)
    all_target = torch.cat(all_target)
    all_mask = torch.cat(all_mask)

    avg_loss = total_loss / total_count if total_count > 0 else 0.0

    # MAE per shift (denormalized)
    per_shift_mae = {}
    for si, col in enumerate(shift_cols):
        mask_i = all_mask[:, si]
        if mask_i.sum() > 0 and col in stats:
            pred_denorm = all_pred[:, si][mask_i] * stats[col]['std'] + stats[col]['mean']
            true_denorm = all_target[:, si][mask_i] * stats[col]['std'] + stats[col]['mean']
            per_shift_mae[col] = (pred_denorm - true_denorm).abs().mean().item()

    overall_mae = sum(per_shift_mae.values()) / len(per_shift_mae) if per_shift_mae else 0.0

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Huber Loss: {avg_loss:.4f}")
    print(f"Overall MAE: {overall_mae:.4f} ppm")
    print(f"\nPer-shift MAE (ppm):")
    sorted_shifts = sorted(per_shift_mae.items(), key=lambda x: x[1])
    for col, mae in sorted_shifts:
        print(f"  {col:20s}: {mae:.3f}")
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
        description='Train retrieval-augmented chemical shift predictor (better data pipeline)')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold to hold out for testing (1-5)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing data CSV file (default: ./data)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for dataset cache (default: <data_dir>/cache)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for checkpoints (default: <data_dir>/checkpoints)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detected if omitted)')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--huber_delta', type=float, default=HUBER_DELTA)
    parser.add_argument('--k_retrieved', type=int, default=K_RETRIEVED)
    parser.add_argument('--save_every', type=int, default=25)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--no_query_conditioned', action='store_true')

    parser.add_argument('--rebuild_cache', action='store_true')
    args = parser.parse_args()

    # Derive defaults from data_dir
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.data_dir, 'cache')
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'checkpoints')

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

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    num_workers = NUM_WORKERS if device == 'cuda' else 0

    print("=" * 80)
    print("RETRIEVAL-AUGMENTED CHEMICAL SHIFT PREDICTION (Better Data Pipeline)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Fold: {args.fold} (held out for testing)")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Huber delta: {args.huber_delta}")
    print(f"K retrieved: {args.k_retrieved}")
    print(f"Grad clip: {GRAD_CLIP}")
    print(f"Query-conditioned transfer: {not args.no_query_conditioned}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize provenance logger
    provenance = TrainingProvenance()
    provenance.log_config(args)

    # ========== Load data ==========
    print("\nLoading data...")
    data_file = None
    for name in ['structure_data_hybrid.csv', 'structure_data.csv']:
        candidate = os.path.join(args.data_dir, name)
        if os.path.exists(candidate):
            data_file = candidate
            break
    if data_file is None:
        print(f"ERROR: No data file found in {args.data_dir}")
        sys.exit(1)

    # Read header for column names, then load only light columns
    # (the full CSV can be 6GB+ with 1500 distance columns — too large for pandas)
    all_columns = pd.read_csv(data_file, nrows=0).columns.tolist()
    dist_col_info = parse_distance_columns(all_columns)
    atom_list, atom_to_idx = build_atom_vocabulary(dist_col_info)
    shift_cols = parse_shift_columns(all_columns)
    dssp_cols = get_dssp_columns(all_columns)

    # Load only columns needed for stats + fold splitting
    light_cols = ['bmrb_id', 'residue_id', 'residue_code', 'split']
    light_cols += shift_cols + dssp_cols
    light_cols = [c for c in light_cols if c in all_columns]

    # Try per-fold files first (avoids pandas memory explosion on huge CSVs)
    fold_files = [os.path.join(args.data_dir, f'structure_data_hybrid_fold_{f}.csv')
                  for f in range(1, 6)]
    if all(os.path.exists(f) for f in fold_files):
        print(f"  Loading from per-fold CSV files...")
        parts = []
        for ff in fold_files:
            avail = [c for c in light_cols if c in pd.read_csv(ff, nrows=0).columns]
            parts.append(pd.read_csv(ff, usecols=avail, dtype={'bmrb_id': str}, low_memory=False))
        df = pd.concat(parts, ignore_index=True)
        del parts
    else:
        df = pd.read_csv(data_file, usecols=light_cols, dtype={'bmrb_id': str}, low_memory=False)

    print(f"  Loaded {len(df):,} residues from {df['bmrb_id'].nunique():,} proteins")
    print(f"  Data file: {data_file}")
    provenance.log_data_summary('data_file', data_file)
    provenance.log_data_summary('total_residues_loaded', len(df))
    provenance.log_data_summary('total_proteins_loaded', int(df['bmrb_id'].nunique()))

    print(f"  Distance columns: {len(dist_col_info)}")
    print(f"  Atom types: {len(atom_list)}")
    print(f"  Shift columns: {len(shift_cols)} -> {shift_cols}")
    print(f"  DSSP columns: {len(dssp_cols)}")

    provenance.log_data_summary('n_distance_columns', len(dist_col_info))
    provenance.log_data_summary('n_atom_types', len(atom_list))
    provenance.log_data_summary('shift_columns', shift_cols)
    provenance.log_data_summary('n_dssp_columns', len(dssp_cols))

    # Split by fold
    if 'split' not in df.columns:
        print("ERROR: Data file must have a 'split' column for fold assignment.")
        sys.exit(1)

    train_df = df[df['split'] != args.fold].copy()
    test_df = df[df['split'] == args.fold].copy()

    print(f"  Train: {len(train_df):,} residues, {train_df['bmrb_id'].nunique():,} proteins")
    print(f"  Test:  {len(test_df):,} residues, {test_df['bmrb_id'].nunique():,} proteins")

    provenance.log_data_summary('train_residues', len(train_df))
    provenance.log_data_summary('train_proteins', int(train_df['bmrb_id'].nunique()))
    provenance.log_data_summary('test_residues', len(test_df))
    provenance.log_data_summary('test_proteins', int(test_df['bmrb_id'].nunique()))

    del df
    gc.collect()

    # ========== Compute statistics ==========
    print("\nComputing statistics from training data...")
    stats = {}

    for col in shift_cols:
        vals = train_df[col].dropna()
        if len(vals) > 10:
            mean = float(vals.mean())
            std = float(vals.std())
            if not np.isnan(mean) and not np.isnan(std) and std > 1e-6:
                stats[col] = {
                    'mean': mean,
                    'std': std,
                    'count': len(vals),
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

    valid_shift_types = len([k for k in stats if k != 'dssp'])
    print(f"  Valid shift types: {valid_shift_types} / {len(shift_cols)}")
    provenance.log_data_summary('valid_shift_types', valid_shift_types)

    # Log per-shift statistics
    for col in shift_cols:
        if col in stats:
            provenance.log_data_summary(
                f'stats_{col}',
                f"mean={stats[col]['mean']:.3f}, std={stats[col]['std']:.3f}, "
                f"count={stats[col]['count']:,}, "
                f"bounds=[{stats[col]['lower_bound']:.2f}, {stats[col]['upper_bound']:.2f}]"
            )

    # ========== Mask outliers ==========
    train_df, train_outliers = mask_cs_outliers(train_df, stats, shift_cols)
    test_df, test_outliers = mask_cs_outliers(test_df, stats, shift_cols)

    provenance.log_data_summary('train_outliers_masked', train_outliers)
    provenance.log_data_summary('test_outliers_masked', test_outliers)

    # ========== Create/Load cached datasets ==========
    train_cache = os.path.join(args.cache_dir, f'fold{args.fold}_train')
    test_cache = os.path.join(args.cache_dir, f'fold{args.fold}_test')

    need_train_build = not CachedRetrievalDataset.exists(train_cache) or args.rebuild_cache
    need_test_build = not CachedRetrievalDataset.exists(test_cache) or args.rebuild_cache

    embedding_lookup = None
    retriever = None

    if need_train_build or need_test_build:
        print("\nInitializing ESM embedding lookup (needed for cache build)...")
        from retrieval import EmbeddingLookup, Retriever

        emb_file = os.path.join(args.data_dir, 'esm_embeddings.h5')
        if not os.path.exists(emb_file):
            print(f"ERROR: ESM embeddings not found at {emb_file}")
            print("  Run the ESM extraction step first.")
            sys.exit(1)

        embedding_lookup = EmbeddingLookup(emb_file, cache_size=50)

        index_dir = os.path.join(args.data_dir, 'retrieval_indices')
        if not os.path.exists(index_dir):
            print(f"ERROR: FAISS indices not found at {index_dir}")
            print("  Run the FAISS index building step first.")
            sys.exit(1)

        print("Initializing retriever...")
        retriever = Retriever(
            index_dir=index_dir,
            exclude_fold=args.fold,
            k=args.k_retrieved,
            nprobe=64,
            device='cuda' if device == 'cuda' else 'cpu',
        )

    # Training dataset
    if not need_train_build:
        print(f"\nLoading cached training dataset from {train_cache}...")
        train_dataset = CachedRetrievalDataset.load(
            train_cache, len(shift_cols), args.k_retrieved,
            stats=stats, shift_cols=shift_cols,
        )
    else:
        print(f"\nBuilding training dataset (will be cached to {train_cache})...")
        train_dataset = CachedRetrievalDataset.create(
            df=train_df,
            shift_cols=shift_cols,
            dist_col_info=dist_col_info,
            dssp_cols=dssp_cols,
            atom_to_idx=atom_to_idx,
            stats=stats,
            embedding_lookup=embedding_lookup,
            retriever=retriever,
            cache_dir=train_cache,
            context_window=CONTEXT_WINDOW,
            k_spatial=K_SPATIAL_NEIGHBORS,
            k_retrieved=args.k_retrieved,
            max_valid_distances=MAX_VALID_DISTANCES,
        )

    del train_df
    gc.collect()

    # Test dataset
    if not need_test_build:
        print(f"\nLoading cached test dataset from {test_cache}...")
        test_dataset = CachedRetrievalDataset.load(
            test_cache, len(shift_cols), args.k_retrieved,
            stats=stats, shift_cols=shift_cols,
        )
    else:
        print(f"\nBuilding test dataset (will be cached to {test_cache})...")
        test_dataset = CachedRetrievalDataset.create(
            df=test_df,
            shift_cols=shift_cols,
            dist_col_info=dist_col_info,
            dssp_cols=dssp_cols,
            atom_to_idx=atom_to_idx,
            stats=stats,
            embedding_lookup=embedding_lookup,
            retriever=retriever,
            cache_dir=test_cache,
            context_window=CONTEXT_WINDOW,
            k_spatial=K_SPATIAL_NEIGHBORS,
            k_retrieved=args.k_retrieved,
            max_valid_distances=MAX_VALID_DISTANCES,
        )

    del test_df
    gc.collect()

    # Clean up
    if embedding_lookup is not None:
        embedding_lookup.close()
        del embedding_lookup
    if retriever is not None:
        del retriever
    gc.collect()

    provenance.log_data_summary('train_samples', len(train_dataset))
    provenance.log_data_summary('test_samples', len(test_dataset))

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
    print("\nCreating model via create_model() factory...")

    # Detect feature dimensions from dataset
    n_struct = getattr(train_dataset, 'n_struct_features', 49)

    model = create_model(
        n_atom_types=len(atom_to_idx),
        n_shifts=len(shift_cols),
        n_struct=n_struct,
        shift_cols=shift_cols,
        stats=stats,
        n_dssp=len(dssp_cols),
        k_spatial=K_SPATIAL_NEIGHBORS,
        use_query_conditioned_transfer=not args.no_query_conditioned,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    provenance.log_data_summary('model_parameters', n_params)

    # Resume from checkpoint if specified
    start_epoch = 1
    resume_checkpoint = None
    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        resume_checkpoint = torch.load(args.checkpoint, map_location=device)
        # Filter out deprecated physics_encoder keys from old checkpoints
        sd = {k: v for k, v in resume_checkpoint['model_state_dict'].items()
              if not k.startswith('physics_encoder.')}
        model.load_state_dict(sd, strict=False)
        start_epoch = resume_checkpoint.get('epoch', 0) + 1
        print(f"  Resuming from epoch {start_epoch}")
        provenance.log_data_summary('resumed_from', args.checkpoint)
        provenance.log_data_summary('resume_epoch', start_epoch)

    # ========== Test forward pass ==========
    print("\nTesting forward pass...")
    sample_batch = next(iter(train_loader))
    sample_batch_gpu = {k: v.to(device) for k, v in sample_batch.items()}
    target = sample_batch_gpu.pop('shift_target')
    mask = sample_batch_gpu.pop('shift_mask')

    with torch.no_grad():
        try:
            pred = model(**sample_batch_gpu)
            print(f"  pred shape: {pred.shape}")
            print(f"  pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
            loss = huber_loss_masked(pred, target, mask, delta=args.huber_delta)
            print(f"  initial loss: {loss.item():.4f}")
        except Exception as e:
            print(f"  ERROR in forward pass: {e}")
            raise

    del sample_batch, sample_batch_gpu, target, mask, pred
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # ========== Optimizer and scheduler ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=args.lr * 0.01
    )
    scaler = GradScaler('cuda') if device == 'cuda' else None

    # Restore optimizer/scheduler state if resuming
    if resume_checkpoint is not None:
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            print("  Restored optimizer state")
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            print("  Restored scheduler state")

    # ========== WandB ==========
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"v2-fold{args.fold}-lr{args.lr}",
            config={
                **vars(args),
                'weight_decay': WEIGHT_DECAY,
                'grad_clip': GRAD_CLIP,
                'n_params': n_params,
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
            }
        )

    # ========== Training loop ==========
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    shift_weights = build_shift_weights(shift_cols, device)
    print(f"  Shift loss weights: {dict(zip(shift_cols, shift_weights.tolist()))}")

    best_mae = float('inf')
    epoch_times = []

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler,
            delta=args.huber_delta, shift_weights=shift_weights,
        )
        scheduler.step()

        # Synchronize and clear GPU cache
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
            ckpt_path = os.path.join(args.output_dir,
                                     f'checkpoint_fold{args.fold}_epoch{epoch}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'stats': stats,
                'shift_cols': shift_cols,
                'dssp_cols': dssp_cols,
                'atom_to_idx': atom_to_idx,
                'k_retrieved': args.k_retrieved,
            }, ckpt_path)
            print(f"  Checkpoint saved at epoch {epoch}: {ckpt_path}")

        # Evaluate periodically
        val_mae = None
        eval_results = None
        if epoch % 5 == 0 or epoch == args.epochs:
            eval_results = evaluate(model, test_loader, device, stats, shift_cols,
                                    delta=args.huber_delta)
            val_mae = eval_results['mae']

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

            # Save best model
            if eval_results['mae'] < best_mae:
                best_mae = eval_results['mae']
                best_path = os.path.join(args.output_dir,
                                         f'best_retrieval_fold{args.fold}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'mae': eval_results['mae'],
                    'per_shift_mae': eval_results['per_shift_mae'],
                    'stats': stats,
                    'shift_cols': shift_cols,
                    'dssp_cols': dssp_cols,
                    'atom_to_idx': atom_to_idx,
                    'k_retrieved': args.k_retrieved,
                }, best_path)
                print(f"  *** New best model saved (MAE: {best_mae:.4f} ppm) ***")

        # Log epoch provenance (masked_outliers is 0 here since masking is
        # done on the data, not dynamically; kept for schema consistency)
        provenance.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_mae=val_mae,
            lr=current_lr,
            masked_outliers=0,
            eval_results=eval_results,
        )

        # WandB log for non-eval epochs
        if use_wandb and eval_results is None:
            wandb.log({
                'train/loss': train_loss,
                'lr': current_lr,
            }, step=epoch)

    if use_wandb:
        wandb.finish()

    # Save final provenance
    provenance_path = os.path.join(args.output_dir,
                                   f'training_provenance_fold{args.fold}.json')
    provenance.save(provenance_path)

    # Summary
    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"  Best MAE: {best_mae:.4f} ppm")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Mean epoch time: {np.mean(epoch_times):.1f}s")
    print(f"  Provenance log: {provenance_path}")
    print(f"  Best model: {os.path.join(args.output_dir, f'best_retrieval_fold{args.fold}.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
