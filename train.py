#!/usr/bin/env python3
"""
Unified training script for chemical shift prediction.

Modes:
    # Full retrieval model (default)
    python train.py --data hybrid --fold 1 --epochs 150

    # Structure-only (no retrieval components)
    python train.py --data hybrid --fold 1 --epochs 200 --structure_only

    # Frozen base encoder, train retrieval only
    python train.py --data hybrid --fold 1 --epochs 150 \
        --freeze_base --base_checkpoint runs/hybrid_struct_fold1/checkpoints/best.pt

    # Custom output directory
    python train.py --data hybrid --fold 1 --epochs 150 --run_name my_experiment
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader, ConcatDataset

from config import (
    DATASET_DIRS, N_ATOM_TYPES, ATOM_TO_IDX,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, HUBER_DELTA,
    WEIGHT_DECAY, K_SPATIAL_NEIGHBORS, K_RETRIEVED,
)
from dataset import CachedRetrievalDataset
from model import create_model
from training_utils import (
    BACKBONE_SHIFTS, NUM_WORKERS, PREFETCH,
    huber_loss_masked, build_shift_weights, train_epoch, evaluate,
    load_base_checkpoint, freeze_base_encoder,
)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train chemical shift predictor')

    # Data
    parser.add_argument('--data', choices=list(DATASET_DIRS.keys()), required=True)
    parser.add_argument('--fold', type=int, default=1, help='Test fold (1-5)')

    # Training
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--huber_delta', type=float, default=HUBER_DELTA)
    parser.add_argument('--save_every', type=int, default=25)

    # Architecture mode
    parser.add_argument('--structure_only', action='store_true',
                        help='Structure-only model (no retrieval components)')
    parser.add_argument('--freeze_base', action='store_true',
                        help='Freeze base encoder, train retrieval only')
    parser.add_argument('--base_checkpoint', type=str, default=None,
                        help='Checkpoint to load base encoder from (for --freeze_base)')

    # Resume
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume training from checkpoint')

    # Output
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for output directory (default: auto)')
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Base output directory (default: runs/)')

    # System
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--cache_device', type=str, default='cpu',
                        help='Device for cache building FAISS queries')

    args = parser.parse_args()

    # Validate flags
    if args.freeze_base and args.structure_only:
        print("ERROR: --freeze_base and --structure_only are mutually exclusive")
        sys.exit(1)
    if args.freeze_base and not args.base_checkpoint and not args.checkpoint:
        print("ERROR: --freeze_base requires --base_checkpoint or --checkpoint")
        sys.exit(1)

    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Derive mode string
    if args.structure_only:
        mode = 'struct'
    elif args.freeze_base:
        mode = 'frozen_retrieval'
    else:
        mode = 'retrieval'

    # Auto-generate run name
    if args.run_name is None:
        args.run_name = f'{args.data}_{mode}_fold{args.fold}'

    run_dir = os.path.join(args.output_dir, args.run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resolve data paths
    data_dir = DATASET_DIRS[args.data]
    cache_dir = os.path.join(data_dir, 'cache')

    print("=" * 70)
    print(f"  Chemical Shift Prediction — {mode}")
    print("=" * 70)
    print(f"  Dataset:    {args.data} ({data_dir})")
    print(f"  Fold:       {args.fold}")
    print(f"  Mode:       {mode}")
    print(f"  Device:     {device}")
    print(f"  Output:     {run_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    # ========== Build cache if needed ==========
    missing_folds = []
    for f in range(1, 6):
        if not CachedRetrievalDataset.exists(os.path.join(cache_dir, f'fold_{f}')):
            missing_folds.append(f)

    if missing_folds:
        print(f"\n  Missing caches for folds: {missing_folds}")
        import subprocess
        embeddings = os.path.join('data', 'esm_embeddings.h5')
        index_dir = os.path.join('data', 'retrieval_indices')
        cmd = [
            sys.executable, '05_build_training_cache.py',
            '--data_dir', data_dir,
            '--embeddings', embeddings,
            '--index_dir', index_dir,
            '--output_dir', cache_dir,
            '--folds', *[str(f) for f in missing_folds],
            '--device', args.cache_device,
        ]
        print(f"  Building cache: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: Cache build failed")
            sys.exit(1)
    else:
        print(f"\n  All fold caches exist in {cache_dir}")

    # ========== Load datasets ==========
    config_path = os.path.join(cache_dir, f'fold_{args.fold}', 'config.json')
    with open(config_path) as f:
        cache_config = json.load(f)
    stats = cache_config['stats']
    shift_cols = cache_config['shift_cols']
    n_shifts = len(shift_cols)

    test_ds = CachedRetrievalDataset.load(
        os.path.join(cache_dir, f'fold_{args.fold}'),
        n_shifts, K_RETRIEVED, stats=stats, shift_cols=shift_cols)

    train_parts = []
    for f in range(1, 6):
        if f == args.fold:
            continue
        fold_cache = os.path.join(cache_dir, f'fold_{f}')
        if not CachedRetrievalDataset.exists(fold_cache):
            print(f"ERROR: Cache not found at {fold_cache}")
            sys.exit(1)
        train_parts.append(CachedRetrievalDataset.load(
            fold_cache, n_shifts, K_RETRIEVED, stats=stats, shift_cols=shift_cols))
    train_ds = ConcatDataset(train_parts)
    n_struct = getattr(train_parts[0], 'n_struct_features', 49)

    nw = NUM_WORKERS if device == 'cuda' else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              prefetch_factor=PREFETCH if nw > 0 else None,
                              persistent_workers=nw > 0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True,
                             prefetch_factor=PREFETCH if nw > 0 else None,
                             persistent_workers=nw > 0)

    print(f"\n  Shifts:     {n_shifts}")
    print(f"  Train:      {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"  Test:       {len(test_ds):,} samples ({len(test_loader)} batches)")

    # ========== Create model ==========
    use_retrieval = not args.structure_only
    model = create_model(
        n_atom_types=N_ATOM_TYPES,
        n_shifts=n_shifts,
        n_struct=n_struct,
        n_dssp=cache_config.get('n_dssp', 9),
        k_spatial=K_SPATIAL_NEIGHBORS,
        use_retrieval=use_retrieval,
    ).to(device)

    start_epoch = 1

    # Load base checkpoint (for --freeze_base)
    if args.base_checkpoint:
        print(f"\n  Loading base encoder: {args.base_checkpoint}")
        load_base_checkpoint(model, args.base_checkpoint, device)

    # Resume from full checkpoint
    ckpt = None
    if args.checkpoint:
        print(f"\n  Resuming from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()
              if not k.startswith('physics_encoder.')}
        model.load_state_dict(sd, strict=False)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"  Resuming from epoch {start_epoch}")

    # Freeze base encoder if requested
    if args.freeze_base:
        freeze_base_encoder(model)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Compile
    if device == 'cuda':
        model = torch.compile(model)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=args.lr * 0.01)
    scaler = GradScaler('cuda') if device == 'cuda' else None

    if ckpt is not None:
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except ValueError:
                print("  Warning: could not restore optimizer state (param mismatch)")
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # Save training config
    train_config = {
        **vars(args),
        'mode': mode,
        'n_shifts': n_shifts,
        'n_params': n_params,
        'n_trainable': n_trainable,
        'shift_cols': shift_cols,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(train_config, f, indent=2)

    # ========== Training loop ==========
    shift_weights = build_shift_weights(shift_cols, device)
    best_mae = float('inf')
    training_log = []

    print(f"\n{'='*70}")
    print(f"  Training: {mode} | {args.data} | fold {args.fold} | epochs {start_epoch}-{args.epochs}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        loss = train_epoch(model, train_loader, optimizer, device, scaler,
                           delta=args.huber_delta, shift_weights=shift_weights)
        scheduler.step()

        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        print(f"  Loss: {loss:.4f}  LR: {lr:.2e}  Time: {dt:.0f}s")

        epoch_log = {'epoch': epoch, 'loss': loss, 'lr': lr, 'time': dt}

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            path = os.path.join(ckpt_dir, f'checkpoint_epoch{epoch}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': loss,
                'stats': stats,
                'shift_cols': shift_cols,
                'atom_to_idx': ATOM_TO_IDX,
                'k_retrieved': K_RETRIEVED,
                'dataset': args.data,
                'mode': mode,
            }, path)
            print(f"  Saved: {path}")

        # Evaluate
        if epoch % 5 == 0 or epoch == args.epochs:
            results = evaluate(model, test_loader, device, stats, shift_cols,
                               args.huber_delta)
            epoch_log['mae'] = results['mae']
            epoch_log['per_shift_mae'] = results['per_shift_mae']

            if results['mae'] < best_mae:
                best_mae = results['mae']
                best_path = os.path.join(ckpt_dir, 'best.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'mae': results['mae'],
                    'per_shift_mae': results['per_shift_mae'],
                    'stats': stats,
                    'shift_cols': shift_cols,
                    'atom_to_idx': ATOM_TO_IDX,
                    'k_retrieved': K_RETRIEVED,
                    'dataset': args.data,
                    'mode': mode,
                }, best_path)
                print(f"  *** New best: {best_mae:.4f} ***")

        training_log.append(epoch_log)

    # Save training log
    log_path = os.path.join(run_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2, default=str)

    print(f"\nDone! Best MAE: {best_mae:.4f}")
    print(f"  Run directory: {run_dir}")
    print(f"  Best checkpoint: {os.path.join(ckpt_dir, 'best.pt')}")


if __name__ == '__main__':
    main()
