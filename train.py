#!/usr/bin/env python3
"""
Unified training script for chemical shift prediction (structure-only model).

Modes:
    # Structure-only model (default)
    python train.py --data hybrid --fold 1 --epochs 150

    # Phase 1: freeze base encoder + struct_head, train only the cross-residue
    # distance features on top of an already-trained baseline
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
from torch.utils.data import DataLoader, ConcatDataset, Subset

from config import (
    DATASET_DIRS, N_ATOM_TYPES, ATOM_TO_IDX,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, HUBER_DELTA,
    WEIGHT_DECAY, K_SPATIAL_NEIGHBORS,
    BIG_RUNS_DIR,
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
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Override the cache directory (default: <data_dir>/cache).')
    parser.add_argument('--mmap_structural', action='store_true',
                        help='mmap the structural arrays (low RAM, slow on spinning '
                             'disk). Default: load fully into RAM (~5 GB/fold).')
    parser.add_argument('--protein_subset', type=str, default=None,
                        help='JSON list of BMRB IDs; restrict train+test to these '
                             'proteins (matched-set comparisons). Default: use all.')
    parser.add_argument('--ablate', type=str, default=None,
                        help='Comma-separated feature group(s) to ZERO OUT at train+eval '
                             'time (true ablation). Valid: intra, cross, spatial, dssp, ss, '
                             'bond_geom, residue_type. Default: none.')

    # Training
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--huber_delta', type=float, default=HUBER_DELTA)
    parser.add_argument('--save_every', type=int, default=25)

    # Architecture mode
    parser.add_argument('--freeze_base', action='store_true',
                        help='Freeze base encoder + struct_head, train only cross-residue '
                             'distance features (Phase 1)')
    parser.add_argument('--base_checkpoint', type=str, default=None,
                        help='Checkpoint to load base encoder from (for --freeze_base)')

    # Resume
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume training from checkpoint')

    # Output
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for output directory (default: auto)')
    parser.add_argument('--output_dir', type=str, default=BIG_RUNS_DIR,
                        help=f'Base output directory (default: {BIG_RUNS_DIR}, '
                             f'on 1TB drive to avoid filling main disk)')

    # System
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--cache_device', type=str, default='cpu',
                        help='Device for cache building')

    args = parser.parse_args()

    # Validate flags. --freeze_base (Phase 1) freezes the base encoder and the
    # existing struct_head, leaving only cross_distance_attention + cross_gate
    # trainable — isolates the contribution of the cross-residue distance
    # features on top of an already-trained baseline.
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

    # Derive mode string (model is always structure-only)
    mode = 'frozen_struct_cross' if args.freeze_base else 'struct'

    # Auto-generate run name
    if args.run_name is None:
        args.run_name = f'{args.data}_{mode}_fold{args.fold}'

    run_dir = os.path.join(args.output_dir, args.run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resolve data paths
    data_dir = DATASET_DIRS[args.data]
    cache_dir = args.cache_dir or os.path.join(data_dir, 'cache')

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
        # Note: don't pass --output_dir; let cache builder default to its
        # 1TB-drive location and symlink <data_dir>/cache -> there. train.py
        # reads from <data_dir>/cache afterwards either way.
        cmd = [
            sys.executable, '05_build_training_cache.py',
            '--data_dir', data_dir,
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
    # Normalization stats: read from fold_{args.fold}/config.json, which is the
    # config of the HELD-OUT (test) fold for this run. DEPENDENCY (review #5):
    # after the 05 fix, each fold_k/config.json carries TRAIN-ONLY stats for
    # fold k (i.e. computed from all folds EXCEPT k, the leave-one-fold-out
    # train set). Since args.fold IS the held-out fold here, those train-only
    # stats are exactly the right ones to apply to train+test for this run — no
    # test/holdout leakage. We must NOT instead build stats from the test fold's
    # own data. These stats are applied to BOTH train and test loaders below, so
    # they MUST be train-only; assert the cache advertises that provenance once
    # 05 stamps it (see TODO). Until 05 writes the flag, this is a soft check.
    config_path = os.path.join(cache_dir, f'fold_{args.fold}', 'config.json')
    with open(config_path) as f:
        cache_config = json.load(f)
    stats = cache_config['stats']
    shift_cols = cache_config['shift_cols']
    n_shifts = len(shift_cols)

    # ---- FIXED normalization (2026-06-08): replace per-fold data-derived per-AA
    # z-norm with citation-grounded constants (random-coil centers + BMRB-filtered
    # sigma) so normalization has ZERO cross-fold / held-out leakage and is identical
    # for every fold. Cache global stats are kept (used only to recover raw ppm from
    # the cache's global-z storage, exact). See fixed_norm.py / project_normalization_redesign.
    import fixed_norm
    stats = fixed_norm.build_fixed_per_aa(stats, shift_cols)
    print("  Normalization: FIXED random-coil + BMRB-filtered sigma (per-AA,nucleus); no data-derived stats")

    # TODO(05): once 05_build_training_cache.py stamps train-only provenance into
    # config.json (e.g. cache_config['stats_train_only'] = True for fold k), turn
    # this into a hard assert. For now warn if a global-stats marker is present.
    if cache_config.get('stats_train_only') is False or cache_config.get('stats_scope') == 'global':
        print("  WARNING: fold config stats are NOT train-only (global/all-data) -- "
              "normalization leak (review #5); rebuild caches with per-fold train-only stats.")

    test_ds = CachedRetrievalDataset.load(
        os.path.join(cache_dir, f'fold_{args.fold}'),
        n_shifts, stats=stats, shift_cols=shift_cols,
        mmap_structural=args.mmap_structural)

    train_parts = []
    for f in range(1, 6):
        if f == args.fold:
            continue
        fold_cache = os.path.join(cache_dir, f'fold_{f}')
        if not CachedRetrievalDataset.exists(fold_cache):
            print(f"ERROR: Cache not found at {fold_cache}")
            sys.exit(1)
        train_parts.append(CachedRetrievalDataset.load(
            fold_cache, n_shifts, stats=stats, shift_cols=shift_cols,
            mmap_structural=args.mmap_structural))
    n_struct = getattr(train_parts[0], 'n_struct_features', 49)
    if args.protein_subset:
        with open(args.protein_subset) as f:
            _subset = set(str(b) for b in json.load(f))
        def _filt(ds, name):
            keep = [i for i in range(len(ds))
                    if str(ds.idx_to_bmrb.get(str(int(ds.samples[i][0])))) in _subset]
            print(f"  [protein_subset] {name}: kept {len(keep)}/{len(ds)} samples")
            return Subset(ds, keep)
        test_ds = _filt(test_ds, f'test fold{args.fold}')
        train_parts = [_filt(p, f'train part {k}') for k, p in enumerate(train_parts)]
    train_ds = ConcatDataset(train_parts)

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
    model = create_model(
        n_atom_types=N_ATOM_TYPES,
        n_shifts=n_shifts,
        n_struct=n_struct,
        n_dssp=cache_config.get('n_dssp', 9),
        k_spatial=K_SPATIAL_NEIGHBORS,
    ).to(device)

    # ---- Feature ablation: zero a feature group at train+eval time ----
    _VALID_ABLATE = {'intra', 'cross', 'spatial', 'dssp', 'ss', 'bond_geom', 'residue_type'}
    if args.ablate:
        _abl = {a.strip() for a in args.ablate.split(',') if a.strip()}
        bad = _abl - _VALID_ABLATE
        if bad:
            raise SystemExit(f"--ablate: unknown feature(s) {sorted(bad)}; "
                             f"valid: {sorted(_VALID_ABLATE)}")
        model.ablate = _abl
        model.spatial_attention.ablate = _abl
        print(f"\n  ABLATION (zeroed at train+eval): {sorted(_abl)}")

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
    # Equivalence-weighted + swap-invariant loss structure (methyls/aromatics 1/N;
    # prochiral pairs scored swap-invariant). Per-residue, indexed by AA code.
    from config import STANDARD_RESIDUES
    _gw, _partner = fixed_norm.build_loss_tensors(shift_cols, STANDARD_RESIDUES)
    group_weight = _gw.to(device)
    partner = _partner.to(device)
    print(f"  Loss: equivalence-weighted + swap-invariant prochiral "
          f"({int((_partner >= 0).any(0).sum())} cols paired in some residue)")
    best_mae = float('inf')
    training_log = []

    print(f"\n{'='*70}")
    print(f"  Training: {mode} | {args.data} | fold {args.fold} | epochs {start_epoch}-{args.epochs}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        loss = train_epoch(model, train_loader, optimizer, device, scaler,
                           delta=args.huber_delta, shift_weights=shift_weights,
                           group_weight=group_weight, partner=partner)
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
                'dataset': args.data,
                'mode': mode,
            }, path)
            print(f"  Saved: {path}")

        # Always-current checkpoint (every epoch) for fine-grained mid-train resume
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': loss,
            'stats': stats,
            'shift_cols': shift_cols,
            'atom_to_idx': ATOM_TO_IDX,
            'dataset': args.data,
            'mode': mode,
        }, os.path.join(ckpt_dir, 'last.pt'))

        # Evaluate
        if epoch % 5 == 0 or epoch == args.epochs:
            results = evaluate(model, test_loader, device, stats, shift_cols,
                               args.huber_delta)
            # Checkpoint-selection objective = backbone-z-MAE (mean z over the 6
            # backbone nuclei), matching the canonical tree. Do NOT select on an
            # all-49 raw-ppm metric (dominated by wide-range sidechain carbons,
            # incomparable to canonical CV numbers). 'mae' is kept as a logging
            # alias of the same value for back-compat.
            select_metric = results['backbone_z_mae']
            epoch_log['mae'] = results['mae']
            epoch_log['backbone_z_mae'] = select_metric
            epoch_log['per_shift_mae'] = results['per_shift_mae']

            if select_metric < best_mae:
                best_mae = select_metric
                best_path = os.path.join(ckpt_dir, 'best.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'mae': results['mae'],
                    'backbone_z_mae': select_metric,
                    'per_shift_mae': results['per_shift_mae'],
                    'stats': stats,
                    'shift_cols': shift_cols,
                    'atom_to_idx': ATOM_TO_IDX,
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
