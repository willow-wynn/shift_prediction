#!/usr/bin/env python3
"""
Train retrieval components with a frozen base encoder.

Phase 2 of the two-phase training:
1. Load structure-only checkpoint into the full retrieval model
2. Freeze: distance_attention, CNN, spatial_attention, struct_head, embeddings
3. Train only: neighbor_encoder, self_attn_layers, cross_attn_layers,
   cross_ffn_layers, direct_transfer, retrieval_gate

Uses caches built from structure-model embeddings (not ESM).

Usage:
    python train_retrieval_frozen.py \
        --struct_checkpoint data/struct_only/checkpoints/best_struct_fold1.pt \
        --cache_dir data/struct_retrieval/cache \
        --fold 1 --epochs 150
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
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from config import (
    N_ATOM_TYPES, N_RESIDUE_TYPES, N_SS_TYPES, N_MISMATCH_TYPES,
    DSSP_COLS, K_SPATIAL_NEIGHBORS, K_RETRIEVED,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, HUBER_DELTA,
    WEIGHT_DECAY, BACKBONE_LOSS_WEIGHT,
)
from dataset import CachedRetrievalDataset
from model import create_model

GRAD_CLIP = 1.0
NUM_WORKERS = 4
PREFETCH = 2
BACKBONE_SHIFTS = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}

# These are the base encoder modules that get FROZEN
FROZEN_PREFIXES = [
    'distance_attention.',
    'residue_embed.',
    'ss_embed.',
    'mismatch_embed.',
    'valid_embed.',
    'dssp_proj.',
    'bond_proj.',
    'input_norm.',
    'input_dropout_layer.',
    'cnn.',
    'spatial_attention.',
    'struct_head.',
]


def load_struct_into_full_model(model, struct_checkpoint_path, device):
    """Load structure-only weights into the full retrieval model.

    Maps matching parameter names. Retrieval components initialize fresh.
    """
    ckpt = torch.load(struct_checkpoint_path, map_location=device, weights_only=False)
    struct_sd = ckpt['model_state_dict']

    model_sd = model.state_dict()
    loaded = 0
    for name, param in struct_sd.items():
        if name in model_sd and model_sd[name].shape == param.shape:
            model_sd[name] = param
            loaded += 1

    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded} parameters from structure checkpoint")
    print(f"  Remaining {len(model_sd) - loaded} parameters initialized fresh (retrieval)")
    return ckpt.get('stats', {}), ckpt.get('shift_cols', [])


def freeze_base_encoder(model):
    """Freeze all base encoder parameters, leave retrieval trainable."""
    frozen = 0
    trainable = 0

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in FROZEN_PREFIXES):
            param.requires_grad = False
            frozen += 1
        else:
            param.requires_grad = True
            trainable += 1

    print(f"  Frozen: {frozen} parameters (base encoder + struct_head)")
    print(f"  Trainable: {trainable} parameters (retrieval components)")
    return trainable


def build_shift_weights(shift_cols, device):
    weights = torch.ones(len(shift_cols), device=device)
    for si, col in enumerate(shift_cols):
        if col in BACKBONE_SHIFTS:
            weights[si] = BACKBONE_LOSS_WEIGHT
    return weights


def huber_loss_masked(pred, target, mask, delta=0.5, shift_weights=None):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    if shift_weights is not None:
        elem_loss = F.huber_loss(pred, target, reduction='none', delta=delta)
        loss = (elem_loss * shift_weights.unsqueeze(0))[mask].mean()
    else:
        loss = F.huber_loss(pred[mask], target[mask], reduction='mean', delta=delta)
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return loss


def train_epoch(model, loader, optimizer, device, scaler, delta, shift_weights):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if device == 'cuda' and batch_idx > 0 and batch_idx % 50 == 0:
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
            loss = huber_loss_masked(pred, target, mask, delta=delta, shift_weights=shift_weights)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            optimizer.step()

        bs = mask.sum()
        total_loss = total_loss + loss.detach() * bs
        total_count = total_count + bs
        if batch_idx % 10 == 0:
            avg = (total_loss / total_count).item() if total_count > 0 else 0.0
            pbar.set_postfix(loss=f"{avg:.4f}")

    return (total_loss / total_count).item() if total_count > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device, stats, shift_cols, delta=0.5):
    model.eval()
    all_pred, all_target, all_mask, all_aa = [], [], [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch.pop('shift_target')
        mask = batch.pop('shift_mask')
        aa_codes = batch['query_residue_code'].cpu()
        ctx = autocast('cuda') if device == 'cuda' else nullcontext()
        with ctx:
            pred = model(**batch)
        all_pred.append(pred.cpu())
        all_target.append(target.cpu())
        all_mask.append(mask.cpu())
        all_aa.append(aa_codes)

    all_pred = torch.cat(all_pred)
    all_target = torch.cat(all_target)
    all_mask = torch.cat(all_mask)
    all_aa = torch.cat(all_aa)

    from config import STANDARD_RESIDUES
    per_aa_stats = stats.get('per_aa', {})

    per_shift_mae = {}
    for si, col in enumerate(shift_cols):
        m = all_mask[:, si]
        if m.sum() > 0 and col in stats:
            masked_pred = all_pred[:, si][m]
            masked_target = all_target[:, si][m]
            masked_aa = all_aa[m]

            pred_ppm = torch.zeros(m.sum())
            true_ppm = torch.zeros(m.sum())
            for j in range(len(masked_pred)):
                aa_idx = int(masked_aa[j])
                aa_name = STANDARD_RESIDUES[aa_idx] if aa_idx < len(STANDARD_RESIDUES) else None
                aa_s = per_aa_stats.get(aa_name, {}).get(col) if aa_name else None
                if aa_s:
                    pred_ppm[j] = masked_pred[j] * aa_s['std'] + aa_s['mean']
                    true_ppm[j] = masked_target[j] * aa_s['std'] + aa_s['mean']
                else:
                    pred_ppm[j] = masked_pred[j] * stats[col]['std'] + stats[col]['mean']
                    true_ppm[j] = masked_target[j] * stats[col]['std'] + stats[col]['mean']

            per_shift_mae[col] = (pred_ppm - true_ppm).abs().mean().item()

    overall = sum(per_shift_mae.values()) / len(per_shift_mae) if per_shift_mae else 0.0

    bb = {k: v for k, v in per_shift_mae.items() if k in BACKBONE_SHIFTS}
    sc = {k: v for k, v in per_shift_mae.items() if k not in BACKBONE_SHIFTS}
    bb_mae = sum(bb.values()) / len(bb) if bb else 0.0
    sc_mae = sum(sc.values()) / len(sc) if sc else 0.0

    print(f"\n  Overall MAE: {overall:.4f}  |  Backbone: {bb_mae:.4f}  |  Sidechain: {sc_mae:.4f}")
    for col, mae in sorted(per_shift_mae.items(), key=lambda x: x[1]):
        marker = '*' if col in BACKBONE_SHIFTS else ' '
        print(f"  {marker} {col:20s}: {mae:.3f}")

    return {'mae': overall, 'per_shift_mae': per_shift_mae}


def main():
    parser = argparse.ArgumentParser(
        description='Train retrieval components with frozen base encoder')
    parser.add_argument('--struct_checkpoint', required=True,
                        help='Path to structure-only model checkpoint')
    parser.add_argument('--cache_dir', default='data/struct_retrieval/cache',
                        help='Cache dir with structure-embedding retrieval')
    parser.add_argument('--output_dir', default='data/struct_retrieval/checkpoints')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--huber_delta', type=float, default=HUBER_DELTA)
    parser.add_argument('--save_every', type=int, default=25)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume retrieval training from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 70)
    print("RETRIEVAL TRAINING (frozen base encoder)")
    print("=" * 70)

    # Load stats from cache
    config_path = os.path.join(args.cache_dir, f'fold_{args.fold}', 'config.json')
    with open(config_path) as f:
        cache_config = json.load(f)
    stats = cache_config['stats']
    shift_cols = cache_config['shift_cols']
    n_shifts = len(shift_cols)
    n_dssp = cache_config.get('n_dssp', 9)

    print(f"  Fold:       {args.fold}")
    print(f"  Shifts:     {n_shifts}")
    print(f"  Device:     {device}")

    # Load datasets
    test_ds = CachedRetrievalDataset.load(
        os.path.join(args.cache_dir, f'fold_{args.fold}'),
        n_shifts, K_RETRIEVED, stats=stats, shift_cols=shift_cols)

    train_parts = []
    for f in range(1, 6):
        if f == args.fold:
            continue
        fold_cache = os.path.join(args.cache_dir, f'fold_{f}')
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

    print(f"  Train: {len(train_ds):,} ({len(train_loader)} batches)")
    print(f"  Test:  {len(test_ds):,} ({len(test_loader)} batches)")

    # Create full model and load structure weights
    print(f"\n  Loading structure checkpoint: {args.struct_checkpoint}")
    model = create_model(
        n_atom_types=N_ATOM_TYPES,
        n_shifts=n_shifts,
        n_struct=n_struct,
        n_dssp=n_dssp,
        k_spatial=K_SPATIAL_NEIGHBORS,
    ).to(device)

    load_struct_into_full_model(model, args.struct_checkpoint, device)
    freeze_base_encoder(model)

    start_epoch = 1
    if args.checkpoint:
        print(f"  Resuming from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        freeze_base_encoder(model)  # Re-freeze after full load
        start_epoch = ckpt.get('epoch', 0) + 1

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_total:,}  Trainable: {n_trainable:,}")

    # Only optimize trainable params
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=args.lr * 0.01)
    scaler = GradScaler('cuda') if device == 'cuda' else None

    os.makedirs(args.output_dir, exist_ok=True)
    shift_weights = build_shift_weights(shift_cols, device)
    best_mae = float('inf')

    print(f"\n{'='*70}")
    print(f"  Training retrieval | fold {args.fold} | epochs {start_epoch}-{args.epochs}")
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

        if epoch % args.save_every == 0:
            path = os.path.join(args.output_dir, f'retrieval_ckpt_fold{args.fold}_epoch{epoch}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': loss,
                'stats': stats,
                'shift_cols': shift_cols,
                'struct_checkpoint': args.struct_checkpoint,
            }, path)
            print(f"  Saved: {path}")

        if epoch % 5 == 0 or epoch == args.epochs:
            results = evaluate(model, test_loader, device, stats, shift_cols, args.huber_delta)
            if results['mae'] < best_mae:
                best_mae = results['mae']
                best_path = os.path.join(args.output_dir, f'best_retrieval_fold{args.fold}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'mae': results['mae'],
                    'per_shift_mae': results['per_shift_mae'],
                    'stats': stats,
                    'shift_cols': shift_cols,
                    'struct_checkpoint': args.struct_checkpoint,
                }, best_path)
                print(f"  *** New best: {best_mae:.4f} ***")

    print(f"\nDone! Best MAE: {best_mae:.4f}")


if __name__ == '__main__':
    main()
