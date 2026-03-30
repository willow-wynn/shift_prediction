#!/usr/bin/env python3
"""
Train a pure structure-only model with NO retrieval pathway.

The model graph is:
  distance_attention → CNN → spatial_attention → struct_head → shifts

No retrieval components, no gates, no retrieval dropout.
The base encoder learns purely from structural features.

After training, the base_encoding (1472-dim) can be extracted as embeddings
for FAISS retrieval, then used to train retrieval components separately
with the base encoder frozen.

Usage:
    python train_structure_only.py --fold 1 --epochs 150
    python train_structure_only.py --fold 1 --epochs 150 --checkpoint data/struct_only/checkpoints/checkpoint_fold1_epoch100.pt
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
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from config import (
    N_ATOM_TYPES, N_RESIDUE_TYPES, N_SS_TYPES, N_MISMATCH_TYPES,
    DSSP_COLS, N_BOND_GEOM,
    DIST_ATTN_EMBED, DIST_ATTN_HIDDEN,
    CNN_CHANNELS, KERNEL_SIZE,
    INPUT_DROPOUT, LAYER_DROPOUTS, HEAD_DROPOUT,
    SPATIAL_ATTN_HIDDEN, K_SPATIAL_NEIGHBORS, K_RETRIEVED,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, HUBER_DELTA,
    WEIGHT_DECAY, BACKBONE_LOSS_WEIGHT,
)
from dataset import CachedRetrievalDataset
from model import (
    DistanceAttentionPerPosition, SpatialNeighborAttention,
    ResidualBlock1D,
)

GRAD_CLIP = 1.0
NUM_WORKERS = 4
PREFETCH = 2
BACKBONE_SHIFTS = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}


# ============================================================================
# Structure-Only Model (no retrieval components)
# ============================================================================

class StructureOnlyModel(nn.Module):
    """Pure structure model: distance attention + CNN + spatial attention + MLP head.

    No retrieval pathway, no gates. Clean graph for learning structural representations.
    """

    def __init__(
        self,
        n_atom_types=N_ATOM_TYPES,
        n_residue_types=N_RESIDUE_TYPES,
        n_ss_types=N_SS_TYPES,
        n_mismatch_types=N_MISMATCH_TYPES,
        n_dssp=len(DSSP_COLS),
        n_shifts=6,
        dist_attn_embed=DIST_ATTN_EMBED,
        dist_attn_hidden=DIST_ATTN_HIDDEN,
        cnn_channels=None,
        kernel=KERNEL_SIZE,
        input_dropout=INPUT_DROPOUT,
        layer_dropouts=None,
        head_dropout=HEAD_DROPOUT,
        spatial_hidden=SPATIAL_ATTN_HIDDEN,
        k_spatial=5,
    ):
        super().__init__()

        cnn_channels = cnn_channels or list(CNN_CHANNELS)
        layer_dropouts = layer_dropouts or list(LAYER_DROPOUTS)

        self.n_shifts = n_shifts

        # Distance attention
        self.distance_attention = DistanceAttentionPerPosition(
            n_atom_types=n_atom_types,
            embed_dim=dist_attn_embed,
            hidden_dim=dist_attn_hidden,
            dropout=0.25,
        )

        # Embeddings
        self.residue_embed = nn.Embedding(n_residue_types + 1, 64)
        self.ss_embed = nn.Embedding(n_ss_types + 1, 32)
        self.mismatch_embed = nn.Embedding(n_mismatch_types + 1, 16)
        self.valid_embed = nn.Linear(1, 16)

        if n_dssp > 0:
            self.dssp_proj = nn.Linear(n_dssp, 32)
            dssp_dim = 32
        else:
            self.dssp_proj = None
            dssp_dim = 0

        self.bond_proj = nn.Linear(N_BOND_GEOM, 16)
        bond_dim = 16

        cnn_input_dim = dist_attn_hidden + 64 + 32 + 16 + 16 + dssp_dim + bond_dim

        self.input_norm = nn.LayerNorm(cnn_input_dim)
        self.input_dropout_layer = nn.Dropout(input_dropout)

        # CNN
        cnn_layers = []
        in_ch = cnn_input_dim
        for out_ch, drop_p in zip(cnn_channels, layer_dropouts):
            cnn_layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            cnn_layers.append(nn.Dropout(drop_p))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_dim = cnn_channels[-1]

        # Spatial attention
        self.spatial_attention = SpatialNeighborAttention(
            n_residue_types=n_residue_types,
            n_ss_types=n_ss_types,
            k_neighbors=k_spatial,
            embed_dim=64,
            hidden_dim=spatial_hidden,
            dropout=0.30,
            dist_attn_hidden=dist_attn_hidden,
        )

        base_encoder_dim = cnn_out_dim + spatial_hidden

        # Prediction head
        self.struct_head = nn.Sequential(
            nn.Linear(base_encoder_dim, 1024), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(head_dropout * 0.5),
            nn.Linear(256, n_shifts),
        )

        with torch.no_grad():
            self.struct_head[-1].weight.zero_()
            self.struct_head[-1].bias.zero_()

    def forward(
        self,
        atom1_idx, atom2_idx, distances, dist_mask,
        residue_idx, ss_idx, mismatch_idx, is_valid,
        dssp_features,
        neighbor_res_idx, neighbor_ss_idx,
        neighbor_dist, neighbor_seq_sep, neighbor_angles,
        neighbor_valid,
        neighbor_atom1_idx=None, neighbor_atom2_idx=None,
        neighbor_distances=None, neighbor_dist_mask=None,
        bond_geom=None,
        # Accept and ignore retrieval inputs (from dataset)
        **kwargs,
    ):
        B = distances.size(0)

        # Distance attention
        dist_emb = self.distance_attention(atom1_idx, atom2_idx, distances, dist_mask)

        # Embeddings
        res_emb = self.residue_embed(residue_idx)
        ss_emb = self.ss_embed(ss_idx)
        mismatch_emb = self.mismatch_embed(mismatch_idx)
        valid_emb = self.valid_embed(is_valid.unsqueeze(-1))

        if bond_geom is None:
            bond_geom = torch.zeros(B, distances.size(1), N_BOND_GEOM, device=distances.device)
        bond_emb = self.bond_proj(bond_geom)

        if self.dssp_proj is not None:
            dssp_emb = self.dssp_proj(dssp_features)
            window_feat = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, dssp_emb, bond_emb], dim=-1)
        else:
            window_feat = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, bond_emb], dim=-1)

        x = self.input_dropout_layer(self.input_norm(window_feat))

        # CNN
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :]

        # Spatial neighbor distance embeddings
        neighbor_dist_embeddings = None
        if neighbor_atom1_idx is not None and neighbor_distances is not None:
            K_sp = neighbor_atom1_idx.size(1)
            M_sp = neighbor_atom1_idx.size(2)
            center_w = distances.size(1) // 2

            q_a1 = atom1_idx[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_a2 = atom2_idx[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_d = distances[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_m = dist_mask[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)

            ca_idx = 1
            ca_a1 = torch.full((B, K_sp, 1), ca_idx, dtype=torch.long, device=distances.device)
            ca_a2 = torch.full((B, K_sp, 1), ca_idx, dtype=torch.long, device=distances.device)
            ca_d = neighbor_dist.unsqueeze(-1)
            ca_m = neighbor_valid.unsqueeze(-1)

            joint_a1 = torch.cat([q_a1, ca_a1, neighbor_atom1_idx], dim=-1)
            joint_a2 = torch.cat([q_a2, ca_a2, neighbor_atom2_idx], dim=-1)
            joint_d = torch.cat([q_d, ca_d, neighbor_distances], dim=-1)
            joint_m = torch.cat([q_m, ca_m, neighbor_dist_mask], dim=-1)

            D_joint = joint_a1.size(-1)
            nb_a1 = joint_a1.reshape(B * K_sp, 1, D_joint)
            nb_a2 = joint_a2.reshape(B * K_sp, 1, D_joint)
            nb_d = joint_d.reshape(B * K_sp, 1, D_joint)
            nb_m = joint_m.reshape(B * K_sp, 1, D_joint)

            nb_dist_emb = self.distance_attention(nb_a1, nb_a2, nb_d, nb_m)
            neighbor_dist_embeddings = nb_dist_emb.squeeze(1).view(B, K_sp, -1)

        x_spatial = self.spatial_attention(
            neighbor_res_idx, neighbor_ss_idx,
            neighbor_dist, neighbor_seq_sep, neighbor_angles,
            neighbor_valid,
            neighbor_dist_embeddings=neighbor_dist_embeddings,
        )

        base_encoding = torch.cat([x_center, x_spatial], dim=-1)
        predictions = self.struct_head(base_encoding)

        predictions = torch.where(
            torch.isnan(predictions), torch.zeros_like(predictions), predictions)

        return predictions


# ============================================================================
# Training
# ============================================================================

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train pure structure-only model (no retrieval)')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--cache_dir', default='data/cache')
    parser.add_argument('--output_dir', default='data/struct_only/checkpoints')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--huber_delta', type=float, default=HUBER_DELTA)
    parser.add_argument('--save_every', type=int, default=25)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 70)
    print("STRUCTURE-ONLY MODEL TRAINING (no retrieval)")
    print("=" * 70)

    # Load stats from cache
    config_path = os.path.join(args.cache_dir, f'fold_{args.fold}', 'config.json')
    with open(config_path) as f:
        cache_config = json.load(f)
    stats = cache_config['stats']
    shift_cols = cache_config['shift_cols']
    n_shifts = len(shift_cols)

    print(f"  Fold:       {args.fold}")
    print(f"  Shifts:     {n_shifts}")
    print(f"  Device:     {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")

    # Load datasets
    test_ds = CachedRetrievalDataset.load(
        os.path.join(args.cache_dir, f'fold_{args.fold}'),
        n_shifts, K_RETRIEVED, stats=stats, shift_cols=shift_cols)

    train_parts = []
    for f in range(1, 6):
        if f == args.fold:
            continue
        fold_cache = os.path.join(args.cache_dir, f'fold_{f}')
        if not CachedRetrievalDataset.exists(fold_cache):
            print(f"ERROR: Cache not found at {fold_cache}")
            sys.exit(1)
        train_parts.append(CachedRetrievalDataset.load(
            fold_cache, n_shifts, K_RETRIEVED, stats=stats, shift_cols=shift_cols))
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

    print(f"  Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"  Test:  {len(test_ds):,} samples ({len(test_loader)} batches)")

    # Create model
    n_dssp = cache_config.get('n_dssp', 9)
    model = StructureOnlyModel(
        n_atom_types=N_ATOM_TYPES,
        n_shifts=n_shifts,
        n_dssp=n_dssp,
        k_spatial=K_SPATIAL_NEIGHBORS,
    ).to(device)

    start_epoch = 1
    if args.checkpoint:
        print(f"\n  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"  Resuming from epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=args.lr * 0.01)
    scaler = GradScaler('cuda') if device == 'cuda' else None

    if args.checkpoint and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if args.checkpoint and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    os.makedirs(args.output_dir, exist_ok=True)
    shift_weights = build_shift_weights(shift_cols, device)
    best_mae = float('inf')

    print(f"\n{'='*70}")
    print(f"  Training structure-only | fold {args.fold} | epochs {start_epoch}-{args.epochs}")
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
            path = os.path.join(args.output_dir, f'checkpoint_fold{args.fold}_epoch{epoch}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': loss,
                'stats': stats,
                'shift_cols': shift_cols,
                'model_type': 'structure_only',
            }, path)
            print(f"  Saved: {path}")

        if epoch % 5 == 0 or epoch == args.epochs:
            results = evaluate(model, test_loader, device, stats, shift_cols, args.huber_delta)
            if results['mae'] < best_mae:
                best_mae = results['mae']
                best_path = os.path.join(args.output_dir, f'best_struct_fold{args.fold}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'mae': results['mae'],
                    'per_shift_mae': results['per_shift_mae'],
                    'stats': stats,
                    'shift_cols': shift_cols,
                    'model_type': 'structure_only',
                }, best_path)
                print(f"  *** New best: {best_mae:.4f} ***")

    print(f"\nDone! Best MAE: {best_mae:.4f}")


if __name__ == '__main__':
    main()
