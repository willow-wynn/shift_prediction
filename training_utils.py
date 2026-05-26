"""
Shared training utilities for chemical shift prediction.

Contains the training loop, evaluation, loss functions, and helper functions
used by train.py. Consolidates logic that was previously duplicated across
main.py, 06_train.py, train_structure_only.py, and train_retrieval_frozen.py.
"""

import gc
import os
import sys
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    BACKBONE_LOSS_WEIGHT, STANDARD_RESIDUES, ATOM_TO_IDX, K_RETRIEVED,
)

# ============================================================================
# Constants
# ============================================================================

GRAD_CLIP = 1.0
NUM_WORKERS = 12
PREFETCH = 4
BACKBONE_SHIFTS = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}

# Base encoder module prefixes (for freezing)
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


# ============================================================================
# Loss Functions
# ============================================================================

def build_shift_weights(shift_cols, device):
    """Build per-shift loss weights. Backbone shifts 2x, sidechain 1x."""
    weights = torch.ones(len(shift_cols), device=device)
    for si, col in enumerate(shift_cols):
        if col in BACKBONE_SHIFTS:
            weights[si] = BACKBONE_LOSS_WEIGHT
    return weights


def huber_loss_masked(pred, target, mask, delta=0.5, shift_weights=None):
    """Huber loss over masked positions with optional per-shift weighting."""
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


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, scaler=None, delta=0.5,
                shift_weights=None):
    """Run one training epoch.

    Returns:
        avg_loss (float)
    """
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)
    nan_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)

    # === PROFILE INSTRUMENTATION (env: CLAUDE_PROFILE=1) ===
    import os as _os, time as _time
    _profile = _os.environ.get('CLAUDE_PROFILE', '0') == '1'
    _t_data = _t_h2d = _t_fwd = _t_bwd = _t_step = 0.0
    _n_prof = 0
    _t_iter_start = _time.perf_counter()
    # =======================================================

    for batch_idx, batch in enumerate(pbar):
        if _profile:
            _t_after_data = _time.perf_counter()
            _t_data += _t_after_data - _t_iter_start

        if device == 'cuda' and batch_idx > 0 and batch_idx % 50 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch.pop('shift_target')
        mask = batch.pop('shift_mask')

        if _profile:
            torch.cuda.synchronize() if device == 'cuda' else None
            _t_after_h2d = _time.perf_counter()
            _t_h2d += _t_after_h2d - _t_after_data

        if mask.sum() == 0:
            if _profile: _t_iter_start = _time.perf_counter()
            continue

        optimizer.zero_grad(set_to_none=True)

        ctx = autocast('cuda') if scaler else nullcontext()
        with ctx:
            pred = model(**batch)
            loss = huber_loss_masked(pred, target, mask, delta=delta,
                                     shift_weights=shift_weights)

        if _profile:
            torch.cuda.synchronize() if device == 'cuda' else None
            _t_after_fwd = _time.perf_counter()
            _t_fwd += _t_after_fwd - _t_after_h2d

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            if _profile: _t_iter_start = _time.perf_counter()
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

        if _profile:
            torch.cuda.synchronize() if device == 'cuda' else None
            _t_after_bwd = _time.perf_counter()
            _t_bwd += _t_after_bwd - _t_after_fwd
            _n_prof += 1
            if _n_prof in (50, 100, 150, 200):
                _total = _t_data + _t_h2d + _t_fwd + _t_bwd
                print(f"\n[PROF n={_n_prof}] data_wait={_t_data*1000/_n_prof:.1f}ms  "
                      f"h2d={_t_h2d*1000/_n_prof:.1f}ms  "
                      f"fwd={_t_fwd*1000/_n_prof:.1f}ms  "
                      f"bwd+step={_t_bwd*1000/_n_prof:.1f}ms  "
                      f"|  total={_total*1000/_n_prof:.1f}ms/batch  "
                      f"data_frac={100*_t_data/_total:.0f}%", flush=True)
            _t_iter_start = _time.perf_counter()

        bs = mask.sum()
        total_loss = total_loss + loss.detach() * bs
        total_count = total_count + bs

        if batch_idx % 10 == 0:
            avg = (total_loss / total_count).item() if total_count > 0 else 0.0
            pbar.set_postfix(loss=f"{avg:.4f}")

    if nan_batches > 0:
        print(f"  Warning: {nan_batches} batches skipped due to NaN")

    return (total_loss / total_count).item() if total_count > 0 else 0.0


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device, stats, shift_cols, delta=0.5):
    """Evaluate model, return per-shift MAE in ppm (denormalized with per-AA stats)."""
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

    per_aa_stats = stats.get('per_aa', {})

    per_shift_mae = {}
    for si, col in enumerate(shift_cols):
        m = all_mask[:, si]
        if m.sum() > 0 and col in stats:
            pred_ppm = torch.zeros(m.sum())
            true_ppm = torch.zeros(m.sum())
            masked_pred = all_pred[:, si][m]
            masked_target = all_target[:, si][m]
            masked_aa = all_aa[m]

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
# Freeze / Load Helpers
# ============================================================================

def load_base_checkpoint(model, checkpoint_path, device):
    """Load base encoder weights from a checkpoint into a model.

    Loads matching parameters by name/shape. Retrieval components initialize fresh.
    Returns the checkpoint dict for extracting stats/metadata.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    src_sd = ckpt.get('model_state_dict', ckpt)
    # Strip torch.compile prefix if present
    src_sd = {k.replace('_orig_mod.', ''): v for k, v in src_sd.items()}

    model_sd = model.state_dict()
    loaded = 0
    for name, param in src_sd.items():
        if name in model_sd and model_sd[name].shape == param.shape:
            model_sd[name] = param
            loaded += 1

    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded} parameters from checkpoint")
    print(f"  Remaining {len(model_sd) - loaded} parameters initialized fresh")
    return ckpt


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

    print(f"  Frozen: {frozen} parameters (base encoder)")
    print(f"  Trainable: {trainable} parameters (retrieval)")
    return trainable
