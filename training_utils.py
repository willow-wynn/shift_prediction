"""
Shared training utilities for chemical shift prediction.

Contains the training loop, evaluation, loss functions, and helper functions
used by train.py.
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
    BACKBONE_LOSS_WEIGHT, STANDARD_RESIDUES, ATOM_TO_IDX,
)
from shift_norm import denormalize, BACKBONE_SHIFTS as _BACKBONE_SHIFTS_CANON

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


def equiv_swap_huber_loss(pred, target, mask, aa_code, base_weights,
                          group_weight, partner, delta=0.5):
    """Huber loss with (a) magnetic-equivalence weighting (methyls/symmetric
    aromatics weighted 1/N via group_weight so a group counts once) and
    (b) swap-invariant scoring of prochiral pairs (min over the two label
    assignments). See equiv_groups.py / fixed_norm.build_loss_tensors.

    pred,target,mask: (B,S); aa_code:(B,) residue-type code; base_weights:(S,);
    group_weight:(n_aa,S); partner:(n_aa,S) long (partner column idx or -1).
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    n_aa = group_weight.size(0)
    aa = aa_code.clamp(0, n_aa - 1)
    w = base_weights.unsqueeze(0) * group_weight[aa]          # (B,S)
    part = partner[aa]                                        # (B,S), -1 = none
    h_ii = F.huber_loss(pred, target, reduction='none', delta=delta)
    has = part >= 0
    safe = part.clamp(min=0)
    t_part = torch.gather(target, 1, safe)
    m_part = torch.gather(mask.to(torch.float32), 1, safe) > 0.5
    h_ij = F.huber_loss(pred, t_part, reduction='none', delta=delta)
    both = has & mask & m_part                                # pair, both observed
    # direct = h(p_i,t_i)+h(p_j,t_j); swap = h(p_i,t_j)+h(p_j,t_i)
    pair_cost = torch.minimum(h_ii + torch.gather(h_ii, 1, safe),
                              h_ij + torch.gather(h_ij, 1, safe))
    elem = torch.where(both, 0.5 * pair_cost, h_ii)           # each pair member gets half
    loss = (elem * w)[mask].mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return loss


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, scaler=None, delta=0.5,
                shift_weights=None, group_weight=None, partner=None):
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
        # query AA code = center-window residue (residue_idx stays in batch for model)
        _ridx = batch['residue_idx']
        aa_code = _ridx[:, _ridx.size(1) // 2]

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
            if group_weight is not None and partner is not None:
                loss = equiv_swap_huber_loss(pred, target, mask, aa_code,
                                             shift_weights, group_weight, partner,
                                             delta=delta)
            else:
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
        _ridx = batch['residue_idx']
        aa_codes = _ridx[:, _ridx.size(1) // 2].cpu()   # center-window residue = query AA code
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

    # Denormalize predictions and targets to ppm ONCE via the canonical
    # per-AA denorm (shared with inference / 07_evaluate so the three can't
    # drift). Center-window AA per row in all_aa; out-of-range AA falls back to
    # global stats — identical to the previous scalar loop for valid residues.
    pred_ppm_all = denormalize(all_pred.numpy(), all_aa.numpy(), stats, shift_cols)
    true_ppm_all = denormalize(all_target.numpy(), all_aa.numpy(), stats, shift_cols)
    abs_ppm_all = np.abs(pred_ppm_all - true_ppm_all)
    mask_np = all_mask.numpy()

    per_shift_mae = {}   # ppm (per-AA denormalized) — for the full table
    per_shift_z = {}     # z-units (model's normalized space) — for the single number
    for si, col in enumerate(shift_cols):
        m = all_mask[:, si]
        if m.sum() > 0 and col in stats:
            masked_pred = all_pred[:, si][m]
            masked_target = all_target[:, si][m]
            # z-space MAE: targets/preds are already per-AA z-scored, so this is
            # the scale-free error (each nucleus weighted equally regardless of ppm range)
            per_shift_z[col] = (masked_pred - masked_target).abs().mean().item()
            col_mask = mask_np[:, si].astype(bool)
            per_shift_mae[col] = float(abs_ppm_all[col_mask, si].mean())

    # SINGLE NUMBER = mean z-MAE over the 6 BACKBONE nuclei only. NEVER include
    # sidechain (it's a bonus output), and NEVER raw-average ppm across nuclei
    # (different scales). This is the canonical checkpoint-selection objective;
    # train.py selects best.pt by 'backbone_z_mae' (alias 'mae'/'backbone_z').
    bb_cols = [c for c in shift_cols if c in _BACKBONE_SHIFTS_CANON and c in per_shift_z]
    backbone_z = sum(per_shift_z[c] for c in bb_cols) / len(bb_cols) if bb_cols else 0.0

    print(f"\n  Backbone-Z MAE (mean z over {len(bb_cols)} backbone nuclei): {backbone_z:.4f}")
    print("  backbone per-nucleus (ppm | z):")
    for col in _BACKBONE_SHIFTS_CANON:
        if col in per_shift_mae:
            print(f"    {col:10s}  {per_shift_mae[col]:.3f} ppm   {per_shift_z[col]:.3f} z")

    return {'mae': backbone_z,            # back-compat alias (logging)
            'backbone_z': backbone_z,     # back-compat alias
            'backbone_z_mae': backbone_z, # canonical checkpoint-selection key
            'per_shift_mae': per_shift_mae, 'per_shift_z': per_shift_z}


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
    """Freeze base encoder + struct_head, leave cross-residue features trainable (Phase 1)."""
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
    print(f"  Trainable: {trainable} parameters (cross-residue features)")
    return trainable
