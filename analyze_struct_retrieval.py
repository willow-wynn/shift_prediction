#!/usr/bin/env python3
"""
Analyze structure-bootstrapped retrieval model behavior.

Compares struct-only vs retrieval predictions, examines gate behavior,
identifies which shifts retrieval helps/hurts.
"""

import json, os, sys, gc
import numpy as np
import torch
from torch.amp import autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    STANDARD_RESIDUES, AA_3_TO_1, K_RETRIEVED,
    K_SPATIAL_NEIGHBORS, N_ATOM_TYPES, N_BOND_GEOM,
)
from dataset import CachedRetrievalDataset
from model import create_model
from train_structure_only import StructureOnlyModel

BACKBONE = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}
device = 'cuda'


def eval_with_hooks(model, loader, stats, shift_cols, per_aa, desc="Eval"):
    """Run model and capture gate values + struct-only predictions."""
    model.eval()

    gate_vals = []
    struct_preds = []

    def hook_gate(module, inp, out):
        gate_vals.append(out.detach().cpu())
    def hook_struct(module, inp, out):
        struct_preds.append(out.detach().cpu())

    h1 = model.retrieval_gate.register_forward_hook(hook_gate)
    h2 = model.struct_head.register_forward_hook(hook_struct)

    all_pred, all_target, all_mask, all_aa = [], [], [], []
    all_gate, all_struct = [], []
    all_n_valid = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            gate_vals.clear()
            struct_preds.clear()

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch.pop('shift_target')
            mask = batch.pop('shift_mask')
            aa = batch['query_residue_code'].cpu()

            # Count valid same-AA neighbors
            same_aa = (batch['retrieved_residue_codes'] == batch['query_residue_code'].unsqueeze(1))
            n_valid = (batch['retrieved_valid'] & same_aa).sum(dim=1).cpu()

            with autocast('cuda'):
                pred = model(**batch)

            all_pred.append(pred.cpu())
            all_target.append(target.cpu())
            all_mask.append(mask.cpu())
            all_aa.append(aa)
            all_n_valid.append(n_valid)
            if gate_vals:
                all_gate.append(gate_vals[0].squeeze(-1))
            if struct_preds:
                all_struct.append(struct_preds[0])

    h1.remove()
    h2.remove()

    return {
        'pred': torch.cat(all_pred),
        'target': torch.cat(all_target),
        'mask': torch.cat(all_mask),
        'aa': torch.cat(all_aa),
        'gate': torch.cat(all_gate) if all_gate else None,
        'struct_pred': torch.cat(all_struct) if all_struct else None,
        'n_valid': torch.cat(all_n_valid),
    }


def denorm_mae(pred, target, mask, aa, si, col, stats, per_aa):
    """Compute MAE with correct per-AA denormalization."""
    m = mask[:, si]
    if m.sum() == 0:
        return None, 0

    mp = pred[:, si][m]
    mt = target[:, si][m]
    ma = aa[m]

    pred_ppm = torch.zeros(m.sum())
    true_ppm = torch.zeros(m.sum())
    for j in range(len(mp)):
        aa_idx = int(ma[j])
        aa_name = STANDARD_RESIDUES[aa_idx] if aa_idx < len(STANDARD_RESIDUES) else None
        aa_s = per_aa.get(aa_name, {}).get(col) if aa_name else None
        if aa_s:
            pred_ppm[j] = mp[j] * aa_s['std'] + aa_s['mean']
            true_ppm[j] = mt[j] * aa_s['std'] + aa_s['mean']
        else:
            pred_ppm[j] = mp[j] * stats[col]['std'] + stats[col]['mean']
            true_ppm[j] = mt[j] * stats[col]['std'] + stats[col]['mean']

    return (pred_ppm - true_ppm).abs().mean().item(), int(m.sum())


def main():
    # Load cache config
    cache_dir = 'data/struct_retrieval_v2/cache'
    with open(f'{cache_dir}/fold_1/config.json') as f:
        cc = json.load(f)
    stats = cc['stats']
    shift_cols = cc['shift_cols']
    per_aa = stats.get('per_aa', {})
    n_shifts = len(shift_cols)
    n_dssp = cc.get('n_dssp', 9)

    test_ds = CachedRetrievalDataset.load(
        f'{cache_dir}/fold_1', n_shifts, K_RETRIEVED,
        stats=stats, shift_cols=shift_cols)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    n_struct = getattr(test_ds, 'n_struct_features', 49)

    # Load retrieval model
    print("Loading retrieval model...")
    ret_model = create_model(
        n_atom_types=N_ATOM_TYPES, n_shifts=n_shifts,
        n_struct=n_struct, n_dssp=n_dssp,
        k_spatial=K_SPATIAL_NEIGHBORS,
    ).to(device)
    ckpt = torch.load('data/struct_retrieval_v2/checkpoints/best_retrieval_fold1.pt',
                       map_location=device, weights_only=False)
    ret_model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Epoch {ckpt['epoch']}")

    print("\nCollecting retrieval model predictions + gates...")
    ret_results = eval_with_hooks(ret_model, loader, stats, shift_cols, per_aa, "Retrieval")

    del ret_model
    torch.cuda.empty_cache()
    gc.collect()

    # ==================== ANALYSIS ====================
    print("\n" + "=" * 70)
    print("RETRIEVAL ANALYSIS")
    print("=" * 70)

    gate = ret_results['gate']  # (N, n_shifts) - retrieval gate values
    struct_pred = ret_results['struct_pred']  # (N, n_shifts)
    final_pred = ret_results['pred']
    target = ret_results['target']
    mask = ret_results['mask']
    aa = ret_results['aa']
    n_valid = ret_results['n_valid']

    # 1. Gate statistics per shift
    print("\n--- Gate values per shift (higher = more retrieval) ---")
    print(f"  {'Shift':>20s}  {'Gate mean':>10s}  {'Gate std':>10s}  {'MAE':>8s}  {'Struct MAE':>10s}  {'Delta':>8s}")

    for si, col in enumerate(shift_cols):
        m = mask[:, si].bool()
        if m.sum() < 100:
            continue

        g = gate[:, si][m]

        # MAE of final (retrieval-blended) prediction
        mae_ret, _ = denorm_mae(final_pred, target, mask, aa, si, col, stats, per_aa)
        # MAE of struct-only prediction (from the same model's struct_head)
        mae_struct, _ = denorm_mae(struct_pred, target, mask, aa, si, col, stats, per_aa)

        if mae_ret is None or mae_struct is None:
            continue

        delta = mae_ret - mae_struct
        marker = '*' if col in BACKBONE else ' '
        status = "WORSE" if delta > 0.01 else ("better" if delta < -0.01 else "~same")
        print(f"  {marker}{col:>19s}  {g.mean():.4f}     {g.std():.4f}     {mae_ret:.3f}   {mae_struct:.3f}      {delta:+.3f}  {status}")

    # 2. Gate vs number of valid neighbors
    print("\n--- Gate value vs # valid same-AA neighbors ---")
    for n_bin in [0, 1, 5, 10, 20, 32]:
        if n_bin == 0:
            bin_mask = n_valid == 0
            label = "0 neighbors"
        elif n_bin < 32:
            bin_mask = (n_valid >= n_bin) & (n_valid < n_bin + 5)
            label = f"{n_bin}-{n_bin+4} neighbors"
        else:
            bin_mask = n_valid >= n_bin
            label = f"{n_bin}+ neighbors"

        if bin_mask.sum() < 10:
            continue

        # Average gate across all shifts for residues in this bin
        g_mean = gate[bin_mask].mean().item()
        count = bin_mask.sum().item()
        print(f"  {label:>20s}: gate={g_mean:.4f}  (n={count:,})")

    # 3. Per-AA gate behavior
    print("\n--- Per amino acid: gate mean + retrieval impact on backbone ---")
    print(f"  {'AA':>5s}  {'Gate':>6s}  {'#Valid':>6s}  {'CA Δ':>7s}  {'N Δ':>7s}  {'H Δ':>7s}  {'Count':>7s}")

    for aa_idx, aa_name in enumerate(STANDARD_RESIDUES):
        if aa_name == 'UNK':
            continue
        aa_mask = (aa == aa_idx)
        if aa_mask.sum() < 50:
            continue

        g_mean = gate[aa_mask].mean().item()
        nv_mean = n_valid[aa_mask].float().mean().item()
        count = aa_mask.sum().item()

        deltas = {}
        for col in ['ca_shift', 'n_shift', 'h_shift']:
            si = shift_cols.index(col) if col in shift_cols else -1
            if si < 0:
                continue
            m = mask[:, si].bool() & aa_mask
            if m.sum() < 10:
                deltas[col] = float('nan')
                continue

            # Create per-AA subset mask
            sub_mask = torch.zeros_like(mask)
            sub_mask[m, si] = True

            mae_r, _ = denorm_mae(final_pred, target, sub_mask, aa, si, col, stats, per_aa)
            mae_s, _ = denorm_mae(struct_pred, target, sub_mask, aa, si, col, stats, per_aa)
            deltas[col] = (mae_r - mae_s) if mae_r and mae_s else float('nan')

        aa1 = AA_3_TO_1.get(aa_name, '?')
        print(f"  {aa1}({aa_name})  {g_mean:.3f}  {nv_mean:5.1f}  "
              f"{deltas.get('ca_shift', float('nan')):+.3f}  "
              f"{deltas.get('n_shift', float('nan')):+.3f}  "
              f"{deltas.get('h_shift', float('nan')):+.3f}  "
              f"{count:6d}")

    # 4. Worst sidechain shifts — what's happening?
    print("\n--- Worst sidechain shifts: gate + neighbor analysis ---")
    worst_shifts = ['cd1_shift', 'cd2_shift', 'cg_shift', 'cz_shift', 'ne2_shift', 'cd_shift']
    for col in worst_shifts:
        if col not in shift_cols:
            continue
        si = shift_cols.index(col)
        m = mask[:, si].bool()
        if m.sum() < 5:
            continue

        g = gate[:, si][m]
        nv = n_valid[m].float()

        mae_r, cnt = denorm_mae(final_pred, target, mask, aa, si, col, stats, per_aa)
        mae_s, _ = denorm_mae(struct_pred, target, mask, aa, si, col, stats, per_aa)

        print(f"  {col:>15s}: gate={g.mean():.3f}±{g.std():.3f}  "
              f"neighbors={nv.mean():.1f}  "
              f"MAE: struct={mae_s:.3f} ret={mae_r:.3f} Δ={mae_r-mae_s:+.3f}  "
              f"(n={cnt})")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
