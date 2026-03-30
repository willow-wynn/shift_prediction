#!/usr/bin/env python3
"""
Analyze why N shift gets the biggest boost from structure-bootstrapped retrieval.

Examines:
1. Gate behavior for N vs other backbone shifts
2. Per-AA breakdown: which amino acids benefit most for N?
3. Per-secondary-structure breakdown
4. Correlation between retrieval quality and N improvement
5. What structural features do the best N-shift neighbors share?
6. Compare retrieval contribution for N vs CA vs H
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
    STANDARD_RESIDUES, AA_3_TO_1, SS_TYPES, K_RETRIEVED,
    K_SPATIAL_NEIGHBORS, N_ATOM_TYPES,
)
from dataset import CachedRetrievalDataset
from model import create_model
from train_structure_only import StructureOnlyModel

BACKBONE = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}
device = 'cuda'


def main():
    cache_dir = 'data/struct_retrieval_v2/cache'
    with open(f'{cache_dir}/fold_1/config.json') as f:
        cc = json.load(f)
    stats = cc['stats']
    shift_cols = cc['shift_cols']
    per_aa = stats.get('per_aa', {})
    n_shifts = len(shift_cols)

    ds = CachedRetrievalDataset.load(
        f'{cache_dir}/fold_1', n_shifts, K_RETRIEVED,
        stats=stats, shift_cols=shift_cols)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    n_struct = getattr(ds, 'n_struct_features', 49)

    # Load both models
    print("Loading structure-only model...")
    s_model = StructureOnlyModel(
        n_atom_types=N_ATOM_TYPES, n_shifts=n_shifts,
        n_dssp=cc.get('n_dssp', 9), k_spatial=K_SPATIAL_NEIGHBORS).to(device)
    s_ckpt = torch.load('data/struct_only/checkpoints/best_struct_fold1.pt',
                         map_location=device, weights_only=False)
    s_model.load_state_dict(s_ckpt['model_state_dict'])
    s_model.eval()

    print("Loading retrieval model...")
    r_model = create_model(
        n_atom_types=N_ATOM_TYPES, n_shifts=n_shifts,
        n_struct=n_struct, n_dssp=cc.get('n_dssp', 9),
        k_spatial=K_SPATIAL_NEIGHBORS).to(device)
    r_ckpt = torch.load('data/struct_retrieval_v2/checkpoints_v3/best_retrieval_fold1.pt',
                         map_location=device, weights_only=False)
    r_model.load_state_dict(r_ckpt['model_state_dict'])
    r_model.eval()

    # Hook retrieval gate
    gate_vals = []
    def hook_gate(m, inp, out):
        gate_vals.append(out.detach().cpu())
    h_gate = r_model.retrieval_gate.register_forward_hook(hook_gate)

    # Collect predictions from both + gate + metadata
    print("\nCollecting predictions...")
    all_s, all_r, all_t, all_m = [], [], [], []
    all_aa, all_ss, all_gate = [], [], []
    all_n_valid, all_ret_dists = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            gate_vals.clear()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch.pop('shift_target')
            mask = batch.pop('shift_mask')
            aa = batch['query_residue_code'].cpu()
            ss = batch['ss_idx'][:, batch['ss_idx'].shape[1] // 2].cpu()

            # Count valid same-AA neighbors
            same_aa = (batch['retrieved_residue_codes'] == batch['query_residue_code'].unsqueeze(1))
            n_valid = (batch['retrieved_valid'] & same_aa).sum(dim=1).cpu()
            ret_dists = batch['retrieved_distances'].cpu()

            with autocast('cuda'):
                s_pred = s_model(**batch)
                r_pred = r_model(**batch)

            all_s.append(s_pred.cpu()); all_r.append(r_pred.cpu())
            all_t.append(target.cpu()); all_m.append(mask.cpu())
            all_aa.append(aa); all_ss.append(ss)
            all_n_valid.append(n_valid)
            all_ret_dists.append(ret_dists)
            if gate_vals:
                all_gate.append(gate_vals[0].squeeze(-1))

    h_gate.remove()

    all_s = torch.cat(all_s); all_r = torch.cat(all_r)
    all_t = torch.cat(all_t); all_m = torch.cat(all_m)
    all_aa = torch.cat(all_aa); all_ss = torch.cat(all_ss)
    all_gate = torch.cat(all_gate)
    all_n_valid = torch.cat(all_n_valid)
    all_ret_dists = torch.cat(all_ret_dists)

    def denorm(pred_vals, target_vals, aa_vals, col):
        """Per-AA denormalization."""
        n = len(pred_vals)
        p_ppm = torch.zeros(n)
        t_ppm = torch.zeros(n)
        for j in range(n):
            aa_idx = int(aa_vals[j])
            aa_name = STANDARD_RESIDUES[aa_idx] if aa_idx < len(STANDARD_RESIDUES) else None
            aa_s = per_aa.get(aa_name, {}).get(col) if aa_name else None
            if aa_s:
                p_ppm[j] = pred_vals[j] * aa_s['std'] + aa_s['mean']
                t_ppm[j] = target_vals[j] * aa_s['std'] + aa_s['mean']
            else:
                p_ppm[j] = pred_vals[j] * stats[col]['std'] + stats[col]['mean']
                t_ppm[j] = target_vals[j] * stats[col]['std'] + stats[col]['mean']
        return p_ppm, t_ppm

    # ==================== ANALYSIS ====================
    print("\n" + "=" * 70)
    print("WHY DOES N SHIFT GET THE BIGGEST RETRIEVAL BOOST?")
    print("=" * 70)

    # 1. Gate comparison across backbone shifts
    print("\n--- 1. Retrieval gate values (backbone shifts) ---")
    print(f"  {'Shift':>10s}  {'Gate mean':>10s}  {'Gate std':>9s}  {'Struct MAE':>10s}  {'Ret MAE':>8s}  {'Δ':>7s}")
    for col in ['ha_shift', 'h_shift', 'ca_shift', 'c_shift', 'cb_shift', 'n_shift']:
        si = shift_cols.index(col)
        m = all_m[:, si].bool()
        if m.sum() == 0: continue
        g = all_gate[:, si][m]

        sp, tp = denorm(all_s[:, si][m], all_t[:, si][m], all_aa[m], col)
        rp, _ = denorm(all_r[:, si][m], all_t[:, si][m], all_aa[m], col)

        mae_s = (sp - tp).abs().mean().item()
        mae_r = (rp - tp).abs().mean().item()
        print(f"  {col:>10s}  {g.mean():.4f}     {g.std():.4f}    {mae_s:.3f}      {mae_r:.3f}  {mae_r-mae_s:+.3f}")

    # 2. Per-AA N shift breakdown
    print("\n--- 2. Per amino acid: N shift improvement ---")
    n_si = shift_cols.index('n_shift')
    print(f"  {'AA':>5s}  {'Gate':>6s}  {'#Nbrs':>6s}  {'Struct':>7s}  {'Ret':>7s}  {'Δ':>7s}  {'Count':>7s}")

    for aa_idx, aa_name in enumerate(STANDARD_RESIDUES):
        if aa_name == 'UNK': continue
        m = all_m[:, n_si].bool() & (all_aa == aa_idx)
        if m.sum() < 50: continue

        g = all_gate[:, n_si][m]
        nv = all_n_valid[m].float()

        sp, tp = denorm(all_s[:, n_si][m], all_t[:, n_si][m], all_aa[m], 'n_shift')
        rp, _ = denorm(all_r[:, n_si][m], all_t[:, n_si][m], all_aa[m], 'n_shift')

        mae_s = (sp - tp).abs().mean().item()
        mae_r = (rp - tp).abs().mean().item()
        aa1 = AA_3_TO_1.get(aa_name, '?')
        print(f"  {aa1}({aa_name})  {g.mean():.3f}  {nv.mean():5.1f}  {mae_s:.3f}   {mae_r:.3f}  {mae_r-mae_s:+.3f}  {m.sum():6d}")

    # 3. Per-secondary-structure breakdown for N
    print("\n--- 3. Per secondary structure: N shift improvement ---")
    ss_names = {0: 'H(helix)', 1: 'E(sheet)', 2: 'C(coil)', 3: 'G(310)',
                4: 'I(pi)', 5: 'T(turn)', 6: 'S(bend)', 7: 'B(bridge)'}
    print(f"  {'SS':>12s}  {'Gate':>6s}  {'Struct':>7s}  {'Ret':>7s}  {'Δ':>7s}  {'Count':>7s}")

    for ss_idx in range(8):
        m = all_m[:, n_si].bool() & (all_ss == ss_idx)
        if m.sum() < 50: continue

        g = all_gate[:, n_si][m]
        sp, tp = denorm(all_s[:, n_si][m], all_t[:, n_si][m], all_aa[m], 'n_shift')
        rp, _ = denorm(all_r[:, n_si][m], all_t[:, n_si][m], all_aa[m], 'n_shift')

        mae_s = (sp - tp).abs().mean().item()
        mae_r = (rp - tp).abs().mean().item()
        print(f"  {ss_names.get(ss_idx, f'SS{ss_idx}'):>12s}  {g.mean():.3f}  {mae_s:.3f}   {mae_r:.3f}  {mae_r-mae_s:+.3f}  {m.sum():6d}")

    # 4. N improvement vs number of valid neighbors
    print("\n--- 4. N shift improvement by neighbor count ---")
    print(f"  {'Neighbors':>12s}  {'Gate':>6s}  {'Struct':>7s}  {'Ret':>7s}  {'Δ':>7s}  {'Count':>7s}")

    m_n = all_m[:, n_si].bool()
    nv_n = all_n_valid[m_n]

    for lo, hi in [(0, 1), (1, 5), (5, 10), (10, 20), (20, 32), (32, 999)]:
        bin_m = m_n.clone()
        bin_m[m_n] = (nv_n >= lo) & (nv_n < hi)
        if bin_m.sum() < 50: continue

        g = all_gate[:, n_si][bin_m]
        sp, tp = denorm(all_s[:, n_si][bin_m], all_t[:, n_si][bin_m], all_aa[bin_m], 'n_shift')
        rp, _ = denorm(all_r[:, n_si][bin_m], all_t[:, n_si][bin_m], all_aa[bin_m], 'n_shift')

        mae_s = (sp - tp).abs().mean().item()
        mae_r = (rp - tp).abs().mean().item()
        label = f"{lo}-{hi-1}" if hi < 999 else f"{lo}+"
        print(f"  {label:>12s}  {g.mean():.3f}  {mae_s:.3f}   {mae_r:.3f}  {mae_r-mae_s:+.3f}  {bin_m.sum():6d}")

    # 5. Error distribution: where does struct fail on N that retrieval fixes?
    print("\n--- 5. N shift: struct errors that retrieval fixes ---")
    sp_n, tp_n = denorm(all_s[:, n_si][m_n], all_t[:, n_si][m_n], all_aa[m_n], 'n_shift')
    rp_n, _ = denorm(all_r[:, n_si][m_n], all_t[:, n_si][m_n], all_aa[m_n], 'n_shift')

    s_err = (sp_n - tp_n).abs()
    r_err = (rp_n - tp_n).abs()
    improvement = s_err - r_err  # positive = retrieval helped

    print(f"  Mean struct error: {s_err.mean():.3f} ppm")
    print(f"  Mean ret error:    {r_err.mean():.3f} ppm")
    print(f"  Mean improvement:  {improvement.mean():.3f} ppm")
    print(f"  Retrieval helped:  {(improvement > 0).sum()}/{len(improvement)} ({100*(improvement > 0).float().mean():.1f}%)")
    print(f"  Retrieval hurt:    {(improvement < 0).sum()}/{len(improvement)} ({100*(improvement < 0).float().mean():.1f}%)")

    # Breakdown by struct error magnitude
    print("\n  By struct error magnitude:")
    print(f"  {'Struct err':>12s}  {'Count':>7s}  {'Helped%':>8s}  {'Mean Δ':>8s}")
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 100.0)]:
        bin_mask = (s_err >= lo) & (s_err < hi)
        if bin_mask.sum() < 10: continue
        helped = (improvement[bin_mask] > 0).float().mean().item()
        mean_delta = improvement[bin_mask].mean().item()
        print(f"  {lo:.1f}-{hi:.1f} ppm  {bin_mask.sum():6d}  {100*helped:6.1f}%  {mean_delta:+.3f}")

    # 6. Compare: what makes N special vs CA and H?
    print("\n--- 6. Why N and not CA or H? ---")
    for col in ['n_shift', 'ca_shift', 'h_shift']:
        si = shift_cols.index(col)
        m = all_m[:, si].bool()
        if m.sum() == 0: continue

        sp, tp = denorm(all_s[:, si][m], all_t[:, si][m], all_aa[m], col)
        rp, _ = denorm(all_r[:, si][m], all_t[:, si][m], all_aa[m], col)

        s_err = (sp - tp).abs()
        r_err = (rp - tp).abs()
        imp = s_err - r_err

        # Correlation between struct error and improvement
        corr = np.corrcoef(s_err.numpy(), imp.numpy())[0, 1]

        # What fraction of the error does retrieval fix?
        frac_fixed = imp.sum() / s_err.sum()

        g = all_gate[:, si][m]

        print(f"\n  {col}:")
        print(f"    Gate mean: {g.mean():.3f}")
        print(f"    Struct MAE: {s_err.mean():.3f}, Ret MAE: {r_err.mean():.3f}")
        print(f"    Fraction of error fixed by retrieval: {frac_fixed:.1%}")
        print(f"    Correlation(struct_error, improvement): {corr:.3f}")
        print(f"    Struct error std: {s_err.std():.3f} (higher = more variable = more room to help)")
        print(f"    Helped/Hurt: {(imp>0).sum()}/{(imp<0).sum()}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
