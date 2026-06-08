#!/usr/bin/env python3
"""
Single-PDB inference: predict chemical shifts from a PDB structure.

Usage:
    python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb
    python inference.py --model data/checkpoints/best_fold1.pt --pdb 1UBQ --chain A -o shifts.csv
"""

import argparse
import json
import os
import sys
import tempfile
import urllib.request
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import torch
from torch.amp import autocast
from contextlib import nullcontext

from config import (
    ATOM_TYPES, ATOM_TO_IDX, N_ATOM_TYPES,
    STANDARD_RESIDUES, RESIDUE_TO_IDX, N_RESIDUE_TYPES,
    SS_TYPES, SS_TO_IDX, N_SS_TYPES,
    N_MISMATCH_TYPES,
    DSSP_COLS, K_SPATIAL_NEIGHBORS,
    CONTEXT_WINDOW, MAX_VALID_DISTANCES,
    STRUCT_DIST_COLS, STRUCT_SC_COLS, N_STRUCT_FEATURES,
    AA_3_TO_1,
    N_BOND_GEOM,
    MAX_CROSS_DISTANCES, CROSS_DIST_CUTOFF, CROSS_H_CUTOFF,
    N_CROSS_OFFSET_TYPES,
)
from model import ShiftPredictor
from shift_norm import denormalize, load_model
from config import BOND_GEOM_COLS
from pdb_utils import parse_pdb, run_dssp
from structure_selection import select_best_nmr_model
from distance_features import (
    compute_all_distance_features,
    calc_cross_residue_distances,
    build_cross_arrays_for_residue,
)
from spatial_neighbors import find_neighbors


# ============================================================================
# Model Loading (auto-detect architecture from checkpoint)
# ============================================================================

# load_model now lives in shift_norm.py (single shared auto-detect loader,
# imported above) — formerly duplicated here and in 07_evaluate.py with
# diverging n_shifts probing. The returned info dict is a superset of the keys
# this file uses ('stats', 'shift_cols', 'atom_to_idx', 'n_atom_types', 'epoch').


# ============================================================================
# Feature Extraction from PDB
# ============================================================================

def extract_features_from_pdb(pdb_path, chain_id=None, atom_to_idx=None, dssp_stats=None):
    """Extract all features needed for the model from a PDB file.

    Returns:
        residues: list of dicts with all features per residue
        residue_ids: list of residue IDs
    """
    if atom_to_idx is None:
        atom_to_idx = ATOM_TO_IDX

    n_atom_types = len(atom_to_idx)

    # Parse PDB. For multi-model NMR ensembles, use the SAME conformer the
    # cache builder used (median-closest model via select_best_nmr_model), so
    # inference coordinates match training. DSSP is still run on the file
    # (model 0) below — mirroring 01_build_datasets.process_protein exactly,
    # which also takes best-model coords + model-0 DSSP.
    print(f"  Parsing PDB: {pdb_path}")
    pdb_data = select_best_nmr_model(pdb_path, chain_id)
    if not pdb_data:
        print("  ERROR: No residues found in PDB")
        return [], []

    # Filter to standard amino acids
    aa_data = {}
    for (chain, res_id), res_info in sorted(pdb_data.items()):
        res_name = res_info['residue_name']
        if res_name in AA_3_TO_1 or res_name in RESIDUE_TO_IDX:
            aa_data[res_id] = res_info

    if not aa_data:
        print("  ERROR: No standard amino acids found")
        return [], []

    res_ids = sorted(aa_data.keys())
    print(f"  Found {len(res_ids)} residues")

    # Run DSSP
    print(f"  Running DSSP...")
    dssp_data = run_dssp(pdb_path)

    # Compute distance features for each residue
    print(f"  Computing distance features...")
    all_dist_features = {}
    for rid in res_ids:
        atoms = aa_data[rid]['atoms']
        all_dist_features[rid] = compute_all_distance_features(atoms)

    # Compute spatial neighbors
    print(f"  Computing spatial neighbors...")
    spatial_data = {rid: aa_data[rid] for rid in res_ids}
    neighbors = find_neighbors(spatial_data, k=K_SPATIAL_NEIGHBORS)

    # Pre-compute inter-residue bond geometry.
    # Index by BOND_GEOM_COLS name (not hardcoded 0..3) so this stays in lockstep
    # with the cache builder's column ordering if BOND_GEOM_COLS ever changes.
    print(f"  Computing bond geometry...")
    _BG_CA_PREV = BOND_GEOM_COLS.index('bond_ca_prev')
    _BG_CA_NEXT = BOND_GEOM_COLS.index('bond_ca_next')
    _BG_PEP_FWD = BOND_GEOM_COLS.index('bond_peptide_fwd')
    _BG_PEP_BKWD = BOND_GEOM_COLS.index('bond_peptide_bkwd')
    bond_geom_by_rid = {}
    for i, rid in enumerate(res_ids):
        bg = np.zeros(N_BOND_GEOM, dtype=np.float32)
        atoms_i = aa_data[rid]['atoms']

        # Previous residue
        if i > 0:
            prev_rid = res_ids[i - 1]
            atoms_prev = aa_data[prev_rid]['atoms']
            if 'CA' in atoms_i and 'CA' in atoms_prev:
                ca_i = atoms_i['CA']
                ca_prev = atoms_prev['CA']
                if np.all(np.isfinite(ca_i)) and np.all(np.isfinite(ca_prev)):
                    bg[_BG_CA_PREV] = np.linalg.norm(ca_i - ca_prev) / 10.0

            if 'C' in atoms_prev and 'N' in atoms_i:
                c_prev = atoms_prev['C']
                n_i = atoms_i['N']
                if np.all(np.isfinite(c_prev)) and np.all(np.isfinite(n_i)):
                    bg[_BG_PEP_BKWD] = np.linalg.norm(n_i - c_prev) / 10.0

        # Next residue
        if i < len(res_ids) - 1:
            next_rid = res_ids[i + 1]
            atoms_next = aa_data[next_rid]['atoms']
            if 'CA' in atoms_i and 'CA' in atoms_next:
                ca_i = atoms_i['CA']
                ca_next = atoms_next['CA']
                if np.all(np.isfinite(ca_i)) and np.all(np.isfinite(ca_next)):
                    bg[_BG_CA_NEXT] = np.linalg.norm(ca_i - ca_next) / 10.0

            if 'C' in atoms_i and 'N' in atoms_next:
                c_i = atoms_i['C']
                n_next = atoms_next['N']
                if np.all(np.isfinite(c_i)) and np.all(np.isfinite(n_next)):
                    bg[_BG_PEP_FWD] = np.linalg.norm(n_next - c_i) / 10.0

        bond_geom_by_rid[rid] = bg

    # Build per-residue feature tensors
    W = 2 * CONTEXT_WINDOW + 1
    M = MAX_VALID_DISTANCES
    residues = []

    for center_idx, center_rid in enumerate(res_ids):
        res_info = aa_data[center_rid]
        res_name = res_info['residue_name']

        # Window of residue IDs
        window_rids = []
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            wrid = center_rid + offset
            if wrid in aa_data:
                window_rids.append(wrid)
            else:
                window_rids.append(-1)

        # Distance features (atom1_idx, atom2_idx, distance values)
        atom1_idx = torch.full((W, M), n_atom_types, dtype=torch.long)
        atom2_idx = torch.full((W, M), n_atom_types, dtype=torch.long)
        distances = torch.zeros(W, M, dtype=torch.float32)
        dist_mask = torch.zeros(W, M, dtype=torch.bool)

        for w, wrid in enumerate(window_rids):
            if wrid < 0 or wrid not in all_dist_features:
                continue
            feats = all_dist_features[wrid]
            n_valid = 0
            for key, val in feats.items():
                if not key.startswith('dist_') or n_valid >= M:
                    continue
                parts = key.split('_')
                if len(parts) != 3:
                    continue
                a1, a2 = parts[1], parts[2]
                if a1 in atom_to_idx and a2 in atom_to_idx:
                    atom1_idx[w, n_valid] = atom_to_idx[a1]
                    atom2_idx[w, n_valid] = atom_to_idx[a2]
                    distances[w, n_valid] = max(-5.0, min(val / 10.0, 10.0))
                    dist_mask[w, n_valid] = True
                    n_valid += 1

        # Residue/SS/mismatch indices for window
        residue_idx = torch.full((W,), N_RESIDUE_TYPES, dtype=torch.long)
        ss_idx = torch.full((W,), N_SS_TYPES, dtype=torch.long)
        mismatch_idx = torch.zeros(W, dtype=torch.long)  # 'match' = 0
        is_valid = torch.zeros(W, dtype=torch.float32)
        dssp_features = torch.zeros(W, len(DSSP_COLS), dtype=torch.float32)

        for w, wrid in enumerate(window_rids):
            if wrid < 0 or wrid not in aa_data:
                continue
            is_valid[w] = 1.0
            wname = aa_data[wrid]['residue_name']
            residue_idx[w] = RESIDUE_TO_IDX.get(wname, RESIDUE_TO_IDX.get('UNK', N_RESIDUE_TYPES))

            # DSSP lookup (try multiple chain keys)
            dssp_entry = None
            for ch_key in pdb_data:
                if ch_key[1] == wrid:
                    chain_letter = ch_key[0]
                    dssp_entry = dssp_data.get((chain_letter, wrid))
                    if dssp_entry:
                        break
            if dssp_entry is None:
                dssp_entry = dssp_data.get(('', wrid)) or dssp_data.get((' ', wrid))

            if dssp_entry:
                ss = dssp_entry.get('secondary_structure', 'UNK')
                ss_idx[w] = SS_TO_IDX.get(ss, SS_TO_IDX.get('UNK', N_SS_TYPES))
                for di, dcol in enumerate(DSSP_COLS):
                    val = dssp_entry.get(dcol, 0.0)
                    val = float(val) if val is not None else 0.0
                    if dssp_stats and dcol in dssp_stats:
                        mean = dssp_stats[dcol]['mean']
                        std = dssp_stats[dcol]['std']
                        val = (val - mean) / std if std > 1e-6 else val - mean
                        val = max(-10.0, min(val, 10.0))
                    dssp_features[w, di] = val

        # Spatial neighbors
        nb_info = neighbors.get(center_rid, {'ids': [-1]*K_SPATIAL_NEIGHBORS,
                                              'dists': [0.0]*K_SPATIAL_NEIGHBORS,
                                              'seps': [0]*K_SPATIAL_NEIGHBORS})
        neighbor_res_idx = torch.full((K_SPATIAL_NEIGHBORS,), N_RESIDUE_TYPES, dtype=torch.long)
        neighbor_ss_idx = torch.full((K_SPATIAL_NEIGHBORS,), N_SS_TYPES, dtype=torch.long)
        neighbor_dist = torch.tensor(nb_info['dists'], dtype=torch.float32)
        neighbor_seq_sep = torch.tensor(nb_info['seps'], dtype=torch.float32)
        neighbor_angles = torch.zeros(K_SPATIAL_NEIGHBORS, 4, dtype=torch.float32)
        neighbor_valid = torch.tensor([nid >= 0 for nid in nb_info['ids']], dtype=torch.bool)

        # Neighbor distance features
        neighbor_atom1_idx = torch.full((K_SPATIAL_NEIGHBORS, M), n_atom_types, dtype=torch.long)
        neighbor_atom2_idx = torch.full((K_SPATIAL_NEIGHBORS, M), n_atom_types, dtype=torch.long)
        neighbor_distances = torch.zeros(K_SPATIAL_NEIGHBORS, M, dtype=torch.float32)
        neighbor_dist_mask = torch.zeros(K_SPATIAL_NEIGHBORS, M, dtype=torch.bool)

        for k, nrid in enumerate(nb_info['ids']):
            if nrid < 0 or nrid not in aa_data:
                continue
            nname = aa_data[nrid]['residue_name']
            neighbor_res_idx[k] = RESIDUE_TO_IDX.get(nname, RESIDUE_TO_IDX.get('UNK', N_RESIDUE_TYPES))

            # Neighbor DSSP
            for ch_key in pdb_data:
                if ch_key[1] == nrid:
                    dssp_nb = dssp_data.get((ch_key[0], nrid))
                    if dssp_nb:
                        ss = dssp_nb.get('secondary_structure', 'UNK')
                        neighbor_ss_idx[k] = SS_TO_IDX.get(ss, SS_TO_IDX.get('UNK', N_SS_TYPES))
                        phi = dssp_nb.get('phi')
                        psi = dssp_nb.get('psi')
                        if phi is not None:
                            neighbor_angles[k, 0] = np.sin(np.radians(phi))
                            neighbor_angles[k, 1] = np.cos(np.radians(phi))
                        if psi is not None:
                            neighbor_angles[k, 2] = np.sin(np.radians(psi))
                            neighbor_angles[k, 3] = np.cos(np.radians(psi))
                        break

            # Neighbor distance features
            if nrid in all_dist_features:
                n_valid = 0
                for key, val in all_dist_features[nrid].items():
                    if not key.startswith('dist_') or n_valid >= M:
                        continue
                    parts = key.split('_')
                    if len(parts) != 3:
                        continue
                    a1, a2 = parts[1], parts[2]
                    if a1 in atom_to_idx and a2 in atom_to_idx:
                        neighbor_atom1_idx[k, n_valid] = atom_to_idx[a1]
                        neighbor_atom2_idx[k, n_valid] = atom_to_idx[a2]
                        neighbor_distances[k, n_valid] = max(-5.0, min(val / 10.0, 10.0))
                        neighbor_dist_mask[k, n_valid] = True
                        n_valid += 1

        # Query residue code
        query_residue_code = torch.tensor(
            RESIDUE_TO_IDX.get(res_name, RESIDUE_TO_IDX.get('UNK', N_RESIDUE_TYPES)),
            dtype=torch.long)

        # Compact structural features (49-dim)
        query_struct = torch.zeros(N_STRUCT_FEATURES, dtype=torch.float32)
        feats = all_dist_features.get(center_rid, {})
        for ci, col in enumerate(STRUCT_DIST_COLS):
            if col in feats:
                query_struct[ci] = feats[col]
        off = len(STRUCT_DIST_COLS)
        for ci, col in enumerate(STRUCT_SC_COLS):
            if col in feats:
                query_struct[off + ci] = feats[col]

        # Bond geometry per window position
        bond_geom = torch.zeros(W, N_BOND_GEOM, dtype=torch.float32)
        for w, wrid in enumerate(window_rids):
            if wrid >= 0 and wrid in bond_geom_by_rid:
                bond_geom[w] = torch.from_numpy(bond_geom_by_rid[wrid])

        # Cross-residue distance pairs (Phase 1) — shared with cache builder
        # via distance_features.build_cross_arrays_for_residue. The parity
        # test verifies inference and cache produce identical bytes.
        cross_a1_np, cross_a2_np, cross_off_np, cross_v_np, n_cross = \
            build_cross_arrays_for_residue(
                center_rid=center_rid,
                aa_data=aa_data,
                res_ids_in_order=res_ids,
                spatial_neighbor_ids=nb_info['ids'],
                atom_to_idx=atom_to_idx,
                context_window=CONTEXT_WINDOW,
                max_cross_distances=MAX_CROSS_DISTANCES,
                n_cross_offset_types=N_CROSS_OFFSET_TYPES,
                heavy_cutoff=CROSS_DIST_CUTOFF,
                h_cutoff=CROSS_H_CUTOFF,
            )
        cross_atom1_idx = torch.from_numpy(cross_a1_np.astype(np.int64))
        cross_atom2_idx = torch.from_numpy(cross_a2_np.astype(np.int64))
        cross_offset_idx = torch.from_numpy(cross_off_np.astype(np.int64))
        cross_distances = torch.from_numpy(cross_v_np.astype(np.float32))
        cross_dist_mask = torch.zeros(MAX_CROSS_DISTANCES, dtype=torch.bool)
        if n_cross > 0:
            cross_dist_mask[:n_cross] = True

        residues.append({
            'residue_id': center_rid,
            'residue_name': res_name,
            'atom1_idx': atom1_idx,
            'atom2_idx': atom2_idx,
            'distances': distances,
            'dist_mask': dist_mask,
            'residue_idx': residue_idx,
            'ss_idx': ss_idx,
            'mismatch_idx': mismatch_idx,
            'is_valid': is_valid,
            'dssp_features': dssp_features,
            'neighbor_res_idx': neighbor_res_idx,
            'neighbor_ss_idx': neighbor_ss_idx,
            'neighbor_dist': neighbor_dist,
            'neighbor_seq_sep': neighbor_seq_sep,
            'neighbor_angles': neighbor_angles,
            'neighbor_valid': neighbor_valid,
            'neighbor_atom1_idx': neighbor_atom1_idx,
            'neighbor_atom2_idx': neighbor_atom2_idx,
            'neighbor_distances': neighbor_distances,
            'neighbor_dist_mask': neighbor_dist_mask,
            'cross_atom1_idx': cross_atom1_idx,
            'cross_atom2_idx': cross_atom2_idx,
            'cross_offset_idx': cross_offset_idx,
            'cross_distances': cross_distances,
            'cross_dist_mask': cross_dist_mask,
            'query_residue_code': query_residue_code,
            'query_struct': query_struct,
            'bond_geom': bond_geom,
        })

    return residues, res_ids


# ============================================================================
# Batch and Run
# ============================================================================

def predict(model, residues, device, stats, shift_cols, batch_size=64):
    """Run model inference on extracted features.

    Denormalizes using per-AA stats when available (matching training normalization),
    falling back to global stats otherwise. Denorm itself is delegated to the ONE
    canonical shift_norm.denormalize (shared with training_utils / 07_evaluate).
    """
    model.eval()
    all_preds = []

    for i in range(0, len(residues), batch_size):
        batch_residues = residues[i:i + batch_size]

        # Collate into batch tensors
        batch = {}
        keys = [k for k in batch_residues[0].keys()
                if k not in ('residue_id', 'residue_name') and isinstance(batch_residues[0][k], torch.Tensor)]
        for k in keys:
            batch[k] = torch.stack([r[k] for r in batch_residues])

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.no_grad():
            ctx = autocast('cuda') if device == 'cuda' else nullcontext()
            with ctx:
                pred = model(**batch)  # (B, n_shifts)

        # Denormalize (per-AA, global fallback) via the canonical implementation.
        pred_np = pred.cpu().numpy()
        codes = np.array([int(res['query_residue_code']) for res in batch_residues])
        pred_np = denormalize(pred_np, codes, stats, shift_cols).astype(pred_np.dtype)

        all_preds.append(pred_np)

    return np.concatenate(all_preds, axis=0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Predict chemical shifts from a PDB structure')
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--pdb', required=True,
                        help='PDB file path or 4-letter PDB ID (e.g. 1UBQ)')
    parser.add_argument('--chain', default=None, help='Chain ID (default: all)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output CSV path (default: stdout)')
    parser.add_argument('--device', default=None, help='Device (auto-detected)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone_only', action='store_true',
                        help='Only output backbone shifts (CA, CB, C, N, H, HA)')
    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("  Chemical Shift Prediction from PDB")
    print("=" * 60)
    print(f"  Model:  {args.model}")
    print(f"  Input:  {args.pdb}")
    print(f"  Chain:  {args.chain or 'all'}")
    print(f"  Device: {device}")

    # Load model
    print(f"\nLoading model...")
    model, info = load_model(args.model, device)
    stats = info['stats']
    shift_cols = info['shift_cols']
    atom_to_idx = info['atom_to_idx']
    print(f"  Epoch {info['epoch']}, {len(shift_cols)} shift types, {info['n_atom_types']} atom types")

    # Resolve PDB path: download if given a PDB ID
    pdb_path = args.pdb
    _tmpfile = None
    if not os.path.exists(pdb_path) and len(pdb_path) == 4 and pdb_path.isalnum():
        pdb_id = pdb_path.upper()
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        print(f"\n  Downloading {pdb_id} from RCSB...")
        try:
            _tmpfile = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
            urllib.request.urlretrieve(url, _tmpfile.name)
            pdb_path = _tmpfile.name
            print(f"  Downloaded to {pdb_path}")
        except Exception as e:
            print(f"  ERROR: Failed to download {pdb_id}: {e}")
            sys.exit(1)

    # Load DSSP normalization stats — prefer the checkpoint's stats['dssp']
    # (matches training distribution) and fall back to the global dssp_stats.json
    dssp_stats = stats.get('dssp') if isinstance(stats, dict) else None
    if dssp_stats:
        print(f"  DSSP stats from checkpoint ({len(dssp_stats)} features)")
    else:
        dssp_stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dssp_stats.json')
        if os.path.exists(dssp_stats_path):
            with open(dssp_stats_path) as f:
                dssp_stats = json.load(f)
            print(f"  DSSP stats from {dssp_stats_path} ({len(dssp_stats)} features)")
        else:
            print(f"  WARNING: No DSSP stats found — DSSP features will not be normalized")

    # Extract features from PDB
    print(f"\nExtracting features...")
    residues, res_ids = extract_features_from_pdb(pdb_path, chain_id=args.chain,
                                                   atom_to_idx=atom_to_idx,
                                                   dssp_stats=dssp_stats)
    if not residues:
        print("ERROR: No features extracted")
        sys.exit(1)

    # Run inference
    print(f"\nRunning inference on {len(residues)} residues...")
    predictions = predict(model, residues, device, stats, shift_cols, args.batch_size)

    # Build output
    import pandas as pd
    rows = []
    backbone_shifts = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}

    for i, res in enumerate(residues):
        row = {
            'residue_id': res['residue_id'],
            'residue_name': res['residue_name'],
        }
        for si, col in enumerate(shift_cols):
            if args.backbone_only and col not in backbone_shifts:
                continue
            name = col.replace('_shift', '').upper()
            row[name] = round(float(predictions[i, si]), 3)
        rows.append(row)

    df = pd.DataFrame(rows)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n  Predictions saved to {args.output}")
    else:
        print(f"\n{'='*60}")
        print(df.to_string(index=False))

    # Print backbone summary
    print(f"\n{'='*60}")
    print(f"  Predicted {len(residues)} residues, {len(shift_cols)} shift types")
    bb_cols = [c for c in shift_cols if c in backbone_shifts]
    for col in sorted(bb_cols):
        si = shift_cols.index(col)
        vals = predictions[:, si]
        name = col.replace('_shift', '').upper()
        print(f"  {name:3s}: mean={np.mean(vals):.2f}  std={np.std(vals):.2f}  "
              f"range=[{np.min(vals):.1f}, {np.max(vals):.1f}]")
    print("=" * 60)

    # Clean up downloaded PDB
    if _tmpfile is not None:
        os.unlink(_tmpfile.name)


if __name__ == '__main__':
    main()
