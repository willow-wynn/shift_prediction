#!/usr/bin/env python3
"""
Single-PDB inference: predict chemical shifts from a PDB structure.

Usage:
    python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb
    python inference.py --model data/checkpoints/best_fold1.pt --pdb 1UBQ --chain A -o shifts.csv
    python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb --no_retrieval
    python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb --index_dir data/retrieval_indices
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
    DSSP_COLS, K_SPATIAL_NEIGHBORS, K_RETRIEVED,
    CONTEXT_WINDOW, MAX_VALID_DISTANCES,
    STRUCT_DIST_COLS, STRUCT_SC_COLS, N_STRUCT_FEATURES,
    AA_3_TO_1, ESM_MODEL_NAME, ESM_EMBED_DIM, ESM_REPR_LAYER,
    FAISS_NPROBE,
    N_BOND_GEOM,
    MAX_CROSS_DISTANCES, CROSS_DIST_CUTOFF, CROSS_H_CUTOFF,
    N_CROSS_OFFSET_TYPES,
)
from model import ShiftPredictorWithRetrieval
from pdb_utils import parse_pdb, run_dssp
from distance_features import compute_all_distance_features, calc_cross_residue_distances
from spatial_neighbors import find_neighbors


# ============================================================================
# Model Loading (auto-detect architecture from checkpoint)
# ============================================================================

def load_model(checkpoint_path, device):
    """Load model from checkpoint with auto-detected architecture."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Infer architecture from weights
    n_atom_types = clean_sd['distance_attention.atom_embed.weight'].shape[0] - 1
    n_dssp = clean_sd['dssp_proj.weight'].shape[1] if 'dssp_proj.weight' in clean_sd else 0

    # n_shifts from struct_head final layer
    struct_keys = sorted([k for k in clean_sd if k.startswith('struct_head.') and k.endswith('.weight')])
    n_shifts = clean_sd[struct_keys[-1]].shape[0]

    cnn_channels = []
    for i in range(0, 10, 2):
        key = f'cnn.{i}.conv1.weight'
        if key in clean_sd:
            cnn_channels.append(clean_sd[key].shape[0])

    spatial_hidden = clean_sd.get('spatial_attention.fallback_embed', torch.zeros(192)).shape[0]
    retrieval_hidden = clean_sd.get('cross_attn_layers.0.shift_embed.weight', torch.zeros(1, 320)).shape[-1]

    # Detect whether the checkpoint includes retrieval-module weights. If not,
    # build a struct-only model — otherwise the retrieval pathway is initialized
    # with RANDOM weights and forward() blends random predictions into the output.
    retrieval_prefixes = ('neighbor_encoder.', 'self_attn_layers.',
                          'cross_attn_layers.', 'cross_ffn_layers.',
                          'direct_transfer.', 'retrieval_gate.')
    has_retrieval = any(k.startswith(retrieval_prefixes) for k in clean_sd)
    # Also trust the mode field if present
    ckpt_mode = checkpoint.get('mode')
    if ckpt_mode == 'struct':
        has_retrieval = False

    model = ShiftPredictorWithRetrieval(
        n_atom_types=n_atom_types,
        n_residue_types=N_RESIDUE_TYPES,
        n_ss_types=N_SS_TYPES,
        n_mismatch_types=N_MISMATCH_TYPES,
        n_dssp=n_dssp,
        n_shifts=n_shifts,
        cnn_channels=cnn_channels,
        spatial_hidden=spatial_hidden,
        retrieval_hidden=retrieval_hidden,
        use_retrieval=has_retrieval,
    ).to(device)

    filtered = {k: v for k, v in clean_sd.items() if not k.startswith('physics_encoder.')}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # Warn loudly if anything unexpected happens
    if missing:
        print(f"  WARNING: {len(missing)} missing keys on load (first: {missing[:3]})")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys in checkpoint (first: {unexpected[:3]})")
    if not has_retrieval:
        print(f"  Loaded as structure-only (no retrieval module)")
    model.eval()

    return model, {
        'stats': checkpoint.get('stats', {}),
        'shift_cols': checkpoint.get('shift_cols', []),
        'atom_to_idx': checkpoint.get('atom_to_idx', ATOM_TO_IDX),
        'k_retrieved': checkpoint.get('k_retrieved', K_RETRIEVED),
        'n_atom_types': n_atom_types,
        'epoch': checkpoint.get('epoch', '?'),
    }


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

    # Parse PDB
    print(f"  Parsing PDB: {pdb_path}")
    pdb_data = parse_pdb(pdb_path, chain_id=chain_id)
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

    # Pre-compute inter-residue bond geometry
    print(f"  Computing bond geometry...")
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
                    bg[0] = np.linalg.norm(ca_i - ca_prev) / 10.0  # bond_ca_prev

            if 'C' in atoms_prev and 'N' in atoms_i:
                c_prev = atoms_prev['C']
                n_i = atoms_i['N']
                if np.all(np.isfinite(c_prev)) and np.all(np.isfinite(n_i)):
                    bg[3] = np.linalg.norm(n_i - c_prev) / 10.0  # bond_peptide_bkwd

        # Next residue
        if i < len(res_ids) - 1:
            next_rid = res_ids[i + 1]
            atoms_next = aa_data[next_rid]['atoms']
            if 'CA' in atoms_i and 'CA' in atoms_next:
                ca_i = atoms_i['CA']
                ca_next = atoms_next['CA']
                if np.all(np.isfinite(ca_i)) and np.all(np.isfinite(ca_next)):
                    bg[1] = np.linalg.norm(ca_i - ca_next) / 10.0  # bond_ca_next

            if 'C' in atoms_i and 'N' in atoms_next:
                c_i = atoms_i['C']
                n_next = atoms_next['N']
                if np.all(np.isfinite(c_i)) and np.all(np.isfinite(n_next)):
                    bg[2] = np.linalg.norm(n_next - c_i) / 10.0  # bond_peptide_fwd

        bond_geom_by_rid[rid] = bg

    # Build per-residue feature tensors
    W = 2 * CONTEXT_WINDOW + 1
    M = MAX_VALID_DISTANCES
    K = K_RETRIEVED
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

        # Cross-residue distance pairs (Phase 1).
        # MUST mirror the cache builder exactly (gated by parity test):
        #   - heavy×heavy with heavy_cutoff, H×heavy with h_cutoff
        #   - offset code: 0=intra (unused here), 1..2*CW+1 = window
        #     (-CW..+CW shifted to 1..2*CW+1, slot CW+1 = self is unused),
        #     2*CW+2..2*CW+1+K_SP = spatial slots
        #   - skip spatial neighbors that are also in the ±CW window (dedup)
        #   - sort by distance ascending, take top MAX_CROSS_DISTANCES
        #   - apply same /10 clip as intra
        M_CR = MAX_CROSS_DISTANCES
        cross_atom1_idx = torch.full((M_CR,), n_atom_types, dtype=torch.long)
        cross_atom2_idx = torch.full((M_CR,), n_atom_types, dtype=torch.long)
        cross_offset_idx = torch.full((M_CR,), N_CROSS_OFFSET_TYPES, dtype=torch.long)
        cross_distances = torch.zeros(M_CR, dtype=torch.float32)
        cross_dist_mask = torch.zeros(M_CR, dtype=torch.bool)

        cross_pairs_buf = []
        atoms_center = aa_data[center_rid]['atoms']
        window_partners = set()

        for off in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            if off == 0:
                continue
            nrid = center_rid + off
            if nrid not in aa_data:
                continue
            window_partners.add(nrid)
            other_atoms = aa_data[nrid]['atoms']
            code = off + CONTEXT_WINDOW + 1   # -CW..+CW -> 1..2*CW+1 (CW+1=self unused)
            for (a1_name, a2_name, d) in calc_cross_residue_distances(
                atoms_center, other_atoms,
                heavy_cutoff=CROSS_DIST_CUTOFF, h_cutoff=CROSS_H_CUTOFF,
            ):
                if a1_name in atom_to_idx and a2_name in atom_to_idx:
                    cross_pairs_buf.append((
                        atom_to_idx[a1_name], atom_to_idx[a2_name], code, d))

        for k, nrid in enumerate(nb_info['ids']):
            if nrid < 0 or nrid not in aa_data or nrid in window_partners:
                continue
            other_atoms = aa_data[nrid]['atoms']
            code = 2 * CONTEXT_WINDOW + 2 + k   # spatial slot k -> 2*CW+2 + k
            for (a1_name, a2_name, d) in calc_cross_residue_distances(
                atoms_center, other_atoms,
                heavy_cutoff=CROSS_DIST_CUTOFF, h_cutoff=CROSS_H_CUTOFF,
            ):
                if a1_name in atom_to_idx and a2_name in atom_to_idx:
                    cross_pairs_buf.append((
                        atom_to_idx[a1_name], atom_to_idx[a2_name], code, d))

        # Distance-ascending priority pruning to M_CR
        cross_pairs_buf.sort(key=lambda t: t[3])
        cross_pairs_buf = cross_pairs_buf[:M_CR]
        for i, (a1_i, a2_i, code, d) in enumerate(cross_pairs_buf):
            cross_atom1_idx[i] = a1_i
            cross_atom2_idx[i] = a2_i
            cross_offset_idx[i] = code
            cross_distances[i] = max(-5.0, min(d / 10.0, 10.0))
            cross_dist_mask[i] = True

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
# Retrieval
# ============================================================================

def _fill_empty_retrieval(residues, n_shifts, k_retrieved):
    """Fill retrieval tensors with zeros (structure-only fallback)."""
    K = k_retrieved
    for res in residues:
        res['retrieved_shifts'] = torch.zeros(K, n_shifts, dtype=torch.float32)
        res['retrieved_shift_masks'] = torch.zeros(K, n_shifts, dtype=torch.bool)
        res['retrieved_residue_codes'] = torch.zeros(K, dtype=torch.long)
        res['retrieved_distances'] = torch.zeros(K, dtype=torch.float32)
        res['retrieved_valid'] = torch.zeros(K, dtype=torch.bool)
        res['neighbor_struct'] = torch.zeros(K, N_STRUCT_FEATURES, dtype=torch.float32)


def extract_base_encodings(model, residues, device, n_shifts, k_retrieved, batch_size=64):
    """Run a structure-only forward pass and capture base_encoding via hook."""
    _fill_empty_retrieval(residues, n_shifts, k_retrieved)

    captured = {}
    def hook_fn(module, inp, out):
        captured['base_encoding'] = inp[0].detach().cpu()
    handle = model.struct_head[0].register_forward_hook(hook_fn)

    model.eval()
    all_encodings = []
    for i in range(0, len(residues), batch_size):
        batch_residues = residues[i:i + batch_size]
        batch = {}
        keys = [k for k in batch_residues[0].keys()
                if k not in ('residue_id', 'residue_name') and isinstance(batch_residues[0][k], torch.Tensor)]
        for k in keys:
            batch[k] = torch.stack([r[k] for r in batch_residues])
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.no_grad():
            ctx = autocast('cuda') if device == 'cuda' else nullcontext()
            with ctx:
                model(**batch)
        all_encodings.append(captured['base_encoding'].numpy())

    handle.remove()
    return np.concatenate(all_encodings, axis=0)


def get_retrieval_features(model, residues, device, n_shifts, k_retrieved, stats,
                           shift_cols, index_dir, batch_size=64, bmrb_id=None):
    """Compute embeddings, query FAISS, and populate retrieval tensors.

    Falls back to structure-only if index_dir is not available.
    """
    if index_dir is None or not os.path.exists(index_dir):
        print("  No retrieval index found — using structure-only mode")
        _fill_empty_retrieval(residues, n_shifts, k_retrieved)
        return residues

    # Step 1: Extract base_encoding embeddings via structure-only forward pass
    print("  Extracting structure embeddings for retrieval...")
    embeddings = extract_base_encodings(model, residues, device, n_shifts,
                                        k_retrieved, batch_size)

    # Step 2: Load retriever and query
    from retrieval import Retriever
    import faiss

    # Use fold 1 index (at inference on new proteins, fold doesn't matter)
    fold_indices = [f for f in os.listdir(index_dir) if f.startswith('index_exclude_fold_')]
    if not fold_indices:
        print("  No FAISS index found — using structure-only mode")
        _fill_empty_retrieval(residues, n_shifts, k_retrieved)
        return residues

    # Pick any fold's index
    fold_num = int(fold_indices[0].split('_')[-1].replace('.faiss', ''))
    print(f"  Loading retrieval index (fold {fold_num} excluded)...")
    retriever = Retriever(index_dir, exclude_fold=fold_num, k=k_retrieved)

    # Use actual BMRB ID for same-protein exclusion, or dummy for new proteins
    exclude_id = bmrb_id if bmrb_id else '__inference__'
    query_bmrb_ids = [exclude_id] * len(residues)
    embeddings = embeddings.astype(np.float32)

    print(f"  Retrieving {k_retrieved} neighbors for {len(residues)} residues...")
    results = retriever.retrieve(embeddings, query_bmrb_ids, k=k_retrieved)

    # Step 3: Build normalization tensors from stats
    means = []
    stds = []
    for col in shift_cols:
        if col in stats:
            means.append(stats[col]['mean'])
            stds.append(max(stats[col]['std'], 1e-6))
        else:
            means.append(0.0)
            stds.append(1.0)
    means_t = torch.tensor(means, dtype=torch.float32)
    stds_t = torch.tensor(stds, dtype=torch.float32)

    # Step 4: Populate retrieval tensors
    K = k_retrieved
    for i, res in enumerate(residues):
        shifts = torch.from_numpy(results['shifts'][i].astype(np.float32))
        shift_masks = torch.from_numpy(results['shift_masks'][i])

        # Normalize retrieved shifts to match training
        shifts = (shifts - means_t.unsqueeze(0)) / stds_t.unsqueeze(0)
        shifts = torch.clamp(shifts, -10, 10)
        shifts = torch.where(torch.isnan(shifts), torch.zeros_like(shifts), shifts)

        res['retrieved_shifts'] = shifts
        res['retrieved_shift_masks'] = shift_masks
        res['retrieved_residue_codes'] = torch.from_numpy(
            results['residue_codes'][i].astype(np.int64))
        res['retrieved_distances'] = torch.from_numpy(
            results['distances'][i].astype(np.float32))
        res['retrieved_valid'] = torch.from_numpy(
            (results['indices'][i] >= 0).astype(bool))
        res['neighbor_struct'] = torch.zeros(K, N_STRUCT_FEATURES, dtype=torch.float32)

    n_valid = sum(int(res['retrieved_valid'].sum()) for res in residues)
    avg_valid = n_valid / max(len(residues), 1)
    print(f"  Retrieved avg {avg_valid:.1f}/{K} valid neighbors per residue")

    return residues


# ============================================================================
# Batch and Run
# ============================================================================

def predict(model, residues, device, stats, shift_cols, batch_size=64):
    """Run model inference on extracted features.

    Denormalizes using per-AA stats when available (matching training normalization),
    falling back to global stats otherwise.
    """
    per_aa_stats = stats.get('per_aa', {})

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

        # Denormalize using per-AA stats when available
        pred_np = pred.cpu().numpy()
        for j, res in enumerate(batch_residues):
            aa_idx = int(res['query_residue_code'])
            aa_name = STANDARD_RESIDUES[aa_idx] if aa_idx < len(STANDARD_RESIDUES) else None
            for si, col in enumerate(shift_cols):
                aa_s = per_aa_stats.get(aa_name, {}).get(col) if aa_name else None
                if aa_s:
                    pred_np[j, si] = pred_np[j, si] * aa_s['std'] + aa_s['mean']
                elif col in stats:
                    pred_np[j, si] = pred_np[j, si] * stats[col]['std'] + stats[col]['mean']

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
    parser.add_argument('--index_dir', default=None,
                        help='Retrieval index directory (auto-detected from checkpoint if not set)')
    parser.add_argument('--no_retrieval', action='store_true',
                        help='Disable retrieval (structure-only prediction)')
    parser.add_argument('--backbone_only', action='store_true',
                        help='Only output backbone shifts (CA, CB, C, N, H, HA)')
    parser.add_argument('--bmrb', default=None,
                        help='BMRB ID for same-protein exclusion during retrieval')
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
    k_retrieved = info['k_retrieved']
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

    # Add retrieval features
    index_dir = args.index_dir
    if not args.no_retrieval and index_dir is None:
        # Auto-detect: check common locations (newest first)
        for candidate in [
            'data/struct_retrieval_v2/retrieval_indices',
            'data/struct_retrieval/retrieval_indices',
            'data/retrieval_indices',
        ]:
            if os.path.exists(candidate):
                index_dir = candidate
                break

    if args.no_retrieval:
        index_dir = None

    residues = get_retrieval_features(
        model, residues, device, len(shift_cols), k_retrieved,
        stats, shift_cols, index_dir, args.batch_size, bmrb_id=args.bmrb)

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
