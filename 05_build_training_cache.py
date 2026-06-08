#!/usr/bin/env python
"""
05 - Training Cache Builder (structure-only)

Builds a memory-mapped cache so that training reads compact numpy arrays from
disk rather than reparsing CSVs/PDBs every epoch.

For each fold, this script:
1. Loads the compiled CSV (filtering to proteins in that fold)
2. Builds compact structural arrays (residue types, distances, DSSP, spatial
   neighbours, bond geometry, and cross-residue distance arrays via 04c)
3. Saves everything as memory-mapped numpy arrays

Cache layout (per fold):
    <output_dir>/fold_{k}/
        config.json
        samples.npy
        structural/  (residue_idx.npy, ss_idx.npy, dssp.npy, cross_*.npy, ...)

Usage:
    python 05_build_training_cache.py --data_dir ./data --output_dir ./data/cache
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import (
    STANDARD_RESIDUES, RESIDUE_TO_IDX, N_RESIDUE_TYPES,
    SS_TYPES, SS_TO_IDX, N_SS_TYPES,
    MISMATCH_TYPES, MISMATCH_TO_IDX, N_MISMATCH_TYPES,
    DSSP_COLS,
    CONTEXT_WINDOW,
    K_SPATIAL_NEIGHBORS,
    MAX_VALID_DISTANCES,
    STRUCT_DIST_COLS, STRUCT_SC_COLS, N_STRUCT_FEATURES,
    BOND_GEOM_COLS, N_BOND_GEOM,
)
from dataset import (
    parse_distance_columns,
    build_atom_vocabulary,
    parse_shift_columns,
    get_dssp_columns,
)

# ----- Cross-residue distance features (Phase 1, Option A) -----
# Build cross arrays inline so a fresh 01->05 yields a COMPLETE cache (intra +
# cross + bond + dssp) with no out-of-band 04c patch. We reuse the EXACT
# PDB-resolution + NMR-model-selection helpers that 04c_compute_cross_arrays.py
# uses, so an integrated build produces byte-identical cross arrays to the
# (validated) 04c patch. Loaded via importlib because the module name starts
# with a digit. See claude/cross_features_full_rebuild_plan.md.
from config import (
    ATOM_TO_IDX, MAX_CROSS_DISTANCES, N_CROSS_OFFSET_TYPES,
    CROSS_DIST_CUTOFF, CROSS_H_CUTOFF,
)
from distance_features import build_cross_arrays_for_residue
import importlib.util as _ilu
_cx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '04c_compute_cross_arrays.py')
_cx_spec = _ilu.spec_from_file_location('cross04c', _cx_path)
cross04c = _ilu.module_from_spec(_cx_spec)
_cx_spec.loader.exec_module(cross04c)


def extract_struct_features(pdf, start_idx, n, flat_struct):
    """Extract 49-dim structural feature vector for each residue.

    Layout: [0:21] backbone distances, [21:26] SC geometry,
            [26:30] sin/cos phi/psi, [30:39] DSSP, [39:49] SS one-hot
    """
    # [0:21] Backbone pairwise distances
    for ci, col in enumerate(STRUCT_DIST_COLS):
        if col in pdf.columns:
            vals = pdf[col].values
            valid = ~np.isnan(vals)
            flat_struct[start_idx:start_idx + n, ci] = np.where(valid, vals, 0.0)

    # [21:26] Sidechain geometry
    off = len(STRUCT_DIST_COLS)
    for ci, col in enumerate(STRUCT_SC_COLS):
        if col in pdf.columns:
            vals = pdf[col].values
            valid = ~np.isnan(vals)
            flat_struct[start_idx:start_idx + n, off + ci] = np.where(valid, vals, 0.0)

    # [26:30] sin/cos phi/psi
    off2 = off + len(STRUCT_SC_COLS)
    for ai, angle_col in enumerate(['phi', 'psi']):
        if angle_col in pdf.columns:
            deg = pdf[angle_col].values.astype(np.float64)
            nan_mask = np.isnan(deg)
            rad = np.deg2rad(np.nan_to_num(deg, nan=0.0))
            sin_vals = np.sin(rad).astype(np.float32)
            cos_vals = np.cos(rad).astype(np.float32)
            sin_vals[nan_mask] = 0.0
            cos_vals[nan_mask] = 0.0
            flat_struct[start_idx:start_idx + n, off2 + ai * 2] = sin_vals
            flat_struct[start_idx:start_idx + n, off2 + ai * 2 + 1] = cos_vals

    # [30:39] DSSP numeric
    off3 = off2 + 4
    for di, col in enumerate(DSSP_COLS):
        if col in pdf.columns:
            vals = pdf[col].values
            valid = ~np.isnan(vals)
            flat_struct[start_idx:start_idx + n, off3 + di] = np.where(valid, vals, 0.0)

    # [39:49] Secondary structure one-hot
    off4 = off3 + len(DSSP_COLS)
    if 'secondary_structure' in pdf.columns:
        for i, ss in enumerate(pdf['secondary_structure'].fillna('UNK').values):
            idx = SS_TO_IDX.get(str(ss).strip(), SS_TO_IDX['UNK'])
            flat_struct[start_idx + i, off4 + idx] = 1.0


# ============================================================================
# Cache Builder
# ============================================================================

def compute_normalization_stats(df, shift_cols, dssp_cols):
    """Compute mean/std for shift and DSSP columns from the training data.

    Computes both global and per-(amino_acid, shift) statistics.
    Per-AA stats enable better normalization for shifts whose distribution
    varies dramatically across amino acid types (e.g., CD1: LEU=24, TYR=132).
    """
    stats = {}
    for col in shift_cols:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                stats[col] = {'mean': float(vals.mean()), 'std': float(vals.std())}
            else:
                stats[col] = {'mean': 0.0, 'std': 1.0}

    dssp_stats = {}
    for col in dssp_cols:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                dssp_stats[col] = {'mean': float(vals.mean()), 'std': float(vals.std())}
            else:
                dssp_stats[col] = {'mean': 0.0, 'std': 1.0}
    stats['dssp'] = dssp_stats

    # Per-(amino_acid, shift) stats
    if 'residue_code' in df.columns:
        per_aa = {}
        for aa in df['residue_code'].dropna().unique():
            aa_str = str(aa).strip()
            if not aa_str:
                continue
            aa_df = df[df['residue_code'] == aa]
            aa_stats = {}
            for col in shift_cols:
                if col in aa_df.columns:
                    vals = aa_df[col].dropna()
                    if len(vals) >= 10:
                        aa_stats[col] = {'mean': float(vals.mean()), 'std': max(float(vals.std()), 0.1)}
            if aa_stats:
                per_aa[aa_str] = aa_stats
        stats['per_aa'] = per_aa

    return stats


def build_cache_for_fold(
    df: pd.DataFrame,
    fold: int,
    shift_cols: list,
    dist_col_info: list,
    dssp_cols: list,
    atom_to_idx: dict,
    stats: dict,
    cache_dir: str,
    context_window: int = CONTEXT_WINDOW,
    k_spatial: int = K_SPATIAL_NEIGHBORS,
    max_valid_distances: int = MAX_VALID_DISTANCES,
    struct_lookup: dict = None,
    build_cross: bool = False,
    build_log: dict = None,
    pdb_search_dirs: list = None,
):
    """Build a complete training cache for one fold.

    This is essentially CachedRetrievalDataset.create() pulled out as a
    standalone function with full provenance logging and checkpoint/resume.
    """
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True)
    (cache_path / 'structural').mkdir()

    n_shifts = len(shift_cols)
    n_atom_types = len(atom_to_idx)
    n_dssp = len(dssp_cols)
    window_size = 2 * context_window + 1

    proteins = list(df.groupby('bmrb_id'))
    total_residues = sum(len(pdf) for _, pdf in proteins)
    n_proteins = len(proteins)

    M = max_valid_distances

    # Pre-compute atom indices
    col_to_atoms = []
    for col, atom1, atom2 in dist_col_info:
        a1_idx = atom_to_idx.get(atom1, n_atom_types)
        a2_idx = atom_to_idx.get(atom2, n_atom_types)
        col_to_atoms.append((a1_idx, a2_idx))
    col_to_atoms = np.array(col_to_atoms, dtype=np.int64)

    print(f"    Building cache for fold {fold}...")
    print(f"      Proteins: {n_proteins}")
    print(f"      Total residues: {total_residues:,}")
    print(f"      Shift columns: {n_shifts}")
    print(f"      Distance columns: {len(dist_col_info)}")
    print(f"      DSSP columns: {n_dssp}")
    print(f"      Context window: {window_size}")
    print(f"      K spatial: {k_spatial}")
    print(f"      Max valid distances: {M}")

    # ========== Allocate structural arrays ==========
    flat_residue_idx = np.zeros(total_residues, dtype=np.int32)
    flat_ss_idx = np.zeros(total_residues, dtype=np.int32)
    flat_mismatch_idx = np.zeros(total_residues, dtype=np.int32)

    flat_dist_atom1 = np.full((total_residues, M), n_atom_types, dtype=np.int16)
    flat_dist_atom2 = np.full((total_residues, M), n_atom_types, dtype=np.int16)
    flat_dist_values = np.zeros((total_residues, M), dtype=np.float16)
    flat_dist_count = np.zeros(total_residues, dtype=np.int16)

    flat_dssp = np.zeros((total_residues, n_dssp), dtype=np.float16)
    flat_shifts = np.zeros((total_residues, n_shifts), dtype=np.float32)
    flat_shift_mask = np.zeros((total_residues, n_shifts), dtype=bool)
    flat_angles = np.zeros((total_residues, 4), dtype=np.float16)

    flat_spatial_ids = np.full((total_residues, k_spatial), -1, dtype=np.int32)
    flat_spatial_dist = np.zeros((total_residues, k_spatial), dtype=np.float16)
    flat_spatial_seq_sep = np.zeros((total_residues, k_spatial), dtype=np.int16)

    flat_window_idx = np.full((total_residues, window_size), -1, dtype=np.int32)

    # Inter-residue bond geometry (4 features per residue)
    flat_bond_geom = np.zeros((total_residues, N_BOND_GEOM), dtype=np.float32)

    # Protein tracking
    protein_offsets = []
    protein_lookup_offsets = []
    protein_min_res = []
    protein_max_res = []

    total_lookup_size = 0
    for _, pdf in proteins:
        rids = pdf['residue_id'].values
        total_lookup_size += int(rids.max()) - int(rids.min()) + 1

    flat_res_id_lookup = np.full(total_lookup_size, -1, dtype=np.int32)
    dist_cols = [info[0] for info in dist_col_info]

    current_offset = 0
    current_lookup_offset = 0
    samples_list = []

    idx_to_bmrb = {}
    global_to_resid = {}

    # Finding #7: track proteins with duplicate residue_ids. The span-sized
    # res_id_lookup is keyed by (residue_id - min_res); a duplicate residue_id
    # would otherwise collapse last-write-wins, so window / spatial-neighbor
    # resolution in dataset.py would silently point at only one copy (the wrong
    # neighbor AA/SS/distances), while BOTH duplicate rows still train. We make
    # the lookup deterministic FIRST-write-wins (rows are sorted by residue_id)
    # and warn with the affected bmrb list so consumers (dataset.py) and 05
    # agree on which copy a residue_id resolves to.
    dup_resid_bmrbs = []

    # ========== Process structural features ==========
    for prot_idx, (protein_id, pdf) in enumerate(
        tqdm(proteins, desc="      Processing structure")
    ):
        pdf = pdf.sort_values('residue_id').reset_index(drop=True)
        n = len(pdf)
        start_idx = current_offset

        residue_ids = pdf['residue_id'].values.astype(np.int32)
        min_res, max_res = int(residue_ids.min()), int(residue_ids.max())
        span = max_res - min_res + 1

        protein_offsets.append(current_offset)
        protein_lookup_offsets.append(current_lookup_offset)
        protein_min_res.append(min_res)
        protein_max_res.append(max_res)

        # Detect duplicate residue_ids within this protein (Finding #7).
        _uniq, _counts = np.unique(residue_ids, return_counts=True)
        if (_counts > 1).any():
            dup_resid_bmrbs.append(str(protein_id))

        for local_idx, rid in enumerate(residue_ids):
            global_idx = start_idx + local_idx
            _slot = current_lookup_offset + (rid - min_res)
            # FIRST-write-wins: only fill an empty slot so duplicate residue_ids
            # deterministically resolve to their FIRST (lowest-row) copy in both
            # 05 and dataset.py. (-1 == unfilled; the array is init'd to -1.)
            if flat_res_id_lookup[_slot] < 0:
                flat_res_id_lookup[_slot] = global_idx
            idx_to_bmrb[str(global_idx)] = str(protein_id)
            global_to_resid[str(global_idx)] = int(rid)

        # Residue types
        for i, code in enumerate(pdf['residue_code'].fillna('UNK').values):
            flat_residue_idx[start_idx + i] = RESIDUE_TO_IDX.get(
                str(code).upper(), RESIDUE_TO_IDX['UNK']
            )

        # Secondary structure
        if 'secondary_structure' in pdf.columns:
            for i, ss in enumerate(pdf['secondary_structure'].fillna('C').values):
                flat_ss_idx[start_idx + i] = SS_TO_IDX.get(str(ss), SS_TO_IDX['UNK'])

        # Mismatch type
        if 'mismatch_type' in pdf.columns:
            for i, mtype in enumerate(pdf['mismatch_type'].fillna('UNK').values):
                flat_mismatch_idx[start_idx + i] = MISMATCH_TO_IDX.get(
                    str(mtype), MISMATCH_TO_IDX['UNK']
                )

        # Sparse distances
        dist_matrix = pdf[dist_cols].values if dist_cols else np.empty((n, 0))

        for i in range(n):
            global_idx = start_idx + i
            if dist_matrix.shape[1] == 0:
                continue
            row = dist_matrix[i]

            valid_mask = ~np.isnan(row)
            valid_indices = np.where(valid_mask)[0]
            n_valid = min(len(valid_indices), M)

            flat_dist_count[global_idx] = n_valid

            if n_valid > 0:
                valid_indices = valid_indices[:n_valid]
                atom_pairs = col_to_atoms[valid_indices]
                flat_dist_atom1[global_idx, :n_valid] = atom_pairs[:, 0]
                flat_dist_atom2[global_idx, :n_valid] = atom_pairs[:, 1]

                valid_values = row[valid_indices] / 10.0
                valid_values = np.clip(valid_values, -5, 10)
                flat_dist_values[global_idx, :n_valid] = valid_values

        # DSSP features
        for di, col in enumerate(dssp_cols):
            if col in pdf.columns:
                vals = pdf[col].values
                valid = ~np.isnan(vals)
                if 'dssp' in stats and col in stats['dssp']:
                    mean = stats['dssp'][col]['mean']
                    std = stats['dssp'][col]['std']
                    if std > 1e-6:
                        normalized = np.where(valid, (vals - mean) / std, 0.0)
                    else:
                        normalized = np.where(valid, vals - mean, 0.0)
                else:
                    normalized = np.where(valid, vals, 0.0)
                normalized = np.clip(normalized, -10, 10)
                flat_dssp[start_idx:start_idx + n, di] = normalized

        # Shifts
        for si, col in enumerate(shift_cols):
            if col in pdf.columns:
                vals = pdf[col].values
                valid = ~np.isnan(vals)

                # Finding #13: only mark observations as supervised targets if we
                # also have normalization stats for this nucleus. If `shift_cols`
                # and `stats` disagree on a column (e.g. a frozen stats table
                # missing a nucleus), writing mask=True while leaving flat_shifts
                # at 0.0 would feed real observations to the loss as the (z=0)
                # mean. Gate the mask on `col in stats` so unstatted observations
                # are simply unsupervised rather than silently corrupted.
                if col in stats:
                    flat_shift_mask[start_idx:start_idx + n, si] = valid
                    mean, std = stats[col]['mean'], stats[col]['std']
                    if std > 1e-6:
                        normalized = np.where(valid, (vals - mean) / std, 0.0)
                    else:
                        normalized = np.where(valid, vals - mean, 0.0)
                    normalized = np.clip(normalized, -10, 10)
                    flat_shifts[start_idx:start_idx + n, si] = normalized
                # else: leave mask False (default) — no stats, so unsupervised.

        # Angles
        for i, angle_col in enumerate(['phi', 'psi']):
            if angle_col in pdf.columns:
                vals = pdf[angle_col].values
                valid = ~np.isnan(vals)
                rad = np.where(valid, np.radians(vals), 0.0)
                flat_angles[start_idx:start_idx + n, i * 2] = np.sin(rad)
                flat_angles[start_idx:start_idx + n, i * 2 + 1] = np.cos(rad)

        # Spatial neighbors
        for k in range(k_spatial):
            id_col = f'spatial_neighbor_{k}_id'
            dist_col_name = f'spatial_neighbor_{k}_dist'
            sep_col = f'spatial_neighbor_{k}_seq_sep'

            if id_col in pdf.columns:
                ids = pdf[id_col].values
                valid = ~np.isnan(ids) & (ids >= 0)
                flat_spatial_ids[start_idx:start_idx + n, k] = np.where(valid, ids, -1)

            if dist_col_name in pdf.columns:
                dists = pdf[dist_col_name].values
                valid = ~np.isnan(dists)
                flat_spatial_dist[start_idx:start_idx + n, k] = np.where(valid, dists, 0.0)

            if sep_col in pdf.columns:
                seps = pdf[sep_col].values
                valid = ~np.isnan(seps)
                flat_spatial_seq_sep[start_idx:start_idx + n, k] = np.where(valid, seps, 0)

        # Window indices
        for local_idx, rid in enumerate(residue_ids):
            global_idx = start_idx + local_idx
            for w, offset in enumerate(range(-context_window, context_window + 1)):
                neighbor_rid = rid + offset
                lookup_idx = neighbor_rid - min_res
                if 0 <= lookup_idx < span:
                    neighbor_global = flat_res_id_lookup[current_lookup_offset + lookup_idx]
                    if neighbor_global >= 0:
                        flat_window_idx[global_idx, w] = neighbor_global

        # Inter-residue bond geometry
        for bi, col in enumerate(BOND_GEOM_COLS):
            if col in pdf.columns:
                vals = pdf[col].values
                valid = ~np.isnan(vals)
                # Normalize: divide by 10 A (same scale as intra-residue distances)
                flat_bond_geom[start_idx:start_idx + n, bi] = np.where(valid, vals / 10.0, 0.0)

        # Build samples (only residues with at least one observed shift)
        for local_idx in range(n):
            global_idx = start_idx + local_idx
            if flat_shift_mask[global_idx].any():
                samples_list.append((global_idx, prot_idx))

        current_offset += n
        current_lookup_offset += span

    # Finding #7: warn about proteins with duplicate residue_ids. Their
    # res_id_lookup entries deterministically resolve to the first copy (above);
    # the remaining copies are still trained but cannot be reached as a window /
    # spatial neighbor. Surface them so they can be cleaned upstream (in 01).
    if dup_resid_bmrbs:
        print(f"      WARNING: {len(dup_resid_bmrbs)} protein(s) have duplicate "
              f"residue_ids; res_id_lookup resolves to the FIRST copy only "
              f"(both rows still train). Affected bmrbs: "
              f"{dup_resid_bmrbs[:20]}"
              f"{' ...' if len(dup_resid_bmrbs) > 20 else ''}")

    # ========== Cross-residue distances (Option A: PDB-aligned to intra) ==========
    # Second pass over the same proteins. For each residue we re-parse the
    # SAME PDB + chain + best-NMR-model that 01 recorded in build_log (so the
    # coordinates are identical to the intra/spatial/bond features above) and
    # compute the cross-pair arrays. flat_spatial_ids holds neighbour residue
    # IDs and is still alive here (freed only after the save block).
    flat_cross_atom1 = flat_cross_atom2 = flat_cross_offset = None
    flat_cross_values = flat_cross_count = None
    bmrb_pdb_used = {}
    if build_cross and build_log:
        print("      Computing cross-residue distance arrays (PDB-aligned)...")
        pad_atom = len(ATOM_TO_IDX)
        M_CR = MAX_CROSS_DISTANCES
        flat_cross_atom1 = np.full((total_residues, M_CR), pad_atom, dtype=np.int16)
        flat_cross_atom2 = np.full((total_residues, M_CR), pad_atom, dtype=np.int16)
        flat_cross_offset = np.full((total_residues, M_CR), N_CROSS_OFFSET_TYPES, dtype=np.int8)
        flat_cross_values = np.zeros((total_residues, M_CR), dtype=np.float16)
        flat_cross_count = np.zeros(total_residues, dtype=np.int16)
        # bond_geom column indices (recomputed from the same parsed coords)
        _BG_CA_PREV = BOND_GEOM_COLS.index('bond_ca_prev')
        _BG_CA_NEXT = BOND_GEOM_COLS.index('bond_ca_next')
        _BG_PEP_FWD = BOND_GEOM_COLS.index('bond_peptide_fwd')
        _BG_PEP_BKWD = BOND_GEOM_COLS.index('bond_peptide_bkwd')
        psd = pdb_search_dirs or ['data/pdbs',
                                  '/home/brooks/1TB/Wynn/data_archive/alphafold']
        n_ok = n_miss = n_filled = 0
        cross_fails = []
        for prot_idx, (protein_id, pdf) in enumerate(
                tqdm(proteins, desc="      Cross")):
            bmrb = str(protein_id)
            entry = build_log.get(bmrb)
            if entry is None:
                n_miss += 1; cross_fails.append(f'{bmrb} not-in-build-log'); continue
            pdb_path, chain_id = entry
            if not os.path.isfile(pdb_path):
                # build_log path is from the build machine; relocate by basename.
                base = os.path.basename(pdb_path); alt = None
                for d in psd:
                    cand = os.path.join(d, base)
                    if os.path.isfile(cand):
                        alt = cand; break
                if alt is None:
                    alt = cross04c.resolve_pdb_path(os.path.splitext(base)[0], psd)
                if alt is None:
                    n_miss += 1; cross_fails.append(f'{bmrb} pdb-not-found'); continue
                pdb_path = alt
            try:
                model = cross04c.select_best_nmr_model(pdb_path, chain_id)
                aa_data = cross04c.model_to_aa(model) if model is not None else {}
            except Exception as e:
                n_miss += 1; cross_fails.append(f'{bmrb} parse-error: {e}'); continue
            if not aa_data:
                n_miss += 1; cross_fails.append(f'{bmrb} no-aa-residues'); continue
            n_ok += 1
            bmrb_pdb_used[bmrb] = pdb_path
            res_ids_in_order = sorted(aa_data.keys())
            start = protein_offsets[prot_idx]
            for local in range(len(pdf)):
                g = start + local
                rid = global_to_resid.get(str(g))
                if rid is None or int(rid) not in aa_data:
                    continue
                sp_ids = flat_spatial_ids[g].tolist()
                a1, a2, off, vals, ncr = build_cross_arrays_for_residue(
                    center_rid=int(rid), aa_data=aa_data,
                    res_ids_in_order=res_ids_in_order,
                    spatial_neighbor_ids=sp_ids,
                    atom_to_idx=ATOM_TO_IDX,
                    context_window=context_window,
                    max_cross_distances=MAX_CROSS_DISTANCES,
                    n_cross_offset_types=N_CROSS_OFFSET_TYPES,
                    heavy_cutoff=CROSS_DIST_CUTOFF,
                    h_cutoff=CROSS_H_CUTOFF,
                )
                flat_cross_atom1[g] = a1; flat_cross_atom2[g] = a2
                flat_cross_offset[g] = off; flat_cross_values[g] = vals
                flat_cross_count[g] = ncr
                n_filled += 1

                # bond_geom from the SAME parsed coords (01's formula exactly).
                # Finding #8: walk adjacency by TRUE SEQUENCE NEIGHBOR
                # (residue_id +/- 1), NOT by consecutive cache rows. Cache-row
                # adjacency bridges internal sequence gaps: at residues ...3,10,11
                # row-adjacency made residue 3's bond_ca_next the ~15 A distance
                # to residue 10 (stored as a ~1.5 bond), or silently 0 if the
                # neighbor row was a CS-only residue absent from aa_data. By
                # indexing aa_data directly at rid-1 / rid+1, a genuine sequence
                # neighbor present in the structure is used, and an absent one
                # leaves the bond at the missing convention (0.0, matching the
                # CSV path's np.where(valid, .., 0.0) and the zero-init array).
                # The id90 CSVs predate bond_geom, so this is the authoritative
                # source and is aligned with cross/intra by construction.
                # Overwrites the CSV value.
                ri = int(rid)
                atoms_i = aa_data[ri]['atoms']
                ca_i = atoms_i.get('CA'); n_i = atoms_i.get('N'); c_i = atoms_i.get('C')
                if ca_i is not None and np.all(np.isfinite(ca_i)):
                    # Previous residue in sequence (rid - 1), if it has structure.
                    prev_entry = aa_data.get(ri - 1)
                    if prev_entry is not None:
                        ap = prev_entry['atoms']
                        cap, cprev = ap.get('CA'), ap.get('C')
                        if cap is not None and np.all(np.isfinite(cap)):
                            flat_bond_geom[g, _BG_CA_PREV] = np.linalg.norm(ca_i - cap) / 10.0
                        if (n_i is not None and cprev is not None
                                and np.all(np.isfinite(n_i)) and np.all(np.isfinite(cprev))):
                            flat_bond_geom[g, _BG_PEP_BKWD] = np.linalg.norm(n_i - cprev) / 10.0
                    # Next residue in sequence (rid + 1), if it has structure.
                    next_entry = aa_data.get(ri + 1)
                    if next_entry is not None:
                        an = next_entry['atoms']
                        can, nnext = an.get('CA'), an.get('N')
                        if can is not None and np.all(np.isfinite(can)):
                            flat_bond_geom[g, _BG_CA_NEXT] = np.linalg.norm(ca_i - can) / 10.0
                        if (c_i is not None and nnext is not None
                                and np.all(np.isfinite(c_i)) and np.all(np.isfinite(nnext))):
                            flat_bond_geom[g, _BG_PEP_FWD] = np.linalg.norm(nnext - c_i) / 10.0
        nzc = float((flat_cross_count > 0).mean())
        print(f"      cross: proteins ok={n_ok} miss={n_miss}  "
              f"residues filled={n_filled:,}/{total_residues:,}  nonzero={nzc:.3f}")
        if cross_fails[:5]:
            print(f"      sample cross fails: {cross_fails[:5]}")
    elif build_cross:
        print("      Cross requested but no build_log provided — skipping cross arrays.")

    # ========== Loss-mask: drop gap_in_structure / mismatch residues ==========
    # Exclude two residue classes from the supervised loss by clearing their
    # shift mask AFTER both flat_mismatch_idx and flat_shift_mask are fully
    # populated:
    #   - gap_in_structure: structure-less insertions. These residues have NO
    #     intramolecular distances of their own (dist_count == 0) yet still
    #     carry shift targets — training their (empty) structure against real
    #     shifts is meaningless.
    #   - mismatch: structure != shift residue. The structure of one residue is
    #     paired with another residue's shift values, so the supervision is
    #     simply wrong.
    # Use named MISMATCH_TO_IDX entries (not hardcoded ints) so this stays
    # correct if MISMATCH_TYPES ordering changes in config.py.
    _bad_mismatch = np.isin(
        flat_mismatch_idx,
        [MISMATCH_TO_IDX['gap_in_structure'], MISMATCH_TO_IDX['mismatch']],
    )
    _n_bad_res = int(_bad_mismatch.sum())
    _n_bad_targets = int(flat_shift_mask[_bad_mismatch].sum())
    flat_shift_mask[_bad_mismatch, :] = False
    print(f"      Loss-mask: excluded {_n_bad_res:,} gap_in_structure/mismatch "
          f"residues ({_n_bad_targets:,} shift targets) from the loss")

    # ========== Save structural data ==========
    print("      Saving structural data...")
    sd = cache_path / 'structural'

    np.save(sd / 'residue_idx.npy', flat_residue_idx)
    np.save(sd / 'ss_idx.npy', flat_ss_idx)
    np.save(sd / 'mismatch_idx.npy', flat_mismatch_idx)
    np.save(sd / 'dist_atom1.npy', flat_dist_atom1)
    np.save(sd / 'dist_atom2.npy', flat_dist_atom2)
    np.save(sd / 'dist_values.npy', flat_dist_values)
    np.save(sd / 'dist_count.npy', flat_dist_count)
    np.save(sd / 'dssp.npy', flat_dssp)
    np.save(sd / 'shifts.npy', flat_shifts)
    np.save(sd / 'shift_mask.npy', flat_shift_mask)
    np.save(sd / 'angles.npy', flat_angles)
    np.save(sd / 'window_idx.npy', flat_window_idx)
    np.save(sd / 'spatial_ids.npy', flat_spatial_ids)
    np.save(sd / 'spatial_dist.npy', flat_spatial_dist)
    np.save(sd / 'spatial_seq_sep.npy', flat_spatial_seq_sep)
    np.save(sd / 'res_id_lookup.npy', flat_res_id_lookup)
    np.save(sd / 'protein_offsets.npy', np.array(protein_offsets, dtype=np.int32))
    np.save(sd / 'protein_lookup_offsets.npy', np.array(protein_lookup_offsets, dtype=np.int32))
    np.save(sd / 'protein_min_res.npy', np.array(protein_min_res, dtype=np.int32))
    np.save(sd / 'protein_max_res.npy', np.array(protein_max_res, dtype=np.int32))

    np.save(sd / 'bond_geom.npy', flat_bond_geom)

    if build_cross and flat_cross_count is not None:
        np.save(sd / 'cross_atom1.npy', flat_cross_atom1)
        np.save(sd / 'cross_atom2.npy', flat_cross_atom2)
        np.save(sd / 'cross_offset.npy', flat_cross_offset)
        np.save(sd / 'cross_values.npy', flat_cross_values)
        np.save(sd / 'cross_count.npy', flat_cross_count)
        # Provenance: which PDB each BMRB's cross (and intra) coords came from.
        with open(sd / 'bmrb_pdb_used.json', 'w') as f:
            json.dump(bmrb_pdb_used, f)

    with open(sd / 'bmrb_mapping.json', 'w') as f:
        json.dump(idx_to_bmrb, f)
    with open(sd / 'global_to_resid.json', 'w') as f:
        json.dump(global_to_resid, f)

    samples = np.array(samples_list, dtype=np.int32)
    np.save(cache_path / 'samples.npy', samples)

    n_samples = len(samples_list)
    residues_no_shifts = total_residues - n_samples
    print(f"      Samples with shifts: {n_samples:,} / {total_residues:,} residues")
    print(f"      Residues with no observed shifts (excluded): {residues_no_shifts:,}")

    # Free structural memory
    del flat_dist_atom1, flat_dist_atom2, flat_dist_values
    del flat_residue_idx, flat_ss_idx, flat_mismatch_idx
    del flat_dssp, flat_shifts, flat_shift_mask, flat_angles
    del flat_spatial_ids, flat_spatial_dist, flat_spatial_seq_sep
    del flat_window_idx, flat_res_id_lookup
    gc.collect()

    # ========== Save config ==========
    stats_for_json = {}
    for col in shift_cols:
        if col in stats:
            stats_for_json[col] = {
                'mean': float(stats[col]['mean']),
                'std': float(stats[col]['std']),
            }
    # Preserve per-AA and DSSP stats
    if 'per_aa' in stats:
        stats_for_json['per_aa'] = stats['per_aa']
    if 'dssp' in stats:
        stats_for_json['dssp'] = stats['dssp']

    config = {
        'n_atom_types': n_atom_types,
        'n_dssp': n_dssp,
        'n_struct_features': N_STRUCT_FEATURES,
        'n_shifts': n_shifts,
        'window_size': window_size,
        'k_spatial': k_spatial,
        'max_valid_distances': max_valid_distances,
        'has_cross_features': bool(build_cross and flat_cross_count is not None),
        'max_cross_distances': MAX_CROSS_DISTANCES if (build_cross and flat_cross_count is not None) else 0,
        'n_cross_offset_types': N_CROSS_OFFSET_TYPES,
        'cross_dist_cutoff': CROSS_DIST_CUTOFF,
        'cross_h_cutoff': CROSS_H_CUTOFF,
        'total_residues': total_residues,
        'n_proteins': n_proteins,
        'n_samples': n_samples,
        'shift_cols': shift_cols,
        'stats': stats_for_json,
    }

    with open(cache_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ---- Provenance ----
    provenance = {
        'fold': fold,
        'n_proteins': n_proteins,
        'total_residues': total_residues,
        'n_samples': n_samples,
        'residues_no_shifts': residues_no_shifts,
    }

    return provenance


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build memory-mapped structure-only training caches.',
    )
    parser.add_argument(
        '--data_dir', default='./data',
        help='Directory containing the compiled CSV (default: ./data)',
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory for caches (default: <data_dir>/cache)',
    )
    parser.add_argument(
        '--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='Which folds to build caches for (default: 1 2 3 4 5)',
    )
    parser.add_argument(
        '--device', default='cpu',
        help='Device for cache building (default: cpu)',
    )
    parser.add_argument(
        '--reuse_stats_from', default=None,
        help='Path to existing cache stats. Can be EITHER a specific fold '
             'directory (e.g. .../cache/fold_1/) — in which case ALL folds '
             'being built use THAT fold\'s stats (correct for training with '
             'a fixed test fold) — OR a parent directory with fold_1..fold_5 '
             'subdirs, in which case each fold_k uses its own stats (useful '
             'only for reproducing an exact prior build).',
    )
    parser.add_argument(
        '--no_cross', action='store_true',
        help='Skip cross-residue distance arrays (legacy intra-only build).',
    )
    parser.add_argument(
        '--build_log', default='/home/brooks/1TB/Wynn/data_archive/build_log.csv',
        help='build_log.csv from 01: exact (pdb_path,chain) per BMRB so cross '
             'distances use coords identical to intra. Required for cross.',
    )
    parser.add_argument(
        '--dataset', choices=['experimental', 'alphafold', 'hybrid'], default=None,
        help='Which build_log dataset rows to use for cross PDB resolution. '
             'Default: inferred from data_dir basename.',
    )
    parser.add_argument(
        '--pdb_search_dirs', nargs='+',
        default=['data/pdbs', '/home/brooks/1TB/Wynn/data_archive/alphafold',
                 '/home/brooks/1TB/Wynn/data_archive/pdbs', 'data/alphafold'],
        help='Dirs to relocate build_log PDB paths by basename (cross pass).',
    )
    parser.add_argument(
        '--allow_unstamped', action='store_true',
        help='Finding #4: permit building on a split column that is NOT '
             'stamp-provenanced (i.e. possibly the raw leaky MD5(bmrb_id)%%n+1 '
             'split from 01, with no identity/UniProt dedup). By default 05 '
             'REFUSES such a build. Only pass this for a deliberate quick/legacy '
             'build where cross-fold leakage is acceptable.',
    )
    parser.add_argument(
        '--train_only_stats', action='store_true',
        help='Finding #5: compute per-fold TRAIN-ONLY normalization stats, '
             'excluding the held-out fold k AND fold 0 (UCB-200) AND fold 6 '
             '(holdout) from each fold_k\'s stats. Mutually exclusive with '
             '--reuse_stats_from (frozen/global stats). Recommended for a '
             'leak-free build; the default non-frozen path only excludes fold k.',
    )
    args = parser.parse_args()

    # Finding #5: --train_only_stats and --reuse_stats_from are mutually
    # exclusive — one computes per-fold train-only stats, the other loads a
    # fixed (frozen/global) stats table.
    if args.train_only_stats and args.reuse_stats_from:
        raise SystemExit(
            "ERROR: --train_only_stats and --reuse_stats_from are mutually "
            "exclusive. --train_only_stats computes per-fold train-only stats; "
            "--reuse_stats_from loads a fixed (frozen) stats table.")

    if args.output_dir is None:
        # Default to 1TB drive — caches are large, main disk is chronically
        # tight. A symlink at <data_dir>/cache is created below.
        ds_name = os.path.basename(os.path.abspath(args.data_dir))
        args.output_dir = f'/home/brooks/1TB/Wynn/{ds_name}_cache'

    # Cross-residue features: load build_log once (maps BMRB -> exact PDB+chain
    # used for intra distances, so cross uses identical coords). Disabled
    # gracefully if --no_cross or build_log missing.
    build_cross = not args.no_cross
    cross_build_log = None
    if build_cross:
        ds_for_log = args.dataset
        if ds_for_log is None:
            _b = os.path.basename(os.path.abspath(args.data_dir)).lower()
            ds_for_log = ('alphafold' if 'alphafold' in _b else
                          'experimental' if 'experimental' in _b else 'hybrid')
        if os.path.isfile(args.build_log):
            cross_build_log = cross04c.load_build_log(args.build_log, ds_for_log)
            print(f"  Cross features ON: build_log[{ds_for_log}] = "
                  f"{len(cross_build_log):,} entries")
        else:
            # Finding #12: a missing build_log used to silently degrade to an
            # intra-only cache (build_cross=False, has_cross_features=false) with
            # only a warning — a strictly worse model trained with no red flag.
            # Fail loudly instead: an intra-only build must be requested
            # explicitly with --no_cross.
            raise FileNotFoundError(
                f"build_log not found at {args.build_log}, so cross-residue "
                f"features cannot be built. Provide a valid --build_log, or pass "
                f"--no_cross to deliberately build a legacy intra-only cache. "
                f"Refusing to silently disable cross features.")
        # Create symlink from <data_dir>/cache → output_dir so train.py picks it up
        link_path = os.path.join(args.data_dir, 'cache')
        if not os.path.exists(link_path):
            os.makedirs(args.output_dir, exist_ok=True)
            os.symlink(args.output_dir, link_path)

    print("=" * 60)
    print("Training Cache Builder (structure-only)")
    print("=" * 60)
    print(f"  Data directory:        {args.data_dir}")
    print(f"  Output directory:      {args.output_dir}")
    print(f"  Folds:                 {args.folds}")
    print(f"  Device:                {args.device}")
    print()

    # Locate compiled CSV
    csv_path = None
    for name in ['structure_data_hybrid.csv', 'structure_data.csv', 'compiled_dataset.csv']:
        candidate = os.path.join(args.data_dir, name)
        if os.path.exists(candidate):
            csv_path = candidate
            break
    # Fallback: if only per-fold files exist, use fold_1 as a column-header source
    # (later passes are already per-fold-aware so we don't need a combined CSV).
    if csv_path is None:
        fold1_csv = os.path.join(args.data_dir, 'structure_data_hybrid_fold_1.csv')
        if os.path.exists(fold1_csv):
            csv_path = fold1_csv
            print(f"  No combined CSV found; using {csv_path} for header inspection")
    if csv_path is None:
        candidates = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
        raise FileNotFoundError(
            f"Cannot find dataset CSV in {args.data_dir}. "
            f"Available CSVs: {candidates}"
        )

    # ======================================================================
    # PASS 1: Read header only → discover columns, build atom vocabulary
    # ======================================================================
    print(f"Reading column headers from {csv_path}...")
    all_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    print(f"  {len(all_columns)} columns total")

    shift_cols = parse_shift_columns(all_columns)
    dist_col_info = parse_distance_columns(all_columns)
    dssp_cols = get_dssp_columns(all_columns)
    _, atom_to_idx = build_atom_vocabulary(dist_col_info)

    fold_col = 'split' if 'split' in all_columns else 'fold'
    if fold_col not in all_columns:
        raise ValueError("CSV must contain a 'split' or 'fold' column")

    # ======================================================================
    # Finding #4: refuse to build on an un-stamped (leaky MD5) split.
    # 01 writes split = MD5(bmrb_id) % n + 1 (no identity/UniProt/AF-vs-exp
    # dedup) and marks it un-stamped; the `stamp` stage rewrites split with the
    # leak-free identity90 map and flips the provenance to "stamped". Building a
    # cache on an un-stamped split puts same-protein duplicates in train+test.
    # We detect "stamped" provenance via (any of):
    #   - a CSV `split_provenance` column whose values are all "stamped", or
    #   - a sidecar "<source_csv>.split_provenance.json" with
    #     {"split_provenance": "stamped"}.
    # If no definitive "stamped" marker is found, ERROR unless --allow_unstamped.
    def _read_provenance_marker():
        # Gather every CSV this build might read (combined + per-fold) and check
        # for a definitive stamped/un-stamped marker. Returns one of:
        #   'stamped', 'unstamped', or None (no marker found at all).
        sources = [csv_path]
        for f in range(1, 6):
            ff = os.path.join(args.data_dir, f'structure_data_hybrid_fold_{f}.csv')
            if os.path.exists(ff):
                sources.append(ff)
        saw_unstamped = False
        saw_stamped = False
        saw_any_marker = False
        for src in sources:
            # 1) Sidecar JSON
            sidecar = src + '.split_provenance.json'
            if os.path.exists(sidecar):
                try:
                    with open(sidecar) as _sf:
                        prov = json.load(_sf).get('split_provenance')
                    if prov is not None:
                        saw_any_marker = True
                        if prov == 'stamped':
                            saw_stamped = True
                        else:
                            saw_unstamped = True
                except Exception:
                    pass
            # 2) CSV column (cheap: header tells us if it exists; read just it)
            try:
                src_cols = pd.read_csv(src, nrows=0).columns
            except Exception:
                src_cols = []
            if 'split_provenance' in src_cols:
                saw_any_marker = True
                col = pd.read_csv(src, usecols=['split_provenance'])['split_provenance']
                uniq = set(col.dropna().astype(str).unique())
                if uniq and uniq.issubset({'stamped'}):
                    saw_stamped = True
                if uniq - {'stamped'}:
                    saw_unstamped = True
        if not saw_any_marker:
            return None
        # Any un-stamped marker anywhere is disqualifying; only all-stamped passes.
        return 'unstamped' if (saw_unstamped or not saw_stamped) else 'stamped'

    _prov = _read_provenance_marker()
    if _prov == 'stamped':
        print("  Split provenance: STAMPED (leak-free identity90 map). OK.")
    elif args.allow_unstamped:
        print(f"  Split provenance: {_prov or 'NO MARKER'} — proceeding anyway "
              f"because --allow_unstamped was passed. WARNING: this split may be "
              f"the raw leaky MD5(bmrb_id) fold (same-protein duplicates split "
              f"across train/test).")
    else:
        raise SystemExit(
            "ERROR (Finding #4): the split column is not stamp-provenanced as "
            f"leak-free (found provenance: {_prov or 'NO MARKER'}). The raw 01 "
            "split is MD5(bmrb_id) % n + 1 with no identity/UniProt dedup, so "
            "building on it leaks same-protein duplicates across train/test.\n"
            "  Run the `stamp` stage to rewrite the split with the leak-free "
            "identity90 map (it sets split_provenance='stamped'), then re-run 05.\n"
            "  To deliberately build on a possibly-leaky split, pass "
            "--allow_unstamped.")

    print(f"  Shift columns:    {len(shift_cols)}")
    print(f"  Distance columns: {len(dist_col_info)}")
    print(f"  DSSP columns:     {len(dssp_cols)}")
    print(f"  Atom types:       {len(atom_to_idx)}")

    # ======================================================================
    # PASS 2: Read LIGHT columns (~100 cols) for stats + struct_lookup
    # This avoids loading all 1500+ distance columns into memory at once.
    # ======================================================================
    light_cols = set(['bmrb_id', 'residue_id', 'residue_code',
                      'secondary_structure', 'phi', 'psi', fold_col])
    light_cols.update(shift_cols)
    light_cols.update(dssp_cols)
    light_cols.update(STRUCT_DIST_COLS)
    light_cols.update(STRUCT_SC_COLS)
    for k in range(K_SPATIAL_NEIGHBORS):
        for suffix in ['_id', '_dist', '_seq_sep']:
            light_cols.add(f'spatial_neighbor_{k}{suffix}')
    light_cols = sorted(c for c in light_cols if c in all_columns)

    # Load light columns — try per-fold files to avoid pandas memory explosion
    print(f"\nLoading {len(light_cols)} light columns for stats + struct lookup...")
    fold_files = [os.path.join(args.data_dir, f'structure_data_hybrid_fold_{f}.csv')
                  for f in range(1, 6)]
    if all(os.path.exists(f) for f in fold_files):
        print("  Using per-fold CSV files...")
        light_parts = []
        for ff in fold_files:
            usecols_avail = [c for c in light_cols if c in pd.read_csv(ff, nrows=0).columns]
            part = pd.read_csv(ff, usecols=usecols_avail, dtype={'bmrb_id': str}, low_memory=False)
            light_parts.append(part)
            print(f"    {os.path.basename(ff)}: {len(part):,} rows")
        light_df = pd.concat(light_parts, ignore_index=True)
        del light_parts
        gc.collect()
    else:
        # Chunked read: the C parser materializes ALL columns before applying
        # usecols, so a plain read_csv(usecols=...) on the full wide CSV
        # (~1556 cols x millions of rows) OOMs. Reading in row-chunks keeps only
        # the light columns per chunk, bounding peak memory. Result is identical.
        _usecols = [c for c in light_cols if c in pd.read_csv(csv_path, nrows=0).columns]
        _parts = []
        for _chunk in pd.read_csv(csv_path, usecols=_usecols, dtype={'bmrb_id': str},
                                  low_memory=False, chunksize=200000):
            _parts.append(_chunk)
        light_df = pd.concat(_parts, ignore_index=True)
        del _parts
        gc.collect()
    n_residues = len(light_df)
    n_proteins = light_df['bmrb_id'].nunique()
    print(f"  {n_residues:,} residues from {n_proteins:,} proteins")

    fold_counts = light_df.groupby(fold_col)['bmrb_id'].nunique()
    print(f"  Fold distribution:")
    for f_val, cnt in fold_counts.items():
        print(f"    Fold {f_val}: {cnt} proteins")

    # Build struct_lookup (vectorized)
    print(f"\nBuilding structural feature lookup...")
    t0 = time.time()
    N = len(light_df)
    all_struct = np.zeros((N, N_STRUCT_FEATURES), dtype=np.float32)
    extract_struct_features(light_df, 0, N, all_struct)

    bmrb_ids_arr = light_df['bmrb_id'].values.astype(str)
    residue_ids_arr = light_df['residue_id'].values.astype(int)
    struct_lookup = {}
    for i in range(N):
        struct_lookup[(bmrb_ids_arr[i], residue_ids_arr[i])] = all_struct[i]
    del all_struct, bmrb_ids_arr, residue_ids_arr
    gc.collect()
    print(f"  Built lookup for {len(struct_lookup):,} residues in {time.time()-t0:.1f}s")

    all_provenance = {}
    os.makedirs(args.output_dir, exist_ok=True)

    # ======================================================================
    # PASS 3: Per-fold — chunked read of full CSV, build cache
    # ======================================================================
    for fold in args.folds:
        print(f"\n{'=' * 50}")
        print(f"  FOLD {fold}")
        print(f"{'=' * 50}")

        # Stats source (resolve ONCE before the fold loop if reusing; here we
        # resolve per-fold only if the reuse path points at a parent dir).
        if args.reuse_stats_from:
            p = args.reuse_stats_from
            # Prefer explicit fold dir (use those stats for ALL folds being built)
            if os.path.exists(os.path.join(p, 'config.json')):
                ref_cfg = os.path.join(p, 'config.json')
            else:
                # Parent-dir form: ref/fold_k/config.json, per-fold
                ref_cfg = os.path.join(p, f'fold_{fold}', 'config.json')
            if not os.path.exists(ref_cfg):
                raise FileNotFoundError(
                    f"--reuse_stats_from set but {ref_cfg} missing")
            with open(ref_cfg) as _f:
                _ref = json.load(_f)
            stats = _ref['stats']
            print(f"    Stats LOADED FROM {ref_cfg}")
            # Finding #5: the reuse/frozen path applies ONE stats table (typically
            # the global frozen_stats over ALL data) to every fold. If those
            # stats were fit including the test fold / fold 0 (UCB-200) / fold 6
            # (holdout), they are methodologically a leak (mean/std, per-AA, DSSP
            # all see held-out data). Magnitude is small (~0.005 ppm) but it
            # taints the holdout/R2 claim. Prefer --train_only_stats for a
            # leak-free per-fold-train-only build. Kept for back-compat / exact
            # reproduction of prior builds.
            print("    WARNING (Finding #5): reusing frozen/global stats — these "
                  "may include test+holdout data (methodological leak). Use "
                  "--train_only_stats for per-fold train-only stats.")
            if _ref.get('shift_cols') and _ref['shift_cols'] != shift_cols:
                raise ValueError(
                    f"shift_cols mismatch between reference cache and current CSV.\n"
                    f"  ref: {_ref['shift_cols'][:5]}... ({len(_ref['shift_cols'])})\n"
                    f"  cur: {shift_cols[:5]}... ({len(shift_cols)})")
        else:
            # Finding #5 (compute side): stats must be fit on TRAIN-ONLY data.
            # The default non-frozen behavior excluded only the held-out fold k,
            # leaking fold 0 (UCB-200) and fold 6 (holdout) into the stats. With
            # --train_only_stats we exclude folds {k, 0, 6} so mean/std, per-AA,
            # and DSSP stats see strictly the training partition.
            if args.train_only_stats:
                _excluded = {fold, 0, 6}
                train_mask = ~light_df[fold_col].isin(_excluded)
                print(f"    Train-only stats: excluding folds {sorted(_excluded)} "
                      f"(held-out {fold} + UCB-200 fold 0 + holdout fold 6)")
            else:
                # Back-compat default: exclude only the held-out fold k. NOTE:
                # this still includes fold 0 and fold 6 in the stats (a small
                # methodological leak — see --train_only_stats).
                train_mask = light_df[fold_col] != fold
            stats = compute_normalization_stats(
                light_df[train_mask], shift_cols, dssp_cols)
            n_train = train_mask.sum()
            print(f"    Stats from {light_df.loc[train_mask, 'bmrb_id'].nunique()} "
                  f"training proteins ({n_train:,} residues)")

        # Load fold data — try per-fold CSV first (memory-efficient), fall back to filtering
        fold_csv = os.path.join(args.data_dir, f'structure_data_hybrid_fold_{fold}.csv')
        t0 = time.time()
        if os.path.exists(fold_csv):
            print(f"    Loading per-fold file: {fold_csv}")
            fold_df = pd.read_csv(fold_csv, dtype={'bmrb_id': str}, low_memory=False)
        else:
            print(f"    Loading fold {fold} from full CSV (chunked)...")
            fold_chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=50000,
                                      dtype={'bmrb_id': str}, low_memory=False):
                fold_chunk = chunk[chunk[fold_col] == fold]
                if len(fold_chunk) > 0:
                    fold_chunks.append(fold_chunk)
            fold_df = pd.concat(fold_chunks, ignore_index=True)
            del fold_chunks
            gc.collect()
        print(f"    Loaded {fold_df['bmrb_id'].nunique()} proteins, "
              f"{len(fold_df):,} residues in {time.time()-t0:.1f}s")

        # Build cache
        cache_dir = os.path.join(args.output_dir, f'fold_{fold}')
        t0 = time.time()

        provenance = build_cache_for_fold(
            df=fold_df,
            fold=fold,
            shift_cols=shift_cols,
            dist_col_info=dist_col_info,
            dssp_cols=dssp_cols,
            atom_to_idx=atom_to_idx,
            stats=stats,
            cache_dir=cache_dir,
            struct_lookup=struct_lookup,
            build_cross=build_cross,
            build_log=cross_build_log,
            pdb_search_dirs=args.pdb_search_dirs,
        )

        elapsed = time.time() - t0
        provenance['build_time_seconds'] = elapsed
        all_provenance[fold] = provenance
        print(f"    Cache built in {elapsed:.1f}s")

        del fold_df
        gc.collect()

    # Cleanup
    del light_df, struct_lookup

    # ---- Final Provenance Report ----
    print("\n" + "=" * 60)
    print("PROVENANCE REPORT")
    print("=" * 60)
    for fold, prov in sorted(all_provenance.items()):
        print(f"\n  Fold {fold}:")
        print(f"    Proteins:                {prov['n_proteins']}")
        print(f"    Total residues:          {prov['total_residues']:,}")
        print(f"    Samples (with shifts):   {prov['n_samples']:,}")
        print(f"    Residues excluded (no shifts): {prov['residues_no_shifts']:,}")
        print(f"    Build time:              {prov['build_time_seconds']:.1f}s")
    print("=" * 60)

    # Save full provenance to JSON
    provenance_path = os.path.join(args.output_dir, 'build_provenance.json')
    # Convert numpy types for JSON serialization
    prov_serializable = {}
    for fold, prov in all_provenance.items():
        prov_serializable[str(fold)] = {
            k: (int(v) if isinstance(v, (np.integer,)) else
                float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in prov.items()
        }
    with open(provenance_path, 'w') as f:
        json.dump(prov_serializable, f, indent=2)
    print(f"\nFull provenance saved to {provenance_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
