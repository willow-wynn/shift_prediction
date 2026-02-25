#!/usr/bin/env python
"""
05 - Training Cache Builder (Better Data Pipeline)

Pre-computes retrieval results and saves a memory-mapped cache so that
training reads from disk (via numpy memmap) rather than hitting FAISS and
HDF5 at every epoch.

Adapted from the CachedRetrievalDataset.create() classmethod in
homologies/dataset_cached.py (and homologies_better_data/dataset.py) with:
- Constants imported from config.py
- Full provenance logging: proteins cached, retrieval stats, failures
- Checkpoint/resume for crash safety
- Standalone argparse CLI

For each fold, this script:
1. Loads the compiled CSV (filtering to proteins in that fold)
2. Loads the FAISS index for that fold
3. Builds compact structural arrays (residue types, distances, DSSP, etc.)
4. Performs batch retrieval via FAISS (with same-protein exclusion)
5. Saves everything as memory-mapped numpy arrays

Cache layout (per fold):
    <output_dir>/fold_{k}/
        config.json
        samples.npy
        structural/  (residue_idx.npy, ss_idx.npy, ... , physics.npy)
        retrieval/   (shifts.npy, shift_masks.npy, residue_codes.npy, ...)

Usage:
    python 05_build_training_cache.py \\
        --data_dir ./data \\
        --embeddings ./data/esm_embeddings.h5 \\
        --index_dir ./data/retrieval_indices \\
        --output_dir ./data/cache
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
    K_RETRIEVED,
    FAISS_NPROBE,
    CONTEXT_WINDOW,
    K_SPATIAL_NEIGHBORS,
    MAX_VALID_DISTANCES,
)


# ============================================================================
# Utility Functions (same as dataset.py)
# ============================================================================

def parse_distance_columns(columns):
    """Parse distance column names to extract atom pairs."""
    dist_cols = []
    pattern = re.compile(r'^dist_([A-Z0-9]+)_([A-Z0-9]+)$')
    for col in columns:
        match = pattern.match(col)
        if match:
            atom1, atom2 = match.groups()
            dist_cols.append((col, atom1, atom2))
    return dist_cols


def build_atom_vocabulary(dist_col_info):
    """Build vocabulary of unique atom types from distance columns."""
    atoms = set()
    for _, atom1, atom2 in dist_col_info:
        atoms.add(atom1)
        atoms.add(atom2)
    atom_list = sorted(atoms)
    atom_to_idx = {a: i for i, a in enumerate(atom_list)}
    return atom_list, atom_to_idx


def parse_shift_columns(columns):
    """Get chemical shift columns."""
    return sorted([c for c in columns if c.endswith('_shift')])


def get_dssp_columns(df_columns):
    """Get available DSSP columns."""
    return [c for c in DSSP_COLS if c in df_columns]


# Physics feature columns (from dataset.py)
PHYSICS_COLS = [
    'ring_current_h', 'ring_current_ha',
    'hse_up', 'hse_down', 'hse_ratio',
    'hbond_dist_1', 'hbond_energy_1',
    'hbond_dist_2', 'hbond_energy_2',
    'order_parameter',
]


def get_physics_columns(df_columns):
    """Get available physics feature columns."""
    return [c for c in PHYSICS_COLS if c in df_columns]


# ============================================================================
# Cache Builder
# ============================================================================

def compute_normalization_stats(df, shift_cols, dssp_cols):
    """Compute mean/std for shift and DSSP columns from the training data.

    Only non-NaN values are considered.
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

    return stats


def build_cache_for_fold(
    df: pd.DataFrame,
    fold: int,
    shift_cols: list,
    dist_col_info: list,
    dssp_cols: list,
    atom_to_idx: dict,
    stats: dict,
    embedding_lookup,
    retriever,
    cache_dir: str,
    context_window: int = CONTEXT_WINDOW,
    k_spatial: int = K_SPATIAL_NEIGHBORS,
    k_retrieved: int = K_RETRIEVED,
    max_valid_distances: int = MAX_VALID_DISTANCES,
    retrieval_batch_size: int = 5000,
    physics_cols: list = None,
):
    """Build a complete training cache for one fold.

    This is essentially CachedRetrievalDataset.create() pulled out as a
    standalone function with full provenance logging and checkpoint/resume.
    """
    cache_path = Path(cache_dir)

    # Auto-detect physics columns if not provided
    if physics_cols is None:
        physics_cols = get_physics_columns(df.columns)
    n_physics = len(physics_cols)

    # Check for resume
    checkpoint_file = cache_path / 'retrieval' / '_checkpoint.txt'
    is_resuming = checkpoint_file.exists()

    if is_resuming:
        print(f"    *** FOUND CHECKPOINT - RESUMING BUILD ***")
    else:
        if cache_path.exists():
            shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True)
        (cache_path / 'structural').mkdir()
        (cache_path / 'retrieval').mkdir()

    n_shifts = len(shift_cols)
    n_atom_types = len(atom_to_idx)
    n_dssp = len(dssp_cols)
    window_size = 2 * context_window + 1

    proteins = list(df.groupby('bmrb_id'))
    total_residues = sum(len(pdf) for _, pdf in proteins)
    n_proteins = len(proteins)

    M = max_valid_distances
    K = k_retrieved

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
    print(f"      Physics columns: {n_physics}")
    print(f"      Context window: {window_size}")
    print(f"      K spatial: {k_spatial}")
    print(f"      K retrieved: {K}")
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

    # Physics features
    flat_physics = (
        np.zeros((total_residues, n_physics), dtype=np.float16)
        if n_physics > 0 else None
    )

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

        for local_idx, rid in enumerate(residue_ids):
            global_idx = start_idx + local_idx
            flat_res_id_lookup[current_lookup_offset + (rid - min_res)] = global_idx
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
                flat_shift_mask[start_idx:start_idx + n, si] = valid

                if col in stats:
                    mean, std = stats[col]['mean'], stats[col]['std']
                    if std > 1e-6:
                        normalized = np.where(valid, (vals - mean) / std, 0.0)
                    else:
                        normalized = np.where(valid, vals - mean, 0.0)
                    normalized = np.clip(normalized, -10, 10)
                    flat_shifts[start_idx:start_idx + n, si] = normalized

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

        # Physics features
        if flat_physics is not None and n_physics > 0:
            for pi, col in enumerate(physics_cols):
                if col in pdf.columns:
                    vals = pdf[col].values
                    valid = ~np.isnan(vals)
                    flat_physics[start_idx:start_idx + n, pi] = np.where(valid, vals, 0.0)

        # Build samples (only residues with at least one observed shift)
        for local_idx in range(n):
            global_idx = start_idx + local_idx
            if flat_shift_mask[global_idx].any():
                samples_list.append((global_idx, prot_idx))

        current_offset += n
        current_lookup_offset += span

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

    if flat_physics is not None:
        np.save(sd / 'physics.npy', flat_physics)

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
    if flat_physics is not None:
        del flat_physics
    gc.collect()

    # ========== Build retrieval data (with checkpoint/resume) ==========
    print("      Building retrieval data (in batches)...")

    rd = cache_path / 'retrieval'
    checkpoint_file = rd / '_checkpoint.txt'

    resume_from = 0
    if checkpoint_file.exists():
        try:
            resume_from = int(checkpoint_file.read_text().strip())
            print(f"      RESUMING from batch starting at index {resume_from}")
        except Exception:
            resume_from = 0

    mmap_mode = 'r+' if resume_from > 0 else 'w+'

    retrieved_shifts = np.lib.format.open_memmap(
        rd / 'shifts.npy', mode=mmap_mode, dtype=np.float16,
        shape=(total_residues, K, n_shifts),
    )
    retrieved_shift_masks = np.lib.format.open_memmap(
        rd / 'shift_masks.npy', mode=mmap_mode, dtype=bool,
        shape=(total_residues, K, n_shifts),
    )
    retrieved_residue_codes = np.lib.format.open_memmap(
        rd / 'residue_codes.npy', mode=mmap_mode, dtype=np.int16,
        shape=(total_residues, K),
    )
    retrieved_distances = np.lib.format.open_memmap(
        rd / 'distances.npy', mode=mmap_mode, dtype=np.float16,
        shape=(total_residues, K),
    )
    retrieved_valid = np.lib.format.open_memmap(
        rd / 'valid.npy', mode=mmap_mode, dtype=bool,
        shape=(total_residues, K),
    )

    # Retrieval stats accumulators
    total_queries = 0
    total_valid_queries = 0
    total_valid_retrieved = 0
    sim_sum = 0.0
    retrieval_failures = 0

    total_batches = (total_residues + retrieval_batch_size - 1) // retrieval_batch_size
    start_batch = resume_from // retrieval_batch_size

    for batch_start in tqdm(
        range(resume_from, total_residues, retrieval_batch_size),
        desc="      Retrieval batches",
        initial=start_batch,
        total=total_batches,
    ):
        batch_end = min(batch_start + retrieval_batch_size, total_residues)
        batch_indices = list(range(batch_start, batch_end))

        batch_bmrb_ids = []
        batch_res_ids = []

        for global_idx in batch_indices:
            bmrb_id = idx_to_bmrb[str(global_idx)]
            rid = global_to_resid[str(global_idx)]
            batch_bmrb_ids.append(bmrb_id)
            batch_res_ids.append(rid)

        total_queries += len(batch_indices)

        try:
            batch_embeddings, batch_valid_mask = embedding_lookup.get_batch(
                batch_bmrb_ids, batch_res_ids,
            )
        except Exception as e:
            retrieval_failures += len(batch_indices)
            print(f"      WARNING: Embedding lookup failed for batch "
                  f"{batch_start}-{batch_end}: {e}")
            checkpoint_file.write_text(str(batch_end))
            continue

        valid_indices = np.where(batch_valid_mask)[0]
        total_valid_queries += len(valid_indices)

        if len(valid_indices) > 0:
            valid_embeddings = batch_embeddings[valid_indices]
            valid_bmrb_ids = [batch_bmrb_ids[i] for i in valid_indices]

            try:
                results = retriever.retrieve(
                    query_embeddings=valid_embeddings,
                    query_bmrb_ids=valid_bmrb_ids,
                    k=K,
                )
            except Exception as e:
                retrieval_failures += len(valid_indices)
                print(f"      WARNING: Retrieval failed for batch "
                      f"{batch_start}-{batch_end}: {e}")
                checkpoint_file.write_text(str(batch_end))
                continue

            # Store retrieval results
            global_indices_arr = batch_start + valid_indices
            retrieved_shifts[global_indices_arr] = results['shifts']
            retrieved_shift_masks[global_indices_arr] = results['shift_masks']
            retrieved_residue_codes[global_indices_arr] = results['residue_codes']
            retrieved_distances[global_indices_arr] = results['distances']
            retrieved_valid[global_indices_arr] = results['indices'] >= 0

            # Accumulate retrieval stats
            valid_mask = results['indices'] >= 0
            total_valid_retrieved += valid_mask.sum()
            if valid_mask.any():
                sim_sum += results['distances'][valid_mask].sum()

        # Flush and checkpoint every batch
        retrieved_shifts.flush()
        retrieved_shift_masks.flush()
        retrieved_residue_codes.flush()
        retrieved_distances.flush()
        retrieved_valid.flush()

        checkpoint_file.write_text(str(batch_end))

        if (batch_end % (retrieval_batch_size * 10)) == 0:
            gc.collect()

    # Final flush
    retrieved_shifts.flush()
    retrieved_shift_masks.flush()
    retrieved_residue_codes.flush()
    retrieved_distances.flush()
    retrieved_valid.flush()

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("      Retrieval data complete - checkpoint removed")

    # Retrieval stats
    avg_sim = sim_sum / max(total_valid_retrieved, 1)
    avg_coverage = total_valid_retrieved / max(total_valid_queries * K, 1)

    del retrieved_shifts, retrieved_shift_masks, retrieved_residue_codes
    del retrieved_distances, retrieved_valid
    gc.collect()

    # ========== Save config ==========
    stats_for_json = {}
    for col in shift_cols:
        if col in stats:
            stats_for_json[col] = {
                'mean': float(stats[col]['mean']),
                'std': float(stats[col]['std']),
            }

    config = {
        'n_atom_types': n_atom_types,
        'n_dssp': n_dssp,
        'n_physics': n_physics,
        'n_shifts': n_shifts,
        'window_size': window_size,
        'k_spatial': k_spatial,
        'k_retrieved': k_retrieved,
        'max_valid_distances': max_valid_distances,
        'total_residues': total_residues,
        'n_proteins': n_proteins,
        'n_samples': n_samples,
        'shift_cols': shift_cols,
        'stats': stats_for_json,
        'physics_cols': physics_cols,
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
        'total_queries': total_queries,
        'valid_queries': total_valid_queries,
        'retrieval_failures': retrieval_failures,
        'avg_cosine_similarity': float(avg_sim),
        'retrieval_coverage': float(avg_coverage),
    }

    return provenance


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build memory-mapped training caches with pre-computed retrieval.',
    )
    parser.add_argument(
        '--data_dir', default='./data',
        help='Directory containing the compiled CSV (default: ./data)',
    )
    parser.add_argument(
        '--embeddings', default=None,
        help='Path to ESM embeddings HDF5 (default: <data_dir>/esm_embeddings.h5)',
    )
    parser.add_argument(
        '--index_dir', default=None,
        help='Directory with FAISS indices (default: <data_dir>/retrieval_indices)',
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory for caches (default: <data_dir>/cache)',
    )
    parser.add_argument(
        '--k', type=int, default=K_RETRIEVED,
        help=f'Number of retrieved neighbors (default: {K_RETRIEVED})',
    )
    parser.add_argument(
        '--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='Which folds to build caches for (default: 1 2 3 4 5)',
    )
    parser.add_argument(
        '--retrieval_batch_size', type=int, default=5000,
        help='Batch size for retrieval queries (default: 5000)',
    )
    parser.add_argument(
        '--device', default='cpu',
        help='Device for FAISS retrieval (default: cpu)',
    )
    args = parser.parse_args()

    if args.embeddings is None:
        args.embeddings = os.path.join(args.data_dir, 'esm_embeddings.h5')
    if args.index_dir is None:
        args.index_dir = os.path.join(args.data_dir, 'retrieval_indices')
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'cache')

    print("=" * 60)
    print("Training Cache Builder (Better Data Pipeline)")
    print("=" * 60)
    print(f"  Data directory:        {args.data_dir}")
    print(f"  Embeddings:            {args.embeddings}")
    print(f"  Index directory:       {args.index_dir}")
    print(f"  Output directory:      {args.output_dir}")
    print(f"  K retrieved:           {args.k}")
    print(f"  Folds:                 {args.folds}")
    print(f"  Retrieval batch size:  {args.retrieval_batch_size}")
    print(f"  Device:                {args.device}")
    print()

    # Locate compiled CSV
    # Try common dataset names in order of preference
    csv_path = None
    for name in ['structure_data.csv', 'compiled_dataset.csv', 'sidechain_structure_data.csv', 'small_structure_data.csv']:
        candidate = os.path.join(args.data_dir, name)
        if os.path.exists(candidate):
            csv_path = candidate
            break
    if csv_path is None:
        candidates = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
        raise FileNotFoundError(
            f"Cannot find dataset CSV in {args.data_dir}. "
            f"Available CSVs: {candidates}"
        )

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'bmrb_id': str})
    n_residues = len(df)
    n_proteins = df['bmrb_id'].nunique()
    print(f"  Loaded {n_residues:,} residues from {n_proteins:,} proteins")

    # Determine fold column
    if 'split' in df.columns:
        fold_col = 'split'
    elif 'fold' in df.columns:
        fold_col = 'fold'
    else:
        raise ValueError("CSV must contain a 'split' or 'fold' column")

    fold_counts = df.groupby(fold_col)['bmrb_id'].nunique()
    print(f"  Fold distribution (proteins):")
    for f_val, cnt in fold_counts.items():
        print(f"    Fold {f_val}: {cnt} proteins")

    # Parse columns
    shift_cols = parse_shift_columns(df.columns.tolist())
    dist_col_info = parse_distance_columns(df.columns.tolist())
    dssp_cols = get_dssp_columns(df.columns.tolist())
    physics_cols = get_physics_columns(df.columns.tolist())
    _, atom_to_idx = build_atom_vocabulary(dist_col_info)

    print(f"\n  Shift columns:    {len(shift_cols)}")
    print(f"  Distance columns: {len(dist_col_info)}")
    print(f"  DSSP columns:     {len(dssp_cols)}")
    print(f"  Physics columns:  {len(physics_cols)}")
    print(f"  Atom types:       {len(atom_to_idx)}")

    # Import retrieval module (lazy to avoid import errors if faiss not installed)
    from retrieval import Retriever, EmbeddingLookup

    # Load embedding lookup once
    print(f"\nLoading embedding lookup from {args.embeddings}...")
    embedding_lookup = EmbeddingLookup(args.embeddings)

    all_provenance = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for fold in args.folds:
        print(f"\n{'=' * 50}")
        print(f"  FOLD {fold}")
        print(f"{'=' * 50}")

        # Filter to proteins in this fold
        fold_df = df[df[fold_col] == fold].copy()
        n_fold_proteins = fold_df['bmrb_id'].nunique()
        n_fold_residues = len(fold_df)

        print(f"    Proteins in fold {fold}: {n_fold_proteins}")
        print(f"    Residues in fold {fold}: {n_fold_residues:,}")

        if n_fold_proteins == 0:
            print(f"    WARNING: No proteins in fold {fold}, skipping")
            continue

        # Compute normalization stats from training data (all folds EXCEPT this one)
        train_df = df[df[fold_col] != fold]
        stats = compute_normalization_stats(train_df, shift_cols, dssp_cols)
        print(f"    Normalization stats computed from {train_df['bmrb_id'].nunique()} "
              f"training proteins ({len(train_df):,} residues)")

        # Load retriever for this fold
        print(f"    Loading FAISS index (excluding fold {fold})...")
        retriever = Retriever(
            index_dir=args.index_dir,
            exclude_fold=fold,
            k=args.k,
            nprobe=FAISS_NPROBE,
            device=args.device,
        )

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
            embedding_lookup=embedding_lookup,
            retriever=retriever,
            cache_dir=cache_dir,
            k_retrieved=args.k,
            retrieval_batch_size=args.retrieval_batch_size,
            physics_cols=physics_cols,
        )

        elapsed = time.time() - t0
        provenance['build_time_seconds'] = elapsed
        all_provenance[fold] = provenance

        print(f"    Cache built in {elapsed:.1f}s")

        # Free retriever memory
        del retriever
        gc.collect()

    # Close embedding lookup
    embedding_lookup.close()

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
        print(f"    Retrieval queries:       {prov['total_queries']:,}")
        print(f"    Valid queries:           {prov['valid_queries']:,}")
        print(f"    Retrieval failures:      {prov['retrieval_failures']:,}")
        print(f"    Avg cosine similarity:   {prov['avg_cosine_similarity']:.4f}")
        print(f"    Retrieval coverage:      {prov['retrieval_coverage']:.4f}")
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
