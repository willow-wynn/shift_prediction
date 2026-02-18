#!/usr/bin/env python3
"""Compile the clean dataset from chemical shifts + PDB structures.

Ties together all Wave 1 modules:
  - pdb_utils: PDB parsing, DSSP, B-factors
  - alignment: sequence alignment between shift and structure data
  - distance_features: dense backbone distances + sidechain summaries
  - spatial_neighbors: K nearest spatial neighbors (CA-based)
  - physics_features: physics-informed features
  - data_quality: filtering with full provenance tracking
  - random_coil: random coil shift tables

Inputs:
  data/chemical_shifts.csv  (output of 00_fetch)
  data/pdb_selection.csv    (output of 01_select)

Outputs:
  data/structure_data.csv           -- main dataset
  data/quality_log.csv              -- provenance log from data_quality
  data/quality_log_summary.csv      -- summary of filtering steps
  data/compilation_report.txt       -- human-readable report
"""

import argparse
import os
import sys
import hashlib
import json
import traceback
import time
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_DIR,
    PDB_DIR,
    DSSP_COLS,
    SHIFT_RANGES,
    STANDARD_RESIDUES,
    AA_3_TO_1,
    K_SPATIAL_NEIGHBORS,
)
from pdb_utils import parse_pdb, run_dssp, extract_bfactors
from alignment import align_sequences, extract_sequences, to_single_letter
from distance_features import (
    compute_all_distance_features,
    get_distance_column_names,
    get_sidechain_summary_names,
)
from spatial_neighbors import find_neighbors
from data_quality import run_full_quality_pipeline, FilterLog, report_quality

# Physics features may not exist yet -- import conditionally
try:
    from physics_features import compute_all_physics_features, get_physics_feature_names
    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False

    def compute_all_physics_features(*args, **kwargs):
        return {}

    def get_physics_feature_names():
        return []


# ============================================================================
# Helpers
# ============================================================================

def assign_fold(bmrb_id, n_folds=5):
    """Deterministic fold assignment using MD5 hash of bmrb_id."""
    return int(hashlib.md5(str(bmrb_id).encode()).hexdigest(), 16) % n_folds + 1


def build_struct_data(residues):
    """Convert parse_pdb output to the dict format expected by spatial_neighbors.

    parse_pdb returns {(chain, res_id): {residue_name, atoms}} keyed by tuple.
    spatial_neighbors.find_neighbors expects {res_id: {atoms: ...}}.

    Returns:
        struct_data: dict mapping res_id -> {residue_name, atoms}
        chain_used:  the chain letter actually used (for logging)
    """
    struct_data = {}
    chain_used = None
    for (chain, res_id), rdata in residues.items():
        if chain_used is None:
            chain_used = chain
        struct_data[res_id] = rdata
    return struct_data, chain_used


def get_structure_sequence(residues):
    """Extract a residue-id-ordered sequence from parsed PDB residues.

    Args:
        residues: dict from parse_pdb -- {(chain, res_id): {residue_name, atoms}}

    Returns:
        dict with keys: residue_ids, residue_names, sequence
    """
    items = sorted(residues.items(), key=lambda kv: kv[0][1])
    res_ids = []
    res_names = []
    for (chain, res_id), rdata in items:
        rname = rdata['residue_name']
        if rname in AA_3_TO_1:
            res_ids.append(res_id)
            res_names.append(rname)
    if not res_ids:
        return None
    return {
        'residue_ids': res_ids,
        'residue_names': res_names,
        'sequence': to_single_letter(res_names),
    }


def get_shift_sequence(shifts_df):
    """Extract residue-id-ordered sequence from a single protein's shift df.

    Args:
        shifts_df: DataFrame for one protein, must contain residue_id and residue_code

    Returns:
        dict with keys: residue_ids, residue_names, sequence
    """
    sub = shifts_df[['residue_id', 'residue_code']].drop_duplicates().sort_values('residue_id')
    # Keep only amino acids
    valid = sub['residue_code'].apply(lambda x: str(x).upper() in AA_3_TO_1)
    sub = sub[valid]
    if len(sub) == 0:
        return None
    return {
        'residue_ids': sub['residue_id'].tolist(),
        'residue_names': sub['residue_code'].tolist(),
        'sequence': to_single_letter(sub['residue_code'].tolist()),
    }


def alignment_to_mapping(alignment, struct_seq, shift_seq):
    """Convert a Biopython alignment into a residue-level mapping.

    Also classifies each position as: match, mismatch, protein_edge.

    Returns:
        list of dicts with keys: struct_res_id, shift_res_id, mismatch_type
    """
    a1, a2 = str(alignment[0]), str(alignment[1])
    i1, i2 = 0, 0
    n_struct = len(struct_seq['residue_ids'])
    n_shift = len(shift_seq['residue_ids'])
    mappings = []

    for c1, c2 in zip(a1, a2):
        if c1 != '-' and c2 != '-':
            struct_rid = struct_seq['residue_ids'][i1]
            shift_rid = shift_seq['residue_ids'][i2]

            # Classify mismatch type
            if c1 == c2:
                # Check if near protein edge (first or last 2 residues)
                if i1 < 2 or i1 >= n_struct - 2 or i2 < 2 or i2 >= n_shift - 2:
                    mtype = 'protein_edge'
                else:
                    mtype = 'match'
            else:
                mtype = 'mismatch'

            mappings.append({
                'struct_res_id': struct_rid,
                'shift_res_id': shift_rid,
                'mismatch_type': mtype,
            })
        # Also track gaps for logging (but don't produce rows for them)
        if c1 != '-':
            i1 += 1
        if c2 != '-':
            i2 += 1

    return mappings


# ============================================================================
# Per-protein processing
# ============================================================================

def process_protein(bmrb_id, shifts_df, pdb_path, chain_id, log):
    """Process a single protein: parse PDB, align, compute features.

    Args:
        bmrb_id: BMRB identifier (str)
        shifts_df: DataFrame of chemical shifts for this protein
        pdb_path: Path to the PDB file
        chain_id: Chain identifier (or None)
        log: FilterLog for provenance tracking

    Returns:
        list of row dicts (one per aligned residue), or empty list on failure
    """
    rows = []

    # ---- Step 1: Parse PDB ----
    residues = parse_pdb(pdb_path, chain_id=chain_id)
    if not residues:
        log.add('parse_pdb', 'No residues parsed from PDB', bmrb_id=bmrb_id, action='skipped')
        return rows

    # ---- Step 2: Run DSSP ----
    dssp_data = run_dssp(pdb_path)

    # ---- Step 3: Extract B-factors ----
    bfactors = extract_bfactors(pdb_path, chain_id=chain_id)

    # ---- Step 4: Build structure data and get sequences ----
    struct_data, chain_used = build_struct_data(residues)
    struct_seq = get_structure_sequence(residues)
    shift_seq = get_shift_sequence(shifts_df)

    if struct_seq is None:
        log.add('alignment', 'No amino acids in structure', bmrb_id=bmrb_id, action='skipped')
        return rows
    if shift_seq is None:
        log.add('alignment', 'No amino acids in shifts', bmrb_id=bmrb_id, action='skipped')
        return rows

    # ---- Step 5: Align sequences ----
    alignment = align_sequences(struct_seq['sequence'], shift_seq['sequence'])
    if alignment is None:
        log.add('alignment', 'Alignment failed', bmrb_id=bmrb_id, action='skipped')
        return rows

    mapping = alignment_to_mapping(alignment, struct_seq, shift_seq)
    if not mapping:
        log.add('alignment', 'No aligned positions', bmrb_id=bmrb_id, action='skipped')
        return rows

    # ---- Step 6: Compute spatial neighbors ----
    neighbors = find_neighbors(struct_data)

    # ---- Step 7: Index shift data for fast lookup ----
    shift_indexed = shifts_df.drop_duplicates(subset='residue_id').set_index('residue_id')

    # Detect shift and ambiguity columns
    shift_cols = sorted([c for c in shifts_df.columns if c.endswith('_shift')])
    ambig_cols = sorted([c for c in shifts_df.columns if c.endswith('_ambiguity_code')])

    # ---- Step 7b: Pre-compute chain-wide coords for physics features ----
    if HAS_PHYSICS:
        sorted_rids = sorted(struct_data.keys())
        all_ca_coords = []
        all_cb_coords = []
        rid_to_idx = {}
        all_residue_dicts = {}  # rid -> per-residue atom coords dict
        for idx, rid in enumerate(sorted_rids):
            rdata = struct_data[rid]
            atoms = rdata.get('atoms', {})
            rid_to_idx[rid] = idx
            all_residue_dicts[rid] = rdata
            ca = atoms.get('CA')
            cb = atoms.get('CB')
            all_ca_coords.append(ca if ca is not None and np.all(np.isfinite(ca)) else np.array([np.nan]*3))
            all_cb_coords.append(cb if cb is not None and np.all(np.isfinite(cb)) else np.array([np.nan]*3))
        chain_ca = np.array(all_ca_coords)
        chain_cb = np.array(all_cb_coords)

    # ---- Step 8: Build rows for each aligned residue ----
    for m in mapping:
        struct_rid = m['struct_res_id']
        shift_rid = m['shift_res_id']
        mismatch_type = m['mismatch_type']

        if shift_rid not in shift_indexed.index:
            continue

        shift_row = shift_indexed.loc[shift_rid]
        # Handle rare case of duplicate index
        if isinstance(shift_row, pd.DataFrame):
            shift_row = shift_row.iloc[0]

        # Get residue code (from shift data, which has the 3-letter code)
        residue_code = shift_row.get('residue_code', 'UNK')
        if pd.isna(residue_code):
            residue_code = 'UNK'

        # Base row
        row = {
            'bmrb_id': str(bmrb_id),
            'residue_id': struct_rid,
            'residue_code': residue_code,
            'mismatch_type': mismatch_type,
        }

        # ---- Chemical shift values ----
        for col in shift_cols + ambig_cols:
            val = shift_row.get(col)
            if val is not None and not pd.isna(val):
                row[col] = val

        # ---- Structural features from parsed PDB ----
        if struct_rid in struct_data:
            rdata = struct_data[struct_rid]
            atoms = rdata.get('atoms', {})

            # CA coordinates
            if 'CA' in atoms and np.all(np.isfinite(atoms['CA'])):
                row['ca_x'], row['ca_y'], row['ca_z'] = atoms['CA'].tolist()

            # Distance features (dense backbone + sidechain summary)
            dist_feats = compute_all_distance_features(atoms)
            row.update(dist_feats)

            # Physics features
            if HAS_PHYSICS:
                residue_data = {
                    'resname': residue_code,
                    'atoms': atoms,
                    'residue_idx': rid_to_idx.get(struct_rid, 0),
                    'ca_coords': chain_ca,
                    'cb_coords': chain_cb,
                    'all_coords': all_residue_dicts,
                }
                # Build neighbor_data: nearby residues for ring current
                nb_data = []
                if struct_rid in neighbors:
                    for nb_id in neighbors[struct_rid].get('ids', []):
                        if nb_id >= 0 and nb_id in struct_data:
                            nb_rdata = struct_data[nb_id]
                            nb_data.append({
                                'resname': nb_rdata.get('residue_name', ''),
                                'atoms': nb_rdata.get('atoms', {}),
                            })
                # DSSP data for this residue
                dssp_key = (chain_used or '', struct_rid)
                dssp_for_phys = dssp_data.get(dssp_key)
                if dssp_for_phys is None:
                    for alt_key in [('', struct_rid), (' ', struct_rid)]:
                        if alt_key in dssp_data:
                            dssp_for_phys = dssp_data[alt_key]
                            break
                bf = bfactors.get((chain_used or '', struct_rid))
                phys_feats = compute_all_physics_features(residue_data, nb_data, dssp_data=dssp_for_phys, bfactor=bf)
                row.update(phys_feats)

        # ---- DSSP data ----
        dssp_key = (chain_used or '', struct_rid)
        if dssp_key in dssp_data:
            dssp_entry = dssp_data[dssp_key]
            row['secondary_structure'] = dssp_entry.get('secondary_structure', 'C')

            phi_val = dssp_entry.get('phi')
            psi_val = dssp_entry.get('psi')
            row['phi'] = phi_val if phi_val is not None else np.nan
            row['psi'] = psi_val if psi_val is not None else np.nan

            # Relative accessibility and H-bond columns
            for col in DSSP_COLS:
                val = dssp_entry.get(col)
                if val is not None:
                    row[col] = val
        else:
            # Try with empty chain (some PDB files have no chain id)
            for alt_key in [('', struct_rid), (' ', struct_rid)]:
                if alt_key in dssp_data:
                    dssp_entry = dssp_data[alt_key]
                    row['secondary_structure'] = dssp_entry.get('secondary_structure', 'C')
                    phi_val = dssp_entry.get('phi')
                    psi_val = dssp_entry.get('psi')
                    row['phi'] = phi_val if phi_val is not None else np.nan
                    row['psi'] = psi_val if psi_val is not None else np.nan
                    for col in DSSP_COLS:
                        val = dssp_entry.get(col)
                        if val is not None:
                            row[col] = val
                    break

        # ---- B-factor ----
        bf_key = (chain_used or '', struct_rid)
        if bf_key in bfactors:
            row['bfactor'] = bfactors[bf_key]
        else:
            for alt_key in [('', struct_rid), (' ', struct_rid)]:
                if alt_key in bfactors:
                    row['bfactor'] = bfactors[alt_key]
                    break

        # ---- Spatial neighbors ----
        if struct_rid in neighbors:
            nb = neighbors[struct_rid]
            for i in range(K_SPATIAL_NEIGHBORS):
                row[f'spatial_neighbor_{i}_id'] = nb['ids'][i]
                row[f'spatial_neighbor_{i}_dist'] = nb['dists'][i]
                row[f'spatial_neighbor_{i}_seq_sep'] = nb['seps'][i]

        rows.append(row)

    return rows


# ============================================================================
# Main compilation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compile clean dataset from chemical shifts + PDB structures.'
    )
    parser.add_argument(
        '--shifts-file',
        default=os.path.join(DATA_DIR, 'chemical_shifts.csv'),
        help='Path to chemical shifts CSV (default: data/chemical_shifts.csv)',
    )
    parser.add_argument(
        '--pdb-selection',
        default=os.path.join(DATA_DIR, 'pdb_selection.csv'),
        help='Path to PDB selection CSV (default: data/pdb_selection.csv)',
    )
    parser.add_argument(
        '--output',
        default=os.path.join(DATA_DIR, 'structure_data.csv'),
        help='Output CSV path (default: data/structure_data.csv)',
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)',
    )
    parser.add_argument(
        '--pdb-dir',
        default=PDB_DIR,
        help='Directory containing PDB files (default: data/pdbs)',
    )
    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shifts_file = os.path.join(script_dir, args.shifts_file) if not os.path.isabs(args.shifts_file) else args.shifts_file
    pdb_selection_file = os.path.join(script_dir, args.pdb_selection) if not os.path.isabs(args.pdb_selection) else args.pdb_selection
    output_file = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    pdb_dir = os.path.join(script_dir, args.pdb_dir) if not os.path.isabs(args.pdb_dir) else args.pdb_dir

    print("=" * 80)
    print("DATASET COMPILATION")
    print("=" * 80)
    print(f"  Shifts file:    {shifts_file}")
    print(f"  PDB selection:  {pdb_selection_file}")
    print(f"  PDB directory:  {pdb_dir}")
    print(f"  Output:         {output_file}")
    print(f"  N folds:        {args.n_folds}")
    print(f"  Physics module: {'available' if HAS_PHYSICS else 'not available (skipping)'}")
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print("\n--- Loading inputs ---")

    if not os.path.exists(shifts_file):
        print(f"ERROR: Shifts file not found: {shifts_file}")
        sys.exit(1)
    if not os.path.exists(pdb_selection_file):
        print(f"ERROR: PDB selection file not found: {pdb_selection_file}")
        sys.exit(1)

    shifts_df = pd.read_csv(shifts_file, dtype={'bmrb_id': str})
    pdb_sel_df = pd.read_csv(pdb_selection_file, dtype={'bmrb_id': str})

    print(f"  Chemical shifts: {shifts_df['bmrb_id'].nunique()} proteins, {len(shifts_df):,} rows")
    print(f"  PDB selection:   {len(pdb_sel_df)} entries")

    # Identify shift and ambiguity columns
    shift_cols = sorted([c for c in shifts_df.columns if c.endswith('_shift')])
    ambig_cols = sorted([c for c in shifts_df.columns if c.endswith('_ambiguity_code')])
    print(f"  Shift columns:   {len(shift_cols)}")
    print(f"  Ambiguity cols:  {len(ambig_cols)}")

    # Build lookup: bmrb_id -> (pdb_path, chain_id)
    pdb_lookup = {}
    for _, row in pdb_sel_df.iterrows():
        bid = str(row['bmrb_id'])
        pdb_file = row.get('pdb_file', row.get('pdb_id', ''))
        chain = row.get('chain_id', row.get('chain', None))
        if pd.isna(chain):
            chain = None

        # Resolve PDB path
        if os.path.isabs(str(pdb_file)):
            pdb_path = str(pdb_file)
        else:
            pdb_path = os.path.join(pdb_dir, str(pdb_file))

        # If the path doesn't include an extension, try common ones
        if not os.path.exists(pdb_path):
            for ext in ['.pdb', '.ent', '.pdb.gz']:
                candidate = pdb_path + ext
                if os.path.exists(candidate):
                    pdb_path = candidate
                    break

        # Fallback: try rcsb_{bmrb_id}.pdb naming convention (DataProduction)
        if not os.path.exists(pdb_path):
            rcsb_candidate = os.path.join(pdb_dir, f'rcsb_{bid}.pdb')
            if os.path.exists(rcsb_candidate):
                pdb_path = rcsb_candidate

        pdb_lookup[bid] = (pdb_path, chain)

    # Only process proteins that appear in both shifts and pdb_selection
    common_ids = sorted(set(shifts_df['bmrb_id'].unique()) & set(pdb_lookup.keys()))
    print(f"\n  Proteins in both shifts and PDB selection: {len(common_ids)}")

    if len(common_ids) == 0:
        print("ERROR: No common proteins found between shifts and PDB selection.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Process each protein
    # ------------------------------------------------------------------
    print("\n--- Processing proteins ---")

    compilation_log = FilterLog()
    compilation_log.add_summary('input', 'Proteins to process', len(common_ids))

    all_rows = []
    failed_proteins = []  # (bmrb_id, error_message)
    succeeded = 0
    skipped_no_pdb = 0

    for bmrb_id in tqdm(common_ids, desc="Compiling proteins"):
        pdb_path, chain_id = pdb_lookup[bmrb_id]

        # Check PDB file exists
        if not os.path.exists(pdb_path):
            failed_proteins.append((bmrb_id, f"PDB file not found: {pdb_path}"))
            compilation_log.add(
                'pdb_missing', f'PDB file not found: {pdb_path}',
                bmrb_id=bmrb_id, action='skipped'
            )
            skipped_no_pdb += 1
            continue

        # Get this protein's shifts
        prot_shifts = shifts_df[shifts_df['bmrb_id'] == bmrb_id]

        try:
            rows = process_protein(bmrb_id, prot_shifts, pdb_path, chain_id, compilation_log)
            if rows:
                all_rows.extend(rows)
                succeeded += 1
            else:
                failed_proteins.append((bmrb_id, "No aligned residues produced"))
                compilation_log.add(
                    'empty_result', 'No rows produced',
                    bmrb_id=bmrb_id, action='skipped'
                )
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"{type(e).__name__}: {e}"
            failed_proteins.append((bmrb_id, error_msg))
            compilation_log.add(
                'processing_error', error_msg,
                bmrb_id=bmrb_id, action='skipped'
            )
            # Print first few failures in detail for debugging
            if len(failed_proteins) <= 5:
                print(f"\n  WARNING: Failed on {bmrb_id}: {error_msg}")
                print(f"  {tb}")

    print(f"\n  Successfully processed: {succeeded} / {len(common_ids)} proteins")
    print(f"  Failed: {len(failed_proteins)} proteins")
    if skipped_no_pdb > 0:
        print(f"  Skipped (no PDB): {skipped_no_pdb}")

    compilation_log.add_summary('compilation', 'Proteins successfully compiled', succeeded)
    compilation_log.add_summary('compilation', 'Proteins failed', len(failed_proteins))

    if not all_rows:
        print("ERROR: No residues compiled. Check inputs and PDB paths.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Build DataFrame
    # ------------------------------------------------------------------
    print("\n--- Building output DataFrame ---")
    df = pd.DataFrame(all_rows)
    print(f"  Raw compiled: {df['bmrb_id'].nunique()} proteins, {len(df):,} residues")

    # ------------------------------------------------------------------
    # 4. Assign fold splits (deterministic by bmrb_id hash)
    # ------------------------------------------------------------------
    print("\n--- Assigning fold splits ---")
    df['split'] = df['bmrb_id'].apply(lambda bid: assign_fold(bid, n_folds=args.n_folds))

    for fold in sorted(df['split'].unique()):
        n_prot = df[df['split'] == fold]['bmrb_id'].nunique()
        n_res = len(df[df['split'] == fold])
        print(f"  Fold {fold}: {n_prot} proteins, {n_res:,} residues")

    # ------------------------------------------------------------------
    # 5. Run data quality pipeline
    # ------------------------------------------------------------------
    print("\n--- Running data quality pipeline ---")
    df, quality_log = run_full_quality_pipeline(df)

    # Report quality
    quality_metrics = report_quality(df, quality_log)

    # ------------------------------------------------------------------
    # 6. Order columns nicely
    # ------------------------------------------------------------------
    base_cols = ['bmrb_id', 'residue_id', 'residue_code', 'split', 'mismatch_type']
    ss_cols = ['secondary_structure', 'phi', 'psi']
    coord_cols = ['ca_x', 'ca_y', 'ca_z']
    bf_cols = ['bfactor']

    # Spatial neighbor columns
    nb_cols = []
    for i in range(K_SPATIAL_NEIGHBORS):
        nb_cols.extend([
            f'spatial_neighbor_{i}_id',
            f'spatial_neighbor_{i}_dist',
            f'spatial_neighbor_{i}_seq_sep',
        ])

    # Distance columns
    dist_col_names = get_distance_column_names() + get_sidechain_summary_names()

    # Physics feature columns
    physics_col_names = get_physics_feature_names() if HAS_PHYSICS else []

    # DSSP h-bond columns
    dssp_hbond_cols = DSSP_COLS  # rel_acc + h-bond columns

    # Build ordered column list
    ordered = (
        base_cols
        + shift_cols
        + ambig_cols
        + ss_cols
        + coord_cols
        + bf_cols
        + nb_cols
        + dist_col_names
        + physics_col_names
        + dssp_hbond_cols
    )
    # Deduplicate while preserving order (some cols like rel_acc appear in DSSP_COLS and might overlap)
    seen = set()
    ordered_deduped = []
    for c in ordered:
        if c not in seen:
            ordered_deduped.append(c)
            seen.add(c)
    ordered = ordered_deduped

    # Only keep columns that exist in the DataFrame
    ordered_present = [c for c in ordered if c in df.columns]
    remaining = [c for c in df.columns if c not in set(ordered_present)]
    df = df[ordered_present + remaining]

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    print("\n--- Saving outputs ---")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Main dataset
    df.to_csv(output_file, index=False)
    print(f"  Saved dataset: {output_file}")
    print(f"    {df['bmrb_id'].nunique()} proteins, {len(df):,} residues, {len(df.columns)} columns")

    # Quality log
    quality_log_path = os.path.join(os.path.dirname(output_file), 'quality_log.csv')
    quality_log.save(quality_log_path)

    # Compilation report
    report_path = os.path.join(os.path.dirname(output_file), 'compilation_report.txt')
    elapsed = time.time() - start_time

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET COMPILATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed time: {elapsed:.1f}s\n")
        f.write("=" * 80 + "\n\n")

        f.write("INPUT FILES\n")
        f.write(f"  Shifts:         {shifts_file}\n")
        f.write(f"  PDB selection:  {pdb_selection_file}\n")
        f.write(f"  PDB directory:  {pdb_dir}\n\n")

        f.write("COMPILATION SUMMARY\n")
        f.write(f"  Total proteins attempted:        {len(common_ids)}\n")
        f.write(f"  Proteins successfully compiled:   {succeeded}\n")
        f.write(f"  Proteins failed:                  {len(failed_proteins)}\n")
        f.write(f"  Skipped (no PDB file):            {skipped_no_pdb}\n\n")

        f.write("OUTPUT SUMMARY\n")
        f.write(f"  Total residues (before quality):  {len(all_rows):,}\n")
        f.write(f"  Total residues (after quality):   {len(df):,}\n")
        f.write(f"  Total proteins (after quality):   {df['bmrb_id'].nunique()}\n")
        f.write(f"  Total columns:                    {len(df.columns)}\n\n")

        f.write("FOLD DISTRIBUTION\n")
        for fold in sorted(df['split'].unique()):
            fold_df = df[df['split'] == fold]
            n_prot = fold_df['bmrb_id'].nunique()
            n_res = len(fold_df)
            f.write(f"  Fold {fold}: {n_prot} proteins, {n_res:,} residues\n")
        f.write("\n")

        f.write("COLUMN TYPES\n")
        f.write(f"  Shift columns:           {len(shift_cols)}\n")
        f.write(f"  Ambiguity columns:       {len(ambig_cols)}\n")
        f.write(f"  Distance columns:        {len([c for c in dist_col_names if c in df.columns])}\n")
        f.write(f"  Sidechain summary cols:  {len([c for c in get_sidechain_summary_names() if c in df.columns])}\n")
        f.write(f"  Physics feature cols:    {len([c for c in physics_col_names if c in df.columns])}\n")
        f.write(f"  Spatial neighbor cols:   {len([c for c in nb_cols if c in df.columns])}\n")
        f.write(f"  DSSP columns:            {len([c for c in DSSP_COLS if c in df.columns]) + len([c for c in ss_cols if c in df.columns])}\n")
        f.write("\n")

        # Shift coverage
        f.write("SHIFT COVERAGE\n")
        for col in shift_cols:
            if col in df.columns:
                n_valid = df[col].notna().sum()
                pct = 100.0 * n_valid / len(df) if len(df) > 0 else 0
                f.write(f"  {col:20s}: {n_valid:>8,} ({pct:5.1f}%)\n")
        f.write("\n")

        # Quality filtering summary
        f.write("DATA QUALITY FILTERING\n")
        for s in quality_log.summaries:
            f.write(f"  {s['step']:30s} | {s['reason']:45s} | {s['count']:>8,}\n")
            if s['details']:
                f.write(f"  {'':30s} | {s['details']}\n")
        f.write("\n")

        # Failed proteins
        if failed_proteins:
            f.write(f"FAILED PROTEINS ({len(failed_proteins)})\n")
            for bid, reason in sorted(failed_proteins):
                f.write(f"  {bid}: {reason}\n")
        else:
            f.write("FAILED PROTEINS: None\n")
        f.write("\n")

        f.write("OUTPUT FILES\n")
        f.write(f"  Dataset:         {output_file}\n")
        f.write(f"  Quality log:     {quality_log_path}\n")
        quality_summary_path = quality_log_path.replace('.csv', '_summary.csv')
        f.write(f"  Quality summary: {quality_summary_path}\n")
        f.write(f"  This report:     {report_path}\n")

    print(f"  Saved report:   {report_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("COMPILATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Proteins:  {df['bmrb_id'].nunique()}")
    print(f"  Residues:  {len(df):,}")
    print(f"  Columns:   {len(df.columns)}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Output:    {output_file}")


if __name__ == '__main__':
    main()
