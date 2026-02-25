#!/usr/bin/env python3
"""
Step 01: Build training datasets from chemical shifts + structures.

Replaces the old 01_select_pdb_structures.py + 02_compile_dataset.py.

Pipeline:
  1. Find PDB structures for each protein (pairs.csv + BMRB API + BLAST fallback)
  2. Select best chain via RMSD-based analysis
  3. Get AlphaFold structures (BMRB -> UniProt -> AlphaFold DB)
  4. Compile features for 3 datasets
  5. Quality filtering (nonstandard discard, nucleotide removal, outlier detection)

Outputs:
  data/structure_data_experimental.csv   -- proteins with experimental PDB structures
  data/structure_data_hybrid.csv         -- experimental where available, AlphaFold otherwise
  data/structure_data_alphafold.csv      -- all AlphaFold structures
  data/build_log.csv                     -- provenance log
"""

import argparse
import os
import sys
import hashlib
import time
import traceback
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_DIR, PDB_DIR, DSSP_COLS, AA_3_TO_1, NUCLEOTIDES,
    K_SPATIAL_NEIGHBORS, STANDARD_RESIDUES, OUTLIER_STD_THRESHOLD,
    ALPHAFOLD_DIR,
)
from pdb_utils import (
    parse_pdb, run_dssp, resolve_pdb_path, lookup_with_chain_fallback, classify_atom,
)
from alignment import align_sequences, to_single_letter
from distance_features import compute_all_distance_features, get_sidechain_summary_names
from spatial_neighbors import find_neighbors
from physics_features import compute_all_physics_features, get_physics_feature_names
from rcsb_search import search_pdb_by_sequence, extract_sequence_from_shifts
from structure_selection import select_best_chain
from alphafold_utils import get_uniprot_for_bmrb, download_alphafold_structure

# Reuse API helpers from old 01_select_pdb_structures if available
from importlib import util as importlib_util
_old_01_spec = importlib_util.find_spec('01_select_pdb_structures')
if _old_01_spec is None:
    # Define minimal versions of what we need
    import json
    import urllib.request

    def lookup_pdb_ids_for_bmrb(bmrb_id):
        url = f'https://api.bmrb.io/v2/search/get_pdb_ids_from_bmrb_id/{bmrb_id}'
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            req.add_header('User-Agent', 'he_lab_pipeline/1.0')
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            if isinstance(data, list):
                return [str(p).upper().strip() for p in data if p]
            if isinstance(data, dict):
                for key in ('data', 'pdb_ids', 'result'):
                    if key in data and isinstance(data[key], list):
                        return [str(p).upper().strip() for p in data[key] if p]
            return []
        except Exception:
            return []

    def download_pdb_file(pdb_id, output_path, retries=3):
        url = f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb'
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'he_lab_pipeline/1.0')
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(output_path, 'wb') as f:
                        f.write(resp.read())
                return True
            except Exception:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return False

    def load_pairs_file(pairs_path):
        if not os.path.exists(pairs_path):
            return {}
        pairs_df = pd.read_csv(pairs_path, dtype={'Entry_ID': str})
        mapping = {}
        for _, row in pairs_df.iterrows():
            bmrb_id = str(row['Entry_ID'])
            pdb_ids_str = str(row.get('pdb_ids', ''))
            if pd.isna(row.get('pdb_ids')) or pdb_ids_str == 'nan':
                continue
            pdb_list = [p.strip().upper() for p in pdb_ids_str.split(',') if p.strip()]
            if pdb_list:
                mapping[bmrb_id] = pdb_list
        return mapping
else:
    from importlib import import_module as _imp
    _old_01 = _imp('01_select_pdb_structures')
    lookup_pdb_ids_for_bmrb = _old_01.lookup_pdb_ids_for_bmrb
    download_pdb_file = _old_01.download_pdb_file
    load_pairs_file = _old_01.load_pairs_file


# ============================================================================
# Helpers
# ============================================================================

def assign_fold(bmrb_id, n_folds=5):
    """Deterministic fold assignment using MD5 hash of bmrb_id."""
    return int(hashlib.md5(str(bmrb_id).encode()).hexdigest(), 16) % n_folds + 1


def get_structure_sequence(residues):
    """Extract sequence from parsed PDB residues dict."""
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
    """Extract sequence from a single protein's shift DataFrame."""
    sub = shifts_df[['residue_id', 'residue_code']].drop_duplicates().sort_values('residue_id')
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
    """Convert alignment to residue-level mapping with mismatch classification."""
    a1, a2 = str(alignment[0]), str(alignment[1])
    i1, i2 = 0, 0
    n_struct = len(struct_seq['residue_ids'])
    n_shift = len(shift_seq['residue_ids'])
    mappings = []

    for c1, c2 in zip(a1, a2):
        if c1 != '-' and c2 != '-':
            struct_rid = struct_seq['residue_ids'][i1]
            shift_rid = shift_seq['residue_ids'][i2]

            if c1 == c2:
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
        if c1 != '-':
            i1 += 1
        if c2 != '-':
            i2 += 1

    return mappings


def has_nonstandard_residues(shifts_df):
    """Check if any residue in the protein is nonstandard.

    Returns True if ANY residue is not in the standard 20 amino acids.
    """
    standard_set = set(STANDARD_RESIDUES) - {'UNK'}
    codes = shifts_df['residue_code'].str.upper().unique()
    for code in codes:
        if code not in standard_set:
            return True
    return False


def has_nucleotides(residues):
    """Check if parsed PDB residues contain nucleotides."""
    for (chain, res_id), rdata in residues.items():
        if rdata['residue_name'] in NUCLEOTIDES:
            return True
    return False


# ============================================================================
# Step 1: Find PDB structures
# ============================================================================

def find_pdb_structures(bmrb_ids, shifts_df, pairs_mapping, pdb_dir,
                        search_dirs, online=False):
    """Find candidate PDB structures for each protein.

    Sources:
      1. pairs.csv local mapping
      2. BMRB API lookup (if online)
      3. BLAST via RCSB sequence search (if online, as fallback)

    Returns:
        dict: bmrb_id -> list of (pdb_path, [chain_ids])
    """
    candidates = {}
    counters = defaultdict(int)

    for bmrb_id in tqdm(bmrb_ids, desc="Finding PDB structures"):
        pdb_ids = set()

        # Source 1: pairs.csv
        if bmrb_id in pairs_mapping:
            for pid in pairs_mapping[bmrb_id]:
                pdb_ids.add(pid.upper())

        # Source 2: BMRB API
        if online and not pdb_ids:
            api_pdbs = lookup_pdb_ids_for_bmrb(bmrb_id)
            for pid in api_pdbs:
                pdb_ids.add(pid.upper())

        # Source 3: BLAST fallback
        if online and not pdb_ids:
            prot_shifts = shifts_df[shifts_df['bmrb_id'] == bmrb_id]
            seq = extract_sequence_from_shifts(prot_shifts)
            if seq:
                hits = search_pdb_by_sequence(seq)
                for hit in hits:
                    pdb_ids.add(hit['pdb_id'].upper())
                if hits:
                    counters['blast_found'] += 1

        if not pdb_ids:
            counters['no_mapping'] += 1
            continue

        # Resolve paths and download if needed
        protein_candidates = []
        for pdb_id in sorted(pdb_ids):
            pdb_path = resolve_pdb_path(pdb_id, search_dirs)
            if pdb_path is None:
                # Try downloading
                pdb_path = os.path.join(pdb_dir, f'{pdb_id}.pdb')
                if not os.path.exists(pdb_path):
                    if online:
                        success = download_pdb_file(pdb_id, pdb_path)
                        if not success:
                            continue
                        counters['downloaded'] += 1
                    else:
                        continue

            # Parse to find available chains
            try:
                residues = parse_pdb(pdb_path)
                chains = sorted(set(ch for ch, _ in residues.keys()))
                if chains:
                    protein_candidates.append((pdb_path, chains))
            except Exception:
                continue

        if protein_candidates:
            candidates[bmrb_id] = protein_candidates
            counters['found'] += 1
        else:
            counters['no_valid_pdb'] += 1

    print(f"\n  PDB structure search results:")
    print(f"    Found structures: {counters['found']:,}")
    print(f"    No mapping:       {counters['no_mapping']:,}")
    print(f"    No valid PDB:     {counters['no_valid_pdb']:,}")
    if online:
        print(f"    BLAST found:      {counters['blast_found']:,}")
        print(f"    Downloaded:       {counters['downloaded']:,}")

    return candidates


# ============================================================================
# Step 2: Select best chain via RMSD
# ============================================================================

def select_structures(candidates, shifts_df):
    """For each protein, select the best chain using RMSD analysis.

    Args:
        candidates: dict bmrb_id -> list of (pdb_path, [chain_ids])
        shifts_df: full shifts DataFrame

    Returns:
        dict: bmrb_id -> (pdb_path, chain_id)
    """
    selections = {}

    for bmrb_id in tqdm(sorted(candidates.keys()), desc="Selecting best chains"):
        prot_shifts = shifts_df[shifts_df['bmrb_id'] == bmrb_id]
        shift_seq = get_shift_sequence(prot_shifts)
        if shift_seq is None:
            continue

        # Build flat list of (pdb_path, chain_id) candidates
        flat_candidates = []
        for pdb_path, chain_ids in candidates[bmrb_id]:
            for chain_id in chain_ids:
                flat_candidates.append((pdb_path, chain_id))

        if not flat_candidates:
            continue

        if len(flat_candidates) == 1:
            selections[bmrb_id] = flat_candidates[0]
            continue

        try:
            best_path, best_chain, _ = select_best_chain(
                flat_candidates, shift_seq['sequence']
            )
            if best_path is not None:
                selections[bmrb_id] = (best_path, best_chain)
            else:
                # Fallback to first candidate
                selections[bmrb_id] = flat_candidates[0]
        except Exception:
            selections[bmrb_id] = flat_candidates[0]

    print(f"  Selected structures for {len(selections):,} proteins")
    return selections


# ============================================================================
# Step 3: Get AlphaFold structures
# ============================================================================

def get_alphafold_structures(bmrb_ids, shifts_df, alphafold_dir, online=False):
    """Download AlphaFold structures for each protein.

    Args:
        bmrb_ids: list of BMRB IDs
        shifts_df: shifts DataFrame
        alphafold_dir: output directory for AlphaFold PDBs
        online: whether to make API calls

    Returns:
        dict: bmrb_id -> (af_pdb_path, chain_id)
    """
    if not online:
        # Check for existing AlphaFold files
        af_structures = {}
        if os.path.isdir(alphafold_dir):
            for bmrb_id in bmrb_ids:
                # Try to find an existing AlphaFold file
                for fname in os.listdir(alphafold_dir):
                    if fname.endswith('.pdb') and fname.startswith('AF-'):
                        # Can't map without UniProt info in offline mode
                        pass
        return af_structures

    os.makedirs(alphafold_dir, exist_ok=True)
    af_structures = {}
    counters = defaultdict(int)

    for bmrb_id in tqdm(bmrb_ids, desc="Fetching AlphaFold structures"):
        # Step 1: Map BMRB -> UniProt
        uniprot_id = get_uniprot_for_bmrb(bmrb_id)

        # Fallback: sequence search
        if uniprot_id is None:
            prot_shifts = shifts_df[shifts_df['bmrb_id'] == bmrb_id]
            seq = extract_sequence_from_shifts(prot_shifts)
            if seq and len(seq) >= 20:
                from alphafold_utils import search_uniprot_by_sequence
                uniprot_id = search_uniprot_by_sequence(seq)
                if uniprot_id:
                    counters['uniprot_via_sequence'] += 1

        if uniprot_id is None:
            counters['no_uniprot'] += 1
            continue

        # Step 2: Download AlphaFold structure
        af_path = download_alphafold_structure(uniprot_id, alphafold_dir)
        if af_path is None:
            counters['af_not_available'] += 1
            continue

        af_structures[bmrb_id] = (af_path, 'A')  # AlphaFold always uses chain A
        counters['af_downloaded'] += 1

    print(f"\n  AlphaFold structure results:")
    print(f"    Downloaded:          {counters['af_downloaded']:,}")
    print(f"    No UniProt mapping:  {counters['no_uniprot']:,}")
    print(f"    AF not available:    {counters['af_not_available']:,}")
    if counters['uniprot_via_sequence']:
        print(f"    UniProt via seq:     {counters['uniprot_via_sequence']:,}")

    return af_structures


# ============================================================================
# Step 4: Compile features for one protein
# ============================================================================

def process_protein(bmrb_id, shifts_df, pdb_path, chain_id):
    """Process a single protein: parse PDB, align, compute features.

    Returns:
        list of row dicts (one per aligned residue), or empty list on failure
    """
    rows = []

    # Parse PDB
    residues = parse_pdb(pdb_path, chain_id=chain_id)
    if not residues:
        return rows

    # Check for nucleotides
    if has_nucleotides(residues):
        return rows

    # Run DSSP
    dssp_data = run_dssp(pdb_path)

    # Build structure data: res_id -> {residue_name, atoms}
    struct_data = {}
    chain_used = None
    for (chain, res_id), rdata in residues.items():
        if chain_used is None:
            chain_used = chain
        struct_data[res_id] = rdata

    # Get sequences
    struct_seq = get_structure_sequence(residues)
    shift_seq = get_shift_sequence(shifts_df)

    if struct_seq is None or shift_seq is None:
        return rows

    # Align
    alignment = align_sequences(struct_seq['sequence'], shift_seq['sequence'])
    if alignment is None:
        return rows

    mapping = alignment_to_mapping(alignment, struct_seq, shift_seq)
    if not mapping:
        return rows

    # Spatial neighbors
    neighbors = find_neighbors(struct_data)

    # Index shift data
    shift_indexed = shifts_df.drop_duplicates(subset='residue_id').set_index('residue_id')
    shift_cols = sorted([c for c in shifts_df.columns if c.endswith('_shift')])
    ambig_cols = sorted([c for c in shifts_df.columns if c.endswith('_ambiguity_code')])

    # Pre-compute chain-wide coords for physics features
    sorted_rids = sorted(struct_data.keys())
    rid_to_idx = {}
    all_residue_coords = {}  # rid -> atom coords dict for h-bond distance calc
    for idx, rid in enumerate(sorted_rids):
        rdata = struct_data[rid]
        atoms = rdata.get('atoms', {})
        rid_to_idx[rid] = idx
        all_residue_coords[idx] = atoms

    # Build rows
    for m in mapping:
        struct_rid = m['struct_res_id']
        shift_rid = m['shift_res_id']
        mismatch_type = m['mismatch_type']

        if shift_rid not in shift_indexed.index:
            continue

        shift_row = shift_indexed.loc[shift_rid]
        if isinstance(shift_row, pd.DataFrame):
            shift_row = shift_row.iloc[0]

        residue_code = shift_row.get('residue_code', 'UNK')
        if pd.isna(residue_code):
            residue_code = 'UNK'

        row = {
            'bmrb_id': str(bmrb_id),
            'residue_id': struct_rid,
            'residue_code': residue_code,
            'mismatch_type': mismatch_type,
        }

        # Chemical shifts
        for col in shift_cols + ambig_cols:
            val = shift_row.get(col)
            if val is not None and not pd.isna(val):
                row[col] = val

        # Structural features
        if struct_rid in struct_data:
            rdata = struct_data[struct_rid]
            atoms = rdata.get('atoms', {})

            # CA coordinates
            if 'CA' in atoms and np.all(np.isfinite(atoms['CA'])):
                row['ca_x'], row['ca_y'], row['ca_z'] = atoms['CA'].tolist()

            # Distance features (all heavy + H-to-heavy)
            dist_feats = compute_all_distance_features(atoms)
            row.update(dist_feats)

        # DSSP data
        dssp_entry = lookup_with_chain_fallback(dssp_data, chain_used or '', struct_rid)
        if dssp_entry is not None:
            row['secondary_structure'] = dssp_entry.get('secondary_structure', 'C')
            phi_val = dssp_entry.get('phi')
            psi_val = dssp_entry.get('psi')
            row['phi'] = phi_val if phi_val is not None else np.nan
            row['psi'] = psi_val if psi_val is not None else np.nan

            for col in DSSP_COLS:
                val = dssp_entry.get(col)
                if val is not None:
                    row[col] = val

            # Physics features (h-bond geometry only)
            dssp_for_phys = dict(dssp_entry)
            dssp_for_phys['residue_idx'] = rid_to_idx.get(struct_rid, 0)
            phys_feats = compute_all_physics_features(
                dssp_data=dssp_for_phys,
                all_coords=all_residue_coords,
            )
            row.update(phys_feats)

        # Spatial neighbors
        if struct_rid in neighbors:
            nb = neighbors[struct_rid]
            for i in range(K_SPATIAL_NEIGHBORS):
                row[f'spatial_neighbor_{i}_id'] = nb['ids'][i]
                row[f'spatial_neighbor_{i}_dist'] = nb['dists'][i]
                row[f'spatial_neighbor_{i}_seq_sep'] = nb['seps'][i]

        rows.append(row)

    return rows


# ============================================================================
# Step 5: Quality filtering
# ============================================================================

def filter_nonstandard_proteins(df):
    """Remove entire proteins that contain any nonstandard residue."""
    standard_set = set(STANDARD_RESIDUES) - {'UNK'}
    proteins_to_remove = set()

    for bmrb_id, group in df.groupby('bmrb_id'):
        codes = group['residue_code'].str.upper().unique()
        for code in codes:
            if code not in standard_set:
                proteins_to_remove.add(bmrb_id)
                break

    if proteins_to_remove:
        n_before = len(df)
        df = df[~df['bmrb_id'].isin(proteins_to_remove)].copy()
        print(f"    Removed {len(proteins_to_remove)} proteins with nonstandard residues "
              f"({n_before - len(df):,} residues)")

    return df


def filter_outliers_by_group(df, sd_threshold=None):
    """Set outlier shift values to NaN per (atom_type, secondary_structure) group.

    For each group, computes mean and std, then sets values > threshold SDs
    from the mean to NaN.
    """
    if sd_threshold is None:
        sd_threshold = OUTLIER_STD_THRESHOLD

    shift_cols = [c for c in df.columns if c.endswith('_shift')]
    ss_col = 'secondary_structure'

    if ss_col not in df.columns:
        # Fallback: no grouping by SS
        for col in shift_cols:
            valid = df[col].dropna()
            if len(valid) < 10:
                continue
            mean = valid.mean()
            std = valid.std()
            if std < 1e-6:
                continue
            outlier_mask = df[col].notna() & ((df[col] - mean).abs() > sd_threshold * std)
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                df.loc[outlier_mask, col] = np.nan
        return df

    total_outliers = 0

    for col in shift_cols:
        for ss_type, group in df.groupby(ss_col):
            valid = group[col].dropna()
            if len(valid) < 10:
                continue
            mean = valid.mean()
            std = valid.std()
            if std < 1e-6:
                continue
            group_idx = group.index
            outlier_mask = df.loc[group_idx, col].notna() & (
                (df.loc[group_idx, col] - mean).abs() > sd_threshold * std
            )
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                df.loc[group_idx[outlier_mask], col] = np.nan
                total_outliers += n_outliers

    print(f"    Set {total_outliers:,} outlier shift values to NaN "
          f"(>{sd_threshold} SD per atom_type x secondary_structure)")

    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build training datasets from chemical shifts + structures.'
    )
    parser.add_argument(
        '--shifts-file', default=os.path.join(DATA_DIR, 'chemical_shifts.csv'),
        help='Path to chemical shifts CSV'
    )
    parser.add_argument(
        '--pairs-file', default='data/pairs.csv',
        help='Path to BMRB->PDB pairs.csv'
    )
    parser.add_argument(
        '--pdb-dir', default=PDB_DIR,
        help='Directory for PDB files'
    )
    parser.add_argument(
        '--alphafold-dir', default=ALPHAFOLD_DIR,
        help='Directory for AlphaFold PDB files'
    )
    parser.add_argument(
        '--output-dir', default=DATA_DIR,
        help='Output directory'
    )
    parser.add_argument(
        '--existing-pdbs', nargs='+', default=['data/pdbs'],
        help='Directories to search for existing PDB files'
    )
    parser.add_argument(
        '--online', action='store_true', default=False,
        help='Enable API calls (BMRB, RCSB, AlphaFold). Default is offline.'
    )
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--max-proteins', type=int, default=None,
        help='Limit to N proteins (for testing)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 01: BUILD TRAINING DATASETS")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(path):
        return os.path.join(script_dir, path) if not os.path.isabs(path) else path

    shifts_file = resolve(args.shifts_file)
    pairs_file = resolve(args.pairs_file)
    pdb_dir = resolve(args.pdb_dir)
    alphafold_dir = resolve(args.alphafold_dir)
    output_dir = resolve(args.output_dir)
    search_dirs = [resolve(d) for d in args.existing_pdbs] + [pdb_dir]

    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print("\n--- Loading inputs ---")

    if not os.path.exists(shifts_file):
        print(f"ERROR: Shifts file not found: {shifts_file}")
        print("  Run 00_fetch_bmrb_shifts.py first.")
        sys.exit(1)

    shifts_df = pd.read_csv(shifts_file, dtype={'bmrb_id': str})
    bmrb_ids = sorted(shifts_df['bmrb_id'].unique())

    if args.max_proteins:
        bmrb_ids = bmrb_ids[:args.max_proteins]

    print(f"  Proteins: {len(bmrb_ids):,}")
    print(f"  Residues: {len(shifts_df):,}")

    # Pre-filter: discard proteins with nonstandard residues
    print("\n--- Pre-filtering: nonstandard residues ---")
    valid_bmrb_ids = []
    n_nonstandard = 0
    standard_set = set(STANDARD_RESIDUES) - {'UNK'}
    for bmrb_id in bmrb_ids:
        prot_shifts = shifts_df[shifts_df['bmrb_id'] == bmrb_id]
        if has_nonstandard_residues(prot_shifts):
            n_nonstandard += 1
        else:
            valid_bmrb_ids.append(bmrb_id)
    print(f"  Discarded {n_nonstandard} proteins with nonstandard residues")
    print(f"  Remaining: {len(valid_bmrb_ids):,}")
    bmrb_ids = valid_bmrb_ids

    # Load pairs mapping
    pairs_mapping = load_pairs_file(pairs_file) if os.path.exists(pairs_file) else {}
    if pairs_mapping:
        print(f"  Loaded {len(pairs_mapping):,} BMRB->PDB mappings from pairs.csv")

    # ------------------------------------------------------------------
    # Step 1: Find PDB structures
    # ------------------------------------------------------------------
    print("\n--- Step 1: Finding PDB structures ---")
    pdb_candidates = find_pdb_structures(
        bmrb_ids, shifts_df, pairs_mapping, pdb_dir,
        search_dirs, online=args.online,
    )

    # ------------------------------------------------------------------
    # Step 2: Select best chains via RMSD
    # ------------------------------------------------------------------
    print("\n--- Step 2: Selecting best chains (RMSD) ---")
    experimental_selections = select_structures(pdb_candidates, shifts_df)

    # ------------------------------------------------------------------
    # Step 3: Get AlphaFold structures
    # ------------------------------------------------------------------
    print("\n--- Step 3: AlphaFold structures ---")
    af_selections = get_alphafold_structures(
        bmrb_ids, shifts_df, alphafold_dir, online=args.online,
    )

    # ------------------------------------------------------------------
    # Step 4: Compile features for each dataset
    # ------------------------------------------------------------------
    print("\n--- Step 4: Compiling features ---")

    # Build dataset configs
    # Dataset 1: experimental only
    # Dataset 2: hybrid (experimental where available, AlphaFold otherwise)
    # Dataset 3: AlphaFold only

    dataset_configs = {
        'experimental': {},
        'hybrid': {},
        'alphafold': {},
    }

    for bmrb_id in bmrb_ids:
        if bmrb_id in experimental_selections:
            dataset_configs['experimental'][bmrb_id] = experimental_selections[bmrb_id]
            dataset_configs['hybrid'][bmrb_id] = experimental_selections[bmrb_id]
        elif bmrb_id in af_selections:
            dataset_configs['hybrid'][bmrb_id] = af_selections[bmrb_id]

        if bmrb_id in af_selections:
            dataset_configs['alphafold'][bmrb_id] = af_selections[bmrb_id]

    print(f"  Experimental: {len(dataset_configs['experimental']):,} proteins")
    print(f"  Hybrid:       {len(dataset_configs['hybrid']):,} proteins")
    print(f"  AlphaFold:    {len(dataset_configs['alphafold']):,} proteins")

    # Build log rows
    log_rows = []

    for dataset_name, selections in dataset_configs.items():
        print(f"\n  Compiling {dataset_name} dataset...")

        all_rows = []
        failed = 0
        succeeded = 0

        for bmrb_id in tqdm(sorted(selections.keys()),
                            desc=f"  {dataset_name}", unit="protein"):
            pdb_path, chain_id = selections[bmrb_id]
            prot_shifts = shifts_df[shifts_df['bmrb_id'] == bmrb_id]

            try:
                rows = process_protein(bmrb_id, prot_shifts, pdb_path, chain_id)
                if rows:
                    all_rows.extend(rows)
                    succeeded += 1
                    log_rows.append({
                        'dataset': dataset_name,
                        'bmrb_id': bmrb_id,
                        'pdb_path': pdb_path,
                        'chain_id': chain_id,
                        'n_residues': len(rows),
                        'status': 'success',
                    })
                else:
                    failed += 1
                    log_rows.append({
                        'dataset': dataset_name,
                        'bmrb_id': bmrb_id,
                        'pdb_path': pdb_path,
                        'chain_id': chain_id,
                        'n_residues': 0,
                        'status': 'empty',
                    })
            except Exception as e:
                failed += 1
                log_rows.append({
                    'dataset': dataset_name,
                    'bmrb_id': bmrb_id,
                    'pdb_path': pdb_path,
                    'chain_id': chain_id,
                    'n_residues': 0,
                    'status': f'error: {e}',
                })
                if failed <= 3:
                    print(f"\n    WARNING: {bmrb_id}: {e}")
                    traceback.print_exc()

        if not all_rows:
            print(f"    No residues compiled for {dataset_name}")
            continue

        df = pd.DataFrame(all_rows)
        print(f"    Raw: {df['bmrb_id'].nunique()} proteins, {len(df):,} residues "
              f"({succeeded} ok, {failed} failed)")

        # Assign folds
        df['split'] = df['bmrb_id'].apply(lambda bid: assign_fold(bid, n_folds=args.n_folds))

        # ------------------------------------------------------------------
        # Step 5: Quality filtering
        # ------------------------------------------------------------------
        print(f"    Quality filtering...")

        # 5a: Remove proteins with nonstandard residues (should already be filtered, but double-check)
        df = filter_nonstandard_proteins(df)

        # 5b: Outlier detection per (atom_type, secondary_structure)
        df = filter_outliers_by_group(df)

        # Save
        output_path = os.path.join(output_dir, f'structure_data_{dataset_name}.csv')
        df.to_csv(output_path, index=False)
        print(f"    Saved: {output_path}")
        print(f"    Final: {df['bmrb_id'].nunique()} proteins, {len(df):,} residues, "
              f"{len(df.columns)} columns")

    # Save build log
    log_path = os.path.join(output_dir, 'build_log.csv')
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"\n  Build log: {log_path}")

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("BUILD COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Time: {elapsed:.1f}s")

    for dataset_name in ['experimental', 'hybrid', 'alphafold']:
        path = os.path.join(output_dir, f'structure_data_{dataset_name}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=['bmrb_id'], dtype={'bmrb_id': str})
            print(f"  {dataset_name:15s}: {df['bmrb_id'].nunique():,} proteins")

    print()


if __name__ == '__main__':
    main()
