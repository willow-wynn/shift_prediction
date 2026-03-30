#!/usr/bin/env python3
"""
Step 01: Build training datasets from chemical shifts + structures.

Replaces the old 01_select_pdb_structures.py + 02_compile_dataset.py.

Pipeline:
  1. Find PDB structures for each protein (pairs.csv + BMRB API)
  2. Select best chain via alignment identity
  3. Get AlphaFold structures (BMRB -> UniProt -> AlphaFold DB)
  4. For NMR structures: select median-representative model
  5. Compile features for 3 datasets
  6. Quality filtering (nonstandard discard, nucleotide removal, outlier detection)

Outputs:
  data/structure_data_hybrid.csv         -- experimental where available, AlphaFold otherwise (all BMRB + AlphaFold)
  data/structure_data_experimental.csv   -- experimental PDB structures only
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_DIR, PDB_DIR, DSSP_COLS, AA_3_TO_1, NUCLEOTIDES,
    K_SPATIAL_NEIGHBORS, STANDARD_RESIDUES, OUTLIER_STD_THRESHOLD,
    ALPHAFOLD_DIR, MIN_SEQUENCE_IDENTITY,
)
from pdb_utils import (
    parse_pdb, parse_pdb_all_models, run_dssp, resolve_pdb_path,
    lookup_with_chain_fallback, classify_atom,
)
from alignment import align_sequences, to_single_letter
from distance_features import compute_all_distance_features, get_sidechain_summary_names
from spatial_neighbors import find_neighbors
from structure_selection import select_best_chain, kabsch_superimpose
from alphafold_utils import get_uniprot_for_bmrb, download_alphafold_structure

import csv
import io
import json
import urllib.request

# ============================================================================
# BMRB->PDB pairs download (always available, regardless of old module)
# ============================================================================

BMRB_PAIRS_URL = (
    'https://bmrb.io/search/query_grid/index.php'
    '?output=csv&data_types[]=pdb_ids&polymer_join_type=AND'
)


def download_pairs_file(output_path, retries=3):
    """Download BMRB->PDB mapping from the BMRB query grid API.

    The raw CSV has many columns; we extract just Entry_ID and pdb_ids.
    """
    print(f"  Downloading BMRB->PDB mapping from BMRB...")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(BMRB_PAIRS_URL)
            req.add_header('User-Agent', 'he_lab_pipeline/1.0')
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode('utf-8')
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"    ERROR: Failed to download pairs after {retries} attempts: {e}")
                return False

    # Parse CSV and extract Entry_ID, pdb_ids
    # BMRB response may have leading blank lines; strip them
    lines = raw.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    raw = '\n'.join(lines)

    reader = csv.DictReader(io.StringIO(raw))
    # Clean column names (BMRB adds whitespace/quotes)
    fieldnames = [f.strip().strip('"') for f in (reader.fieldnames or [])]
    reader.fieldnames = fieldnames

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_written = 0
    with open(output_path, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Entry_ID', 'pdb_ids'])
        for row in reader:
            entry_id = row.get('Entry_ID', '').strip()
            pdb_ids = row.get('pdb_ids', '').strip()
            if entry_id and pdb_ids:
                writer.writerow([entry_id, pdb_ids])
                n_written += 1

    print(f"  Saved {n_written:,} BMRB->PDB mappings to {output_path}")
    return True


def load_pairs_file(pairs_path):
    """Load BMRB->PDB mapping from pairs.csv."""
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


def lookup_pdb_ids_for_bmrb(bmrb_id):
    """Look up PDB IDs for a BMRB entry via REST API."""
    url = f'https://api.bmrb.io/v2/search/get_pdb_ids_from_bmrb_id/{bmrb_id}'
    try:
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/json')
        req.add_header('User-Agent', 'he_lab_pipeline/1.0')
        with urllib.request.urlopen(req, timeout=15) as resp:
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
    """Download a PDB file from RCSB."""
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
        elif c1 != '-' and c2 == '-':
            # Structure residue has no shift counterpart
            struct_rid = struct_seq['residue_ids'][i1]
            mappings.append({
                'struct_res_id': struct_rid,
                'shift_res_id': np.nan,
                'mismatch_type': 'gap_in_cs',
            })
        elif c1 == '-' and c2 != '-':
            # Shift residue has no structure counterpart
            shift_rid = shift_seq['residue_ids'][i2]
            mappings.append({
                'struct_res_id': np.nan,
                'shift_res_id': shift_rid,
                'mismatch_type': 'gap_in_structure',
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

def _get_chains_fast(pdb_path):
    """Extract chain IDs from a PDB file without full coordinate parsing.

    ~100x faster than parse_pdb since it only reads chain letters from ATOM
    records, skipping all coordinate parsing and numpy array construction.
    """
    chains = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            rec = line[:6]
            if rec.startswith('ENDMDL'):
                break
            if rec.startswith('ATOM') or rec.startswith('HETATM'):
                ch = line[21].strip()
                if ch:
                    chains.add(ch)
    return sorted(chains)


def _find_pdb_for_one_protein(bmrb_id, pdb_ids, pdb_dir, search_dirs, online):
    """Worker: resolve paths and extract chain IDs for one protein's PDB IDs.

    Returns:
        (bmrb_id, protein_candidates, local_counters)
    """
    local_counters = defaultdict(int)
    protein_candidates = []

    for pdb_id in sorted(pdb_ids):
        pdb_path = resolve_pdb_path(pdb_id, search_dirs)
        if pdb_path is None:
            pdb_path = os.path.join(pdb_dir, f'{pdb_id}.pdb')
            if not os.path.exists(pdb_path):
                if online:
                    success = download_pdb_file(pdb_id, pdb_path)
                    if not success:
                        continue
                    local_counters['downloaded'] += 1
                else:
                    continue

        try:
            chains = _get_chains_fast(pdb_path)
            if chains:
                protein_candidates.append((pdb_path, chains))
        except Exception:
            continue

    if protein_candidates:
        local_counters['found'] += 1
    else:
        local_counters['no_valid_pdb'] += 1

    return bmrb_id, protein_candidates, local_counters


def find_pdb_structures(bmrb_ids, shifts_df, pairs_mapping, pdb_dir,
                        search_dirs, online=False, max_workers=8):
    """Find candidate PDB structures for each protein.

    Sources:
      1. pairs.csv local mapping
      2. BMRB API lookup (if online)

    Proteins with no PDB mapping go straight to AlphaFold-only.

    Returns:
        dict: bmrb_id -> list of (pdb_path, [chain_ids])
    """
    # Phase 1: gather PDB IDs per protein
    bmrb_to_pdb_ids = {}
    counters = defaultdict(int)

    # 1a: instant dict lookups from pairs.csv
    need_api = []
    for bmrb_id in bmrb_ids:
        if bmrb_id in pairs_mapping:
            pdb_ids = {pid.upper() for pid in pairs_mapping[bmrb_id]}
            bmrb_to_pdb_ids[bmrb_id] = pdb_ids
        else:
            need_api.append(bmrb_id)

    print(f"  {len(bmrb_to_pdb_ids):,} from pairs.csv, "
          f"{len(need_api):,} need API lookup")

    # 1b: parallel BMRB API lookups for the rest
    if online and need_api:
        def _api_lookup(bmrb_id):
            return bmrb_id, lookup_pdb_ids_for_bmrb(bmrb_id)

        with ThreadPoolExecutor(max_workers=32) as api_pool:
            futures = {api_pool.submit(_api_lookup, bid): bid for bid in need_api}
            need_blast = []
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="BMRB API lookups"):
                bmrb_id, api_pdbs = future.result()
                if api_pdbs:
                    bmrb_to_pdb_ids[bmrb_id] = {p.upper() for p in api_pdbs}
                else:
                    need_blast.append(bmrb_id)

    counters['no_mapping'] = len(bmrb_ids) - len(bmrb_to_pdb_ids)
    print(f"  {len(bmrb_to_pdb_ids):,} proteins have PDB IDs, "
          f"{counters['no_mapping']:,} have no mapping")

    # Phase 2: resolve paths + extract chain IDs (parallel)
    # ProcessPoolExecutor bypasses the GIL for CPU-bound file parsing;
    # ThreadPoolExecutor is better when downloads dominate (online mode).
    candidates = {}
    Executor = ThreadPoolExecutor if online else ProcessPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _find_pdb_for_one_protein,
                bmrb_id, pdb_ids, pdb_dir, search_dirs, online,
            ): bmrb_id
            for bmrb_id, pdb_ids in bmrb_to_pdb_ids.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Resolving PDB structures"):
            bmrb_id, protein_candidates, local_counters = future.result()
            if protein_candidates:
                candidates[bmrb_id] = protein_candidates
            for k, v in local_counters.items():
                counters[k] += v

    print(f"\n  PDB structure search results:")
    print(f"    Found structures: {counters['found']:,}")
    print(f"    No mapping:       {counters['no_mapping']:,}")
    print(f"    No valid PDB:     {counters['no_valid_pdb']:,}")
    if online:
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

    # Pre-index shifts for the proteins we need
    shifts_by_protein = {
        bid: grp for bid, grp in shifts_df[
            shifts_df['bmrb_id'].isin(candidates.keys())
        ].groupby('bmrb_id')
    }

    for bmrb_id in tqdm(sorted(candidates.keys()), desc="Selecting best chains"):
        prot_shifts = shifts_by_protein.get(bmrb_id)
        if prot_shifts is None:
            continue
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
    # BMRB->UniProt mapping cache for offline mode
    mapping_cache_path = os.path.join(alphafold_dir, 'bmrb_uniprot_mapping.json')

    if not online:
        # Offline: use cached BMRB->UniProt mappings + existing AlphaFold PDBs
        from alphafold_utils import AF_MODEL_VERSION
        af_structures = {}
        mapping = {}
        # Load primary mapping
        if os.path.exists(mapping_cache_path):
            with open(mapping_cache_path) as f:
                mapping = json.load(f)
        # Merge supplementary mappings (from find_missing_uniprots.py)
        new_mapping_path = os.path.join(alphafold_dir, 'new_uniprot_mappings.json')
        if os.path.exists(new_mapping_path):
            with open(new_mapping_path) as f:
                new_mapping = json.load(f)
            mapping.update(new_mapping)
            print(f"  Merged {len(new_mapping)} supplementary UniProt mappings")

        if not mapping:
            print("  Offline mode: no AlphaFold mapping cache found. Run with --online first.")
            return af_structures

        for bmrb_id in bmrb_ids:
            uniprot_id = mapping.get(str(bmrb_id))
            if uniprot_id is None:
                continue
            af_path = os.path.join(alphafold_dir, f'AF-{uniprot_id}-F1-{AF_MODEL_VERSION}.pdb')
            if os.path.exists(af_path):
                af_structures[bmrb_id] = (af_path, 'A')
        print(f"  Offline: found {len(af_structures)} AlphaFold structures")
        return af_structures

    os.makedirs(alphafold_dir, exist_ok=True)
    af_structures = {}
    counters = defaultdict(int)
    bmrb_uniprot_mapping = {}

    # Load existing mapping cache to avoid redundant API calls
    if os.path.exists(mapping_cache_path):
        with open(mapping_cache_path) as f:
            bmrb_uniprot_mapping = json.load(f)

    for bmrb_id in tqdm(bmrb_ids, desc="Fetching AlphaFold structures"):
        # Step 1: Map BMRB -> UniProt (check cache first)
        if str(bmrb_id) in bmrb_uniprot_mapping:
            uniprot_id = bmrb_uniprot_mapping[str(bmrb_id)]
        else:
            uniprot_id = get_uniprot_for_bmrb(bmrb_id)
            if uniprot_id is not None:
                bmrb_uniprot_mapping[str(bmrb_id)] = uniprot_id

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

    # Save mapping cache for offline use
    with open(mapping_cache_path, 'w') as f:
        json.dump(bmrb_uniprot_mapping, f, indent=2)

    print(f"\n  AlphaFold structure results:")
    print(f"    Downloaded:          {counters['af_downloaded']:,}")
    print(f"    No UniProt mapping:  {counters['no_uniprot']:,}")
    print(f"    AF not available:    {counters['af_not_available']:,}")
    print(f"    BMRB->UniProt mappings cached: {len(bmrb_uniprot_mapping):,}")

    return af_structures


# ============================================================================
# Step 4: Compile features for one protein
# ============================================================================

def select_best_nmr_model(pdb_path, chain_id):
    """Select the NMR model closest to the median structure.

    For multi-model PDB files (NMR):
      1. Parse all models
      2. Superimpose all onto Model 1 via Kabsch on shared CA atoms
      3. Compute median CA coordinates
      4. Return the model with lowest RMSD from median

    For single-model files, returns the model directly.

    Args:
        pdb_path: Path to PDB file
        chain_id: Chain to use

    Returns:
        residues dict for the best model, or None on failure
    """
    models = parse_pdb_all_models(pdb_path, chain_id=chain_id)
    if not models:
        return None
    if len(models) == 1:
        return models[0]

    # Extract CA coords from each model
    # Use residue IDs from model 0 as reference
    ref_keys = []
    for key in sorted(models[0].keys()):
        if 'CA' in models[0][key].get('atoms', {}):
            ref_keys.append(key)

    if len(ref_keys) < 10:
        return models[0]

    # Find keys present in ALL models with CA
    common_keys = set(ref_keys)
    for model in models[1:]:
        model_ca_keys = {k for k in model if 'CA' in model[k].get('atoms', {})}
        common_keys &= model_ca_keys

    common_keys = sorted(common_keys)
    if len(common_keys) < 10:
        return models[0]

    n_models = len(models)
    n_pos = len(common_keys)

    # Extract CA coordinate matrices (n_models x n_pos x 3)
    all_coords = np.zeros((n_models, n_pos, 3))
    for mi, model in enumerate(models):
        for pi, key in enumerate(common_keys):
            all_coords[mi, pi, :] = model[key]['atoms']['CA']

    # Superimpose all models onto model 0
    ref_coords = all_coords[0]
    aligned_coords = np.zeros_like(all_coords)
    aligned_coords[0] = ref_coords

    for mi in range(1, n_models):
        rot, trans = kabsch_superimpose(all_coords[mi], ref_coords)
        aligned_coords[mi] = all_coords[mi] @ rot + trans

    # Compute median coordinates
    median_coords = np.median(aligned_coords, axis=0)

    # Compute per-model RMSD from median
    rmsds = np.zeros(n_models)
    for mi in range(n_models):
        diff = aligned_coords[mi] - median_coords
        rmsds[mi] = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    best_idx = int(np.argmin(rmsds))
    return models[best_idx]


def process_protein(bmrb_id, shifts_df, pdb_path, chain_id):
    """Process a single protein: parse PDB, align, compute features.

    Returns:
        list of row dicts (one per aligned residue), or empty list on failure
    """
    rows = []

    # For NMR structures (multi-model), select the median-representative model
    residues = select_best_nmr_model(pdb_path, chain_id)
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

    # Compute sequence identity and reject below threshold
    n_aligned = sum(1 for m in mapping if m['mismatch_type'] not in ('gap_in_cs', 'gap_in_structure'))
    n_match = sum(1 for m in mapping if m['mismatch_type'] in ('match', 'protein_edge'))
    if n_aligned == 0 or (n_match / n_aligned) < MIN_SEQUENCE_IDENTITY:
        return rows

    # Spatial neighbors
    neighbors = find_neighbors(struct_data)

    # Index shift data
    shift_indexed = shifts_df.drop_duplicates(subset='residue_id').set_index('residue_id')
    shift_cols = sorted([c for c in shifts_df.columns if c.endswith('_shift')])
    ambig_cols = sorted([c for c in shifts_df.columns if c.endswith('_ambiguity_code')])

    sorted_rids = sorted(struct_data.keys())

    # Build rows
    for m in mapping:
        struct_rid = m['struct_res_id']
        shift_rid = m['shift_res_id']
        mismatch_type = m['mismatch_type']

        # For gap_in_cs: no shift data, only structural features
        has_shift = not (isinstance(shift_rid, float) and np.isnan(shift_rid))
        has_struct = not (isinstance(struct_rid, float) and np.isnan(struct_rid))

        if has_shift and shift_rid not in shift_indexed.index:
            continue

        # Get shift data if available
        if has_shift:
            shift_row = shift_indexed.loc[shift_rid]
            if isinstance(shift_row, pd.DataFrame):
                shift_row = shift_row.iloc[0]
            residue_code = shift_row.get('residue_code', 'UNK')
            if pd.isna(residue_code):
                residue_code = 'UNK'
        else:
            shift_row = None
            # Use struct residue name for gap_in_cs rows
            if has_struct and struct_rid in struct_data:
                rname = struct_data[struct_rid].get('residue_name', 'UNK')
                residue_code = rname if rname in AA_3_TO_1 else 'UNK'
            else:
                residue_code = 'UNK'

        row = {
            'bmrb_id': str(bmrb_id),
            'residue_id': struct_rid if has_struct else shift_rid,
            'residue_code': residue_code,
            'mismatch_type': mismatch_type,
        }

        # Chemical shifts (only if shift data exists)
        if shift_row is not None:
            for col in shift_cols + ambig_cols:
                val = shift_row.get(col)
                if val is not None and not pd.isna(val):
                    row[col] = val

        # Structural features
        if has_struct and struct_rid in struct_data:
            rdata = struct_data[struct_rid]
            atoms = rdata.get('atoms', {})

            # CA coordinates
            if 'CA' in atoms and np.all(np.isfinite(atoms['CA'])):
                row['ca_x'], row['ca_y'], row['ca_z'] = atoms['CA'].tolist()

            # Distance features (all heavy + H-to-heavy)
            dist_feats = compute_all_distance_features(atoms)
            row.update(dist_feats)

        # DSSP data
        if has_struct:
            dssp_entry = lookup_with_chain_fallback(dssp_data, chain_used or '', struct_rid)
        else:
            dssp_entry = None
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

        # Spatial neighbors
        if has_struct and struct_rid in neighbors:
            nb = neighbors[struct_rid]
            for i in range(K_SPATIAL_NEIGHBORS):
                row[f'spatial_neighbor_{i}_id'] = nb['ids'][i]
                row[f'spatial_neighbor_{i}_dist'] = nb['dists'][i]
                row[f'spatial_neighbor_{i}_seq_sep'] = nb['seps'][i]

        rows.append(row)

    # --- Inter-residue backbone bond geometry ---
    # Second pass: compute distances between sequential neighbors.
    # These require coordinates from adjacent residues which are only
    # available after the full protein has been processed.
    rows_by_rid = {r['residue_id']: r for r in rows if 'ca_x' in r}
    sorted_row_rids = sorted(rows_by_rid.keys())

    for i, rid in enumerate(sorted_row_rids):
        r = rows_by_rid[rid]
        ca_i = np.array([r['ca_x'], r['ca_y'], r['ca_z']])

        atoms_i = struct_data[rid]['atoms'] if rid in struct_data else {}

        # Previous residue (i-1)
        if i > 0:
            prev_rid = sorted_row_rids[i - 1]
            rp = rows_by_rid[prev_rid]
            ca_prev = np.array([rp['ca_x'], rp['ca_y'], rp['ca_z']])
            r['bond_ca_prev'] = float(np.linalg.norm(ca_i - ca_prev))

            # Peptide bond backward: C(i-1) -> N(i)
            atoms_prev = struct_data.get(prev_rid, {}).get('atoms', {})
            if 'C' in atoms_prev and 'N' in atoms_i:
                c_prev = atoms_prev['C']
                n_i = atoms_i['N']
                if np.all(np.isfinite(c_prev)) and np.all(np.isfinite(n_i)):
                    r['bond_peptide_bkwd'] = float(np.linalg.norm(n_i - c_prev))

        # Next residue (i+1)
        if i < len(sorted_row_rids) - 1:
            next_rid = sorted_row_rids[i + 1]
            rn = rows_by_rid[next_rid]
            ca_next = np.array([rn['ca_x'], rn['ca_y'], rn['ca_z']])
            r['bond_ca_next'] = float(np.linalg.norm(ca_i - ca_next))

            # Peptide bond forward: C(i) -> N(i+1)
            atoms_next = struct_data.get(next_rid, {}).get('atoms', {})
            if 'C' in atoms_i and 'N' in atoms_next:
                c_i = atoms_i['C']
                n_next = atoms_next['N']
                if np.all(np.isfinite(c_i)) and np.all(np.isfinite(n_next)):
                    r['bond_peptide_fwd'] = float(np.linalg.norm(n_next - c_i))

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
    """Set outlier shift values to NaN per (residue_code, shift_col) group.

    For each group, computes mean and std, then sets values > threshold SDs
    from the mean to NaN.
    """
    if sd_threshold is None:
        sd_threshold = OUTLIER_STD_THRESHOLD

    shift_cols = [c for c in df.columns if c.endswith('_shift')]
    group_col = 'residue_code'

    if group_col not in df.columns:
        # Fallback: no grouping
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
        for res_code, group in df.groupby(group_col):
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
          f"(>{sd_threshold} SD per residue_code x shift_col)")

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
    standard_set = set(STANDARD_RESIDUES) - {'UNK'}
    nonstandard_proteins = (
        shifts_df[~shifts_df['residue_code'].str.upper().isin(standard_set)]
        ['bmrb_id'].unique()
    )
    nonstandard_set = set(nonstandard_proteins)
    bmrb_ids = [bid for bid in bmrb_ids if bid not in nonstandard_set]
    print(f"  Discarded {len(nonstandard_set)} proteins with nonstandard residues")
    print(f"  Remaining: {len(bmrb_ids):,}")

    # Load pairs mapping (auto-download from BMRB if missing)
    if not os.path.exists(pairs_file):
        print(f"\n  pairs.csv not found at {pairs_file}, downloading from BMRB...")
        download_pairs_file(pairs_file)

    pairs_mapping = load_pairs_file(pairs_file)
    if pairs_mapping:
        print(f"  Loaded {len(pairs_mapping):,} BMRB->PDB mappings from pairs.csv")
    else:
        print("  WARNING: No BMRB->PDB mappings available. "
              "Structure finding will rely on BMRB API and BLAST.")

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
    print("\n--- Step 2: Selecting best chains (alignment identity) ---")
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
    # Dataset 1 (hybrid): all BMRB + AlphaFold — experimental where available, AlphaFold otherwise
    # Dataset 2 (experimental): experimental PDB structures only
    # Dataset 3 (alphafold): all AlphaFold structures

    dataset_configs = {
        'hybrid': {},
        'experimental': {},
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

    print(f"  Hybrid:       {len(dataset_configs['hybrid']):,} proteins (primary dataset)")
    print(f"  Experimental: {len(dataset_configs['experimental']):,} proteins")
    print(f"  AlphaFold:    {len(dataset_configs['alphafold']):,} proteins")

    # Build log rows
    log_rows = []

    # Pre-index shifts once for all datasets
    all_dataset_bmrb_ids = set()
    for selections in dataset_configs.values():
        all_dataset_bmrb_ids.update(selections.keys())
    shifts_by_protein = {
        bid: grp for bid, grp in shifts_df[
            shifts_df['bmrb_id'].isin(all_dataset_bmrb_ids)
        ].groupby('bmrb_id')
    }

    for dataset_name, selections in dataset_configs.items():
        print(f"\n  Compiling {dataset_name} dataset...")

        failed = 0
        succeeded = 0
        total_residues = 0

        # Process in batches to limit peak memory
        BATCH_SIZE = 2000  # proteins per batch
        sorted_ids = sorted(selections.keys())
        all_batch_dfs = []

        for batch_start in range(0, len(sorted_ids), BATCH_SIZE):
            batch_ids = sorted_ids[batch_start:batch_start + BATCH_SIZE]
            batch_rows = []

            for bmrb_id in tqdm(batch_ids,
                                desc=f"  {dataset_name} [{batch_start}:{batch_start+len(batch_ids)}]",
                                unit="protein"):
                pdb_path, chain_id = selections[bmrb_id]
                prot_shifts = shifts_by_protein.get(bmrb_id)
                if prot_shifts is None:
                    continue

                try:
                    rows = process_protein(bmrb_id, prot_shifts, pdb_path, chain_id)
                    if rows:
                        batch_rows.extend(rows)
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

            if batch_rows:
                batch_df = pd.DataFrame(batch_rows)
                batch_df['split'] = batch_df['bmrb_id'].apply(
                    lambda bid: assign_fold(bid, n_folds=args.n_folds))
                all_batch_dfs.append(batch_df)
                del batch_rows
                print(f"    Batch done: {len(batch_df):,} residues, "
                      f"cumulative: {sum(len(d) for d in all_batch_dfs):,}")

        if not all_batch_dfs:
            print(f"    No residues compiled for {dataset_name}")
            continue

        # Collect all columns across batches, then write with uniform schema
        output_path = os.path.join(output_dir, f'structure_data_{dataset_name}.csv')
        all_cols = set()
        for batch_df in all_batch_dfs:
            all_cols.update(batch_df.columns)
        all_cols = sorted(all_cols)
        print(f"    Writing {len(all_batch_dfs)} batches ({len(all_cols)} columns) to {output_path}...")
        total_rows = 0
        for bi, batch_df in enumerate(all_batch_dfs):
            # Reindex to uniform columns (missing cols become NaN)
            batch_df = batch_df.reindex(columns=all_cols)
            batch_df.to_csv(output_path, mode='a' if bi > 0 else 'w',
                            index=False, header=(bi == 0))
            total_rows += len(batch_df)
        del all_batch_dfs
        import gc; gc.collect()

        print(f"    Saved: {output_path}")
        print(f"    Total: {succeeded} proteins, {total_rows:,} residues, "
              f"({succeeded} ok, {failed} failed)")
        print(f"    Note: Run analyze_data_quality.py for quality filtering")

        gc.collect()

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

    for dataset_name in ['hybrid', 'experimental', 'alphafold']:
        path = os.path.join(output_dir, f'structure_data_{dataset_name}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=['bmrb_id'], dtype={'bmrb_id': str})
            print(f"  {dataset_name:15s}: {df['bmrb_id'].nunique():,} proteins")

    print()


if __name__ == '__main__':
    main()
