#!/usr/bin/env python3
"""
Step 01: Select the best PDB structure for each BMRB entry.

For each protein in chemical_shifts.csv, finds associated PDB structures and
selects the best one based on quality criteria:

1. Skip superseded entries
2. Prefer X-ray over NMR/cryo-EM
3. Among X-ray: pick lowest resolution, require < 2.5 A
4. Among NMR: pick most recent deposition
5. If multiple PDBs per BMRB: pick best by above criteria

Also checks for existing PDB files in local directories before downloading.

Uses the BMRB REST API to look up PDB IDs associated with each BMRB entry,
then queries the RCSB REST API for structural metadata.

Outputs:
- data/pdb_selection.csv          -- selected PDB for each protein
- data/pdb_selection_log.csv      -- provenance log (what was rejected and why)
- data/pdbs/                      -- downloaded PDB files
"""

import argparse
import os
import sys
import time
import json
import urllib.request
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np
from tqdm import tqdm

# Allow imports from the homologies_better_data package
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, PDB_DIR
from data_quality import FilterLog


# ============================================================================
# Constants
# ============================================================================

RCSB_ENTRY_URL = 'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
RCSB_POLYMER_URL = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}'
RCSB_DOWNLOAD_URL = 'https://files.rcsb.org/download/{pdb_id}.pdb'
BMRB_PDB_LOOKUP_URL = 'https://api.bmrb.io/v2/search/get_pdb_ids_from_bmrb_id/{bmrb_id}'
BMRB_ENTRY_URL = 'https://api.bmrb.io/v2/entry/{bmrb_id}?format=json'

# Local pairs.csv that maps BMRB -> PDB IDs (from existing pipeline)
DEFAULT_PAIRS_FILE = 'data/pairs.csv'

MAX_RESOLUTION = 2.5  # Angstroms; reject X-ray structures worse than this
REQUEST_DELAY = 0.1    # Seconds between API requests (polite but faster)
MAX_RETRIES = 3
MAX_WORKERS = 8        # Concurrent API threads

# Thread-local rate limiter
_rate_lock = threading.Lock()
_last_request_time = 0.0


# ============================================================================
# API Helpers
# ============================================================================

def _rate_limit():
    """Thread-safe rate limiter to stay polite to APIs."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        _last_request_time = time.monotonic()


def _fetch_json(url, retries=MAX_RETRIES):
    """Fetch JSON from a URL with retries and polite delay.

    Returns parsed JSON dict, or None on failure.
    """
    for attempt in range(retries):
        _rate_limit()
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            req.add_header('User-Agent', 'he_lab_pipeline/1.0')
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # Not found, don't retry
            if e.code == 429:
                # Rate limited -- back off
                wait = 5 * (attempt + 1)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def lookup_pdb_ids_for_bmrb(bmrb_id):
    """Look up PDB IDs associated with a BMRB entry via the BMRB REST API.

    Returns a list of PDB ID strings (uppercase), or empty list on failure.
    """
    url = BMRB_PDB_LOOKUP_URL.format(bmrb_id=bmrb_id)
    data = _fetch_json(url)
    if data is None:
        return []

    # Response is typically a list of PDB IDs or a dict with a list
    if isinstance(data, list):
        return [str(pdb).upper().strip() for pdb in data if pdb]
    if isinstance(data, dict):
        # Some API versions return {'data': [...]} or similar
        for key in ('data', 'pdb_ids', 'result'):
            if key in data and isinstance(data[key], list):
                return [str(pdb).upper().strip() for pdb in data[key] if pdb]
    return []


def query_rcsb(pdb_id):
    """Query RCSB REST API for PDB entry metadata.

    Returns a dict with keys:
        pdb_id, method, resolution, r_free, deposition_date, revision_date,
        is_superseded, superseded_by, entity_chains, status
    Returns None on failure.
    """
    url = RCSB_ENTRY_URL.format(pdb_id=pdb_id.upper())
    data = _fetch_json(url)
    if data is None:
        return None

    result = {
        'pdb_id': pdb_id.upper(),
        'method': None,
        'resolution': None,
        'r_free': None,
        'deposition_date': None,
        'revision_date': None,
        'is_superseded': False,
        'superseded_by': None,
        'entity_chains': {},  # entity_id -> [chain_ids]
        'status': None,
    }

    try:
        # Experimental method
        exptl = data.get('exptl', [])
        if exptl and isinstance(exptl, list):
            result['method'] = exptl[0].get('method', '').upper()

        # Resolution
        # Try rcsb_entry_info first, then refine
        entry_info = data.get('rcsb_entry_info', {})
        if entry_info:
            result['resolution'] = entry_info.get('resolution_combined')
            if isinstance(result['resolution'], list) and result['resolution']:
                result['resolution'] = result['resolution'][0]

        # Also check refine section
        refine = data.get('refine', [])
        if refine and isinstance(refine, list):
            refine_data = refine[0]
            if result['resolution'] is None:
                result['resolution'] = refine_data.get('ls_d_res_high')
            result['r_free'] = refine_data.get('ls_R_factor_R_free')

        # Deposition and revision dates
        audit = data.get('rcsb_accession_info', {})
        result['deposition_date'] = audit.get('deposit_date', '')
        result['revision_date'] = audit.get('revision_date', '')

        # Check if superseded
        pdb_audit = data.get('pdbx_audit_revision_history', [])
        if pdb_audit and isinstance(pdb_audit, list):
            # Check for supersession in revision history
            for rev in pdb_audit:
                if isinstance(rev, dict) and rev.get('data_content_type') == 'supersede':
                    result['is_superseded'] = True

        # Also check for obsolete status
        entry_container = data.get('rcsb_entry_container_identifiers', {})
        result['status'] = entry_info.get('structure_determination_methodology', '')

        # Entity -> chain mapping
        polymer_entities = data.get('polymer_entities', [])
        if not polymer_entities:
            # Try alternative path
            entity_ids = entry_container.get('polymer_entity_ids', [])
            if entity_ids:
                for eid in entity_ids:
                    result['entity_chains'][str(eid)] = []

        # Get chain info from entry
        struct = data.get('struct', {})
        cell = data.get('cell', {})

        # Check pdbx_database_status for obsolete flag
        db_status = data.get('pdbx_database_status', {})
        if db_status:
            status_code = db_status.get('status_code', '')
            if status_code and status_code.upper() in ('OBS', 'OBSOLETE'):
                result['is_superseded'] = True
            pdb_superseded_by = db_status.get('pdb_format_compatible', '')

    except Exception as e:
        print(f"    WARNING: Error parsing RCSB data for {pdb_id}: {e}")

    return result


def get_chains_for_pdb(pdb_id):
    """Get protein chain IDs for a PDB entry from RCSB.

    Returns a list of chain ID strings, or ['A'] as fallback.
    """
    url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}'
    data = _fetch_json(url)
    if data is None:
        return ['A']

    chains = []
    try:
        # Try to get polymer entity info
        container = data.get('rcsb_entry_container_identifiers', {})
        polymer_ids = container.get('polymer_entity_ids', [])

        for eid in polymer_ids:
            entity_url = RCSB_POLYMER_URL.format(pdb_id=pdb_id.upper(), entity_id=eid)
            entity_data = _fetch_json(entity_url)
            if entity_data is None:
                continue

            # Get chain IDs for this entity
            entity_container = entity_data.get('rcsb_polymer_entity_container_identifiers', {})
            auth_chains = entity_container.get('auth_asym_ids', [])
            if auth_chains:
                chains.extend(auth_chains)


    except Exception:
        pass

    return chains if chains else ['A']


RCSB_GRAPHQL_URL = 'https://data.rcsb.org/graphql'
RCSB_GRAPHQL_BATCH = 2000  # Max IDs per GraphQL request


def _bulk_fetch_rcsb_metadata(pdb_ids):
    """Fetch metadata for many PDB IDs in a few bulk GraphQL requests.

    Returns dict: pdb_id (uppercase) -> metadata dict compatible with select_best_pdb().
    """
    metadata = {}
    pdb_list = list(pdb_ids)

    for batch_start in range(0, len(pdb_list), RCSB_GRAPHQL_BATCH):
        batch = pdb_list[batch_start:batch_start + RCSB_GRAPHQL_BATCH]
        batch_n = batch_start // RCSB_GRAPHQL_BATCH + 1
        total_batches = (len(pdb_list) + RCSB_GRAPHQL_BATCH - 1) // RCSB_GRAPHQL_BATCH
        print(f"  Fetching RCSB metadata batch {batch_n}/{total_batches} ({len(batch)} PDBs)...")

        # Build GraphQL query
        ids_str = json.dumps(batch)
        query = {
            "query": """
            query($ids: [String!]!) {
              entries(entry_ids: $ids) {
                rcsb_id
                exptl {
                  method
                }
                rcsb_entry_info {
                  resolution_combined
                }
                refine {
                  ls_d_res_high
                  ls_R_factor_R_free
                }
                rcsb_accession_info {
                  deposit_date
                  revision_date
                  status_code
                }
              }
            }
            """,
            "variables": {"ids": batch}
        }

        payload = json.dumps(query).encode('utf-8')
        req = urllib.request.Request(
            RCSB_GRAPHQL_URL,
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'he_lab_pipeline/1.0',
            },
            method='POST',
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            print(f"    WARNING: GraphQL batch {batch_n} failed: {e}")
            continue

        entries = result.get('data', {}).get('entries', [])
        if entries is None:
            entries = []

        for entry in entries:
            if entry is None:
                continue
            pdb_id = entry.get('rcsb_id', '').upper()
            if not pdb_id:
                continue

            # Parse method
            method = None
            exptl = entry.get('exptl') or []
            if exptl and isinstance(exptl, list):
                method = (exptl[0].get('method') or '').upper()

            # Parse resolution
            resolution = None
            entry_info = entry.get('rcsb_entry_info') or {}
            res_combined = entry_info.get('resolution_combined')
            if isinstance(res_combined, list) and res_combined:
                resolution = res_combined[0]
            elif isinstance(res_combined, (int, float)):
                resolution = res_combined

            # Fallback to refine
            r_free = None
            refine = entry.get('refine') or []
            if refine and isinstance(refine, list):
                ref = refine[0]
                if resolution is None:
                    resolution = ref.get('ls_d_res_high')
                r_free = ref.get('ls_R_factor_R_free')

            # Parse dates and status
            accession = entry.get('rcsb_accession_info') or {}
            deposition_date = accession.get('deposit_date', '') or ''
            status_code = (accession.get('status_code') or '').upper()
            is_superseded = status_code in ('OBS', 'OBSOLETE')

            metadata[pdb_id] = {
                'pdb_id': pdb_id,
                'method': method,
                'resolution': resolution,
                'r_free': r_free,
                'deposition_date': deposition_date,
                'revision_date': accession.get('revision_date', ''),
                'is_superseded': is_superseded,
                'superseded_by': None,
                'entity_chains': {},
                'status': status_code,
            }

    return metadata


def download_pdb_file(pdb_id, output_path, retries=MAX_RETRIES):
    """Download a PDB file from RCSB.

    Returns True on success, False on failure.
    """
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'he_lab_pipeline/1.0')
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(output_path, 'wb') as f:
                    f.write(resp.read())
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    WARNING: Failed to download PDB {pdb_id}: {e}")
                return False
    return False


# ============================================================================
# Local PDB/Pairs Lookup
# ============================================================================

def load_pairs_file(pairs_path):
    """Load BMRB-to-PDB mapping from pairs.csv.

    Returns dict: bmrb_id (str) -> list of PDB IDs (str, uppercase)
    """
    if not os.path.exists(pairs_path):
        print(f"  Pairs file not found: {pairs_path}")
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

    print(f"  Loaded {len(mapping):,} BMRB->PDB mappings from {pairs_path}")
    return mapping


def find_existing_pdb(pdb_id, chain_id, existing_dirs):
    """Search for an existing PDB file in local directories.

    Checks multiple naming conventions:
    - {pdb_id}.pdb
    - {pdb_id}{chain_id}.pdb  (CSpred convention)
    - {pdb_id.lower()}.pdb

    Returns the path if found, None otherwise.
    """
    pdb_upper = pdb_id.upper()
    pdb_lower = pdb_id.lower()

    name_patterns = [
        f'{pdb_upper}.pdb',
        f'{pdb_lower}.pdb',
        f'{pdb_upper}{chain_id}.pdb' if chain_id else None,
        f'{pdb_lower}{chain_id}.pdb' if chain_id else None,
    ]
    # Filter out None entries
    name_patterns = [n for n in name_patterns if n is not None]

    for dir_path in existing_dirs:
        if not os.path.isdir(dir_path):
            continue
        for name in name_patterns:
            full_path = os.path.join(dir_path, name)
            if os.path.exists(full_path):
                return full_path

    return None


# ============================================================================
# PDB Selection Logic
# ============================================================================

def select_best_pdb(candidates, log=None, bmrb_id=None):
    """Select the best PDB from a list of candidates based on quality criteria.

    Args:
        candidates: list of dicts from query_rcsb()
        log: FilterLog instance
        bmrb_id: for provenance logging

    Returns:
        (best_candidate_dict, reason_string) or (None, reason_string)
    """
    if log is None:
        log = FilterLog()

    if not candidates:
        return None, 'no_candidates'

    # Step 1: Remove superseded entries
    active = []
    for c in candidates:
        if c.get('is_superseded', False):
            log.add(
                step='pdb_selection',
                reason=f'superseded entry',
                bmrb_id=bmrb_id,
                column='pdb_id',
                value=c['pdb_id'],
                action='rejected',
            )
        else:
            active.append(c)

    if not active:
        return None, 'all_superseded'

    # Step 2: Separate by experimental method
    xray = [c for c in active if c.get('method', '') in ('X-RAY DIFFRACTION', 'X-RAY')]
    nmr = [c for c in active if c.get('method', '') in ('SOLUTION NMR', 'SOLID-STATE NMR', 'NMR')]
    cryo = [c for c in active if 'ELECTRON' in c.get('method', '')]
    other = [c for c in active if c not in xray and c not in nmr and c not in cryo]

    # Step 3: Try X-ray first (prefer lowest resolution < 2.5 A)
    if xray:
        # Filter by resolution threshold
        good_xray = []
        for c in xray:
            res = c.get('resolution')
            if res is not None:
                try:
                    res_float = float(res)
                    if res_float <= MAX_RESOLUTION:
                        good_xray.append((res_float, c))
                    else:
                        log.add(
                            step='pdb_selection',
                            reason=f'X-ray resolution {res_float:.2f} A > {MAX_RESOLUTION} A threshold',
                            bmrb_id=bmrb_id,
                            column='pdb_id',
                            value=c['pdb_id'],
                            action='rejected',
                        )
                except (ValueError, TypeError):
                    # Resolution is present but unparseable; include with penalty
                    good_xray.append((MAX_RESOLUTION, c))
            else:
                # No resolution reported for X-ray; include with penalty
                good_xray.append((MAX_RESOLUTION, c))

        if good_xray:
            good_xray.sort(key=lambda x: x[0])
            best = good_xray[0][1]
            res_str = f"{good_xray[0][0]:.2f}" if good_xray[0][0] != MAX_RESOLUTION else "unknown"
            return best, f'xray_best_resolution_{res_str}A'

        # All X-ray structures are too low resolution; fall through to NMR/other

    # Step 4: Try NMR (prefer most recent deposition)
    if nmr:
        # Sort by deposition date descending (most recent first)
        dated_nmr = []
        for c in nmr:
            dep_date = c.get('deposition_date', '') or ''
            dated_nmr.append((dep_date, c))

        dated_nmr.sort(key=lambda x: x[0], reverse=True)
        best = dated_nmr[0][1]
        date_str = dated_nmr[0][0][:10] if dated_nmr[0][0] else 'unknown'
        return best, f'nmr_most_recent_{date_str}'

    # Step 5: Try cryo-EM (prefer lowest resolution)
    if cryo:
        res_cryo = []
        for c in cryo:
            res = c.get('resolution')
            if res is not None:
                try:
                    res_cryo.append((float(res), c))
                except (ValueError, TypeError):
                    res_cryo.append((99.0, c))
            else:
                res_cryo.append((99.0, c))

        res_cryo.sort(key=lambda x: x[0])
        best = res_cryo[0][1]
        res_str = f"{res_cryo[0][0]:.2f}" if res_cryo[0][0] != 99.0 else "unknown"
        return best, f'cryo_em_best_resolution_{res_str}A'

    # Step 6: Anything else -- just pick first
    if other:
        return other[0], f'other_method_{other[0].get("method", "unknown")}'

    return None, 'no_suitable_candidates'


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Select the best PDB structure for each BMRB entry.'
    )
    parser.add_argument(
        '--shifts-file', default=os.path.join(DATA_DIR, 'chemical_shifts.csv'),
        help='Path to cleaned chemical shifts CSV (default: data/chemical_shifts.csv)'
    )
    parser.add_argument(
        '--pdb-dir', default=PDB_DIR,
        help=f'Directory to store downloaded PDB files (default: {PDB_DIR})'
    )
    parser.add_argument(
        '--existing-pdbs', nargs='+',
        default=['data/pdbs'],
        help='Directories to search for existing PDB files'
    )
    parser.add_argument(
        '--pairs-file', default=DEFAULT_PAIRS_FILE,
        help=f'Path to BMRB->PDB pairs.csv (default: {DEFAULT_PAIRS_FILE})'
    )
    parser.add_argument(
        '--output-dir', default=DATA_DIR,
        help=f'Output directory (default: {DATA_DIR})'
    )
    parser.add_argument(
        '--skip-download', action='store_true', default=False,
        help='Skip downloading PDB files (only generate selection CSV)'
    )
    parser.add_argument(
        '--max-proteins', type=int, default=None,
        help='Limit processing to first N proteins (for testing)'
    )
    parser.add_argument(
        '--workers', type=int, default=MAX_WORKERS,
        help=f'Number of concurrent API threads (default: {MAX_WORKERS})'
    )
    parser.add_argument(
        '--online', action='store_true', default=False,
        help='Enable API calls to BMRB/RCSB. Default is offline mode which '
             'uses only pairs.csv and local PDB files (no network).'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 01: SELECT PDB STRUCTURES FOR BMRB ENTRIES")
    print("=" * 70)

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(path):
        return os.path.join(script_dir, path) if not os.path.isabs(path) else path

    shifts_file = resolve(args.shifts_file)
    pdb_dir = resolve(args.pdb_dir)
    output_dir = resolve(args.output_dir)
    pairs_file = resolve(args.pairs_file)
    existing_dirs = [resolve(d) for d in args.existing_pdbs]

    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    output_selection = os.path.join(output_dir, 'pdb_selection.csv')
    output_log = os.path.join(output_dir, 'pdb_selection_log.csv')

    log = FilterLog()

    # ------------------------------------------------------------------
    # Step 1: Load BMRB IDs from shift data
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading BMRB IDs from shift data...")

    if not os.path.exists(shifts_file):
        print(f"  ERROR: Shift file not found: {shifts_file}")
        print(f"  Run 00_fetch_bmrb_shifts.py first.")
        sys.exit(1)

    shift_df = pd.read_csv(shifts_file, dtype={'bmrb_id': str}, usecols=['bmrb_id'])
    bmrb_ids = sorted(shift_df['bmrb_id'].unique())

    if args.max_proteins is not None:
        bmrb_ids = bmrb_ids[:args.max_proteins]
        print(f"  (Limited to first {args.max_proteins} proteins for testing)")

    print(f"  Found {len(bmrb_ids):,} unique BMRB IDs")
    log.add_summary('input', 'BMRB IDs from shift data', len(bmrb_ids))

    # ------------------------------------------------------------------
    # Step 2: Load BMRB -> PDB mapping from pairs.csv (local cache)
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading BMRB -> PDB mapping...")
    pairs_mapping = load_pairs_file(pairs_file)

    # ------------------------------------------------------------------
    # Step 3: For each BMRB ID, find and evaluate PDB structures
    # ------------------------------------------------------------------
    selection_rows = []
    counters = defaultdict(int)

    if not args.online:
        # ============================================================
        # DEFAULT MODE: bulk RCSB GraphQL query + local PDB files
        # One network request for all PDB metadata, then local selection.
        # ============================================================
        print(f"\n[Step 3] Selecting best PDBs (bulk metadata query)...")

        # Index local PDB files for fast lookup
        local_pdbs = set()
        for d in existing_dirs + [pdb_dir]:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith('.pdb'):
                        local_pdbs.add(f.replace('.pdb', '').upper())
        print(f"  Local PDB files found: {len(local_pdbs):,}")

        # Collect all unique PDB IDs we need metadata for
        all_pdb_ids = set()
        bmrb_to_pdbs = {}
        no_mapping_ids = []
        for bmrb_id in bmrb_ids:
            if bmrb_id in pairs_mapping:
                pids = [pid.upper() for pid in pairs_mapping[bmrb_id]]
                bmrb_to_pdbs[bmrb_id] = pids
                all_pdb_ids.update(pids)
            else:
                no_mapping_ids.append(bmrb_id)

        print(f"  BMRB IDs with PDB mapping: {len(bmrb_to_pdbs):,}")
        print(f"  BMRB IDs without mapping:  {len(no_mapping_ids):,}")
        print(f"  Unique PDB IDs to query:   {len(all_pdb_ids):,}")

        # Bulk fetch metadata from RCSB GraphQL
        pdb_metadata = _bulk_fetch_rcsb_metadata(sorted(all_pdb_ids))
        print(f"  Metadata retrieved for:    {len(pdb_metadata):,} / {len(all_pdb_ids):,} PDBs")

        # Now select best PDB per protein
        for bmrb_id in tqdm(bmrb_ids, desc="Selecting PDBs", unit="protein"):
            if bmrb_id not in bmrb_to_pdbs:
                counters['no_pdb_mapping'] += 1
                log.add(
                    step='pdb_lookup',
                    reason='no PDB IDs in pairs.csv',
                    bmrb_id=bmrb_id,
                    action='skipped',
                )
                selection_rows.append({
                    'bmrb_id': bmrb_id,
                    'pdb_id': None, 'chain_id': None, 'resolution': None,
                    'r_free': None, 'method': None, 'reason': 'no_pdb_mapping',
                })
                continue

            pdb_ids = bmrb_to_pdbs[bmrb_id]

            # Build candidate list from bulk metadata
            candidates = []
            for pid in pdb_ids:
                if pid in pdb_metadata:
                    candidates.append(pdb_metadata[pid])
                else:
                    log.add(
                        step='rcsb_query',
                        reason='no metadata returned from RCSB bulk query',
                        bmrb_id=bmrb_id, column='pdb_id', value=pid,
                        action='rejected',
                    )
                    counters['rcsb_query_failed'] += 1

            # Select best using existing quality criteria
            best, reason = select_best_pdb(candidates, log=log, bmrb_id=bmrb_id)

            if best is None:
                # Fallback: if we have local files, just pick the first available
                for pid in pdb_ids:
                    if pid in local_pdbs:
                        best = {'pdb_id': pid, 'resolution': None, 'r_free': None, 'method': None}
                        reason = 'fallback_local_file'
                        break

            if best is None:
                counters['no_suitable_pdb'] += 1
                log.add(
                    step='pdb_selection',
                    reason=f'no suitable PDB: {reason}',
                    bmrb_id=bmrb_id, action='skipped',
                )
                selection_rows.append({
                    'bmrb_id': bmrb_id,
                    'pdb_id': None, 'chain_id': None, 'resolution': None,
                    'r_free': None, 'method': None, 'reason': reason,
                })
                continue

            pdb_id = best['pdb_id']
            counters['selected'] += 1

            # Check local availability
            if pdb_id in local_pdbs:
                counters['found_existing'] += 1
            else:
                counters['pdb_not_local'] += 1

            selection_rows.append({
                'bmrb_id': bmrb_id,
                'pdb_id': pdb_id,
                'chain_id': 'A',
                'resolution': best.get('resolution'),
                'r_free': best.get('r_free'),
                'method': best.get('method'),
                'reason': reason,
            })

    else:
        # ============================================================
        # ONLINE MODE: query BMRB + RCSB APIs for full metadata
        # ============================================================
        print(f"\n[Step 3] Evaluating PDB structures ONLINE for {len(bmrb_ids):,} proteins...")
        print(f"  Using {args.workers} concurrent threads")

        _log_lock = threading.Lock()

        def _process_one_protein(bmrb_id):
            """Process a single BMRB ID: lookup PDBs, query RCSB, select best.
            Returns (row_dict, local_counters, local_log_entries)."""
            local_counters = defaultdict(int)
            local_log = []

            # Collect PDB IDs from all sources
            pdb_ids = set()

            # Source 1: pairs.csv
            if bmrb_id in pairs_mapping:
                for pid in pairs_mapping[bmrb_id]:
                    pdb_ids.add(pid.upper())

            # Source 2: BMRB API (only if pairs.csv had nothing)
            if not pdb_ids:
                api_pdbs = lookup_pdb_ids_for_bmrb(bmrb_id)
                for pid in api_pdbs:
                    pdb_ids.add(pid.upper())

            if not pdb_ids:
                local_counters['no_pdb_mapping'] += 1
                local_log.append(dict(
                    step='pdb_lookup',
                    reason='no PDB IDs found for BMRB entry',
                    bmrb_id=bmrb_id,
                    action='skipped',
                ))
                return {
                    'bmrb_id': bmrb_id,
                    'pdb_id': None, 'chain_id': None, 'resolution': None,
                    'r_free': None, 'method': None, 'reason': 'no_pdb_mapping',
                }, local_counters, local_log

            # Query RCSB for metadata on each PDB
            candidates = []
            for pdb_id in sorted(pdb_ids):
                metadata = query_rcsb(pdb_id)
                if metadata is not None:
                    candidates.append(metadata)
                else:
                    local_log.append(dict(
                        step='rcsb_query',
                        reason='RCSB query failed or entry not found',
                        bmrb_id=bmrb_id, column='pdb_id', value=pdb_id,
                        action='rejected',
                    ))
                    local_counters['rcsb_query_failed'] += 1

            # Select best PDB (thread-local FilterLog to avoid contention)
            thread_log = FilterLog()
            best, reason = select_best_pdb(candidates, log=thread_log, bmrb_id=bmrb_id)
            if hasattr(thread_log, 'entries'):
                for entry in thread_log.entries:
                    local_log.append(entry)

            if best is None:
                local_counters['no_suitable_pdb'] += 1
                local_log.append(dict(
                    step='pdb_selection',
                    reason=f'no suitable PDB: {reason}',
                    bmrb_id=bmrb_id, action='skipped',
                ))
                return {
                    'bmrb_id': bmrb_id,
                    'pdb_id': None, 'chain_id': None, 'resolution': None,
                    'r_free': None, 'method': None, 'reason': reason,
                }, local_counters, local_log

            chain_id = 'A'
            pdb_id = best['pdb_id']

            existing_path = find_existing_pdb(pdb_id, chain_id, existing_dirs + [pdb_dir])

            if existing_path:
                local_counters['found_existing'] += 1
                target_path = os.path.join(pdb_dir, f'{pdb_id}.pdb')
                if not os.path.exists(target_path) and not args.skip_download:
                    try:
                        shutil.copy2(existing_path, target_path)
                    except Exception:
                        pass
            elif not args.skip_download:
                target_path = os.path.join(pdb_dir, f'{pdb_id}.pdb')
                if not os.path.exists(target_path):
                    success = download_pdb_file(pdb_id, target_path)
                    if success:
                        local_counters['downloaded'] += 1
                    else:
                        local_counters['download_failed'] += 1
                        local_log.append(dict(
                            step='pdb_download', reason='download failed',
                            bmrb_id=bmrb_id, column='pdb_id', value=pdb_id,
                            action='failed',
                        ))
                else:
                    local_counters['already_downloaded'] += 1
            else:
                local_counters['skipped_download'] += 1

            local_counters['selected'] += 1

            return {
                'bmrb_id': bmrb_id, 'pdb_id': pdb_id, 'chain_id': chain_id,
                'resolution': best.get('resolution'), 'r_free': best.get('r_free'),
                'method': best.get('method'), 'reason': reason,
            }, local_counters, local_log

        # Run with thread pool + progress bar
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_one_protein, bid): bid for bid in bmrb_ids}

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Selecting PDBs", unit="protein"):
                row, local_counters, local_log_entries = future.result()
                selection_rows.append(row)
                for k, v in local_counters.items():
                    counters[k] += v
                with _log_lock:
                    for entry in local_log_entries:
                        if isinstance(entry, dict):
                            log.add(**entry)

        # Sort back to original order
        bmrb_order = {bid: i for i, bid in enumerate(bmrb_ids)}
        selection_rows.sort(key=lambda r: bmrb_order.get(r['bmrb_id'], 0))

    # ------------------------------------------------------------------
    # Step 4: Save outputs
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Saving outputs...")

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(output_selection, index=False)
    print(f"  Saved selection: {output_selection}")

    log.add_summary('pdb_selection', 'Total BMRB IDs processed', len(bmrb_ids))
    log.add_summary('pdb_selection', 'PDBs selected', counters['selected'])
    log.add_summary('pdb_selection', 'No PDB mapping found', counters['no_pdb_mapping'])
    log.add_summary('pdb_selection', 'No suitable PDB', counters['no_suitable_pdb'])
    log.add_summary('pdb_selection', 'Found existing PDB files', counters['found_existing'])
    log.add_summary('pdb_selection', 'Downloaded from RCSB', counters['downloaded'])
    log.add_summary('pdb_selection', 'Download failed', counters['download_failed'])
    log.add_summary('pdb_selection', 'Already downloaded', counters['already_downloaded'])
    log.add_summary('pdb_selection', 'RCSB query failures', counters['rcsb_query_failed'])

    log.save(output_log)

    # ------------------------------------------------------------------
    # Step 5: Print comprehensive summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total BMRB IDs:            {len(bmrb_ids):>8,}")
    print(f"  PDB selected:              {counters['selected']:>8,}")
    print(f"  No PDB mapping:            {counters['no_pdb_mapping']:>8,}")
    print(f"  No suitable PDB:           {counters['no_suitable_pdb']:>8,}")
    print(f"  RCSB query failures:       {counters['rcsb_query_failed']:>8,}")

    n_with_pdb = selection_df['pdb_id'].notna().sum()
    n_without_pdb = selection_df['pdb_id'].isna().sum()
    print(f"\n  Proteins with PDB:         {n_with_pdb:>8,}")
    print(f"  Proteins without PDB:      {n_without_pdb:>8,}")

    if not args.skip_download:
        print(f"\n  PDB files:")
        print(f"    Found locally:           {counters['found_existing']:>8,}")
        print(f"    Downloaded:              {counters['downloaded']:>8,}")
        print(f"    Already in output dir:   {counters['already_downloaded']:>8,}")
        print(f"    Download failed:         {counters['download_failed']:>8,}")

    # Method breakdown
    if n_with_pdb > 0:
        print(f"\n  Experimental method breakdown:")
        method_counts = selection_df[selection_df['pdb_id'].notna()]['method'].value_counts()
        for method, count in method_counts.items():
            pct = 100.0 * count / n_with_pdb
            print(f"    {method or 'unknown':30s}: {count:>6,} ({pct:5.1f}%)")

    # Resolution stats for X-ray
    xray_mask = selection_df['method'].str.contains('X-RAY', na=False)
    if xray_mask.any():
        xray_res = selection_df.loc[xray_mask, 'resolution'].dropna().astype(float)
        if len(xray_res) > 0:
            print(f"\n  X-ray resolution statistics:")
            print(f"    Mean:   {xray_res.mean():.2f} A")
            print(f"    Median: {xray_res.median():.2f} A")
            print(f"    Min:    {xray_res.min():.2f} A")
            print(f"    Max:    {xray_res.max():.2f} A")

    # Selection reason breakdown
    print(f"\n  Selection reasons:")
    reason_counts = selection_df['reason'].value_counts()
    for reason, count in reason_counts.head(15).items():
        print(f"    {reason:45s}: {count:>6,}")

    log.print_report()

    print(f"\nOutputs:")
    print(f"  {output_selection}")
    print(f"  {output_log}")
    print(f"  {pdb_dir}/ ({counters['selected']} PDB files)")
    print()


if __name__ == '__main__':
    main()
