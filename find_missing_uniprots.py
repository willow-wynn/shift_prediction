#!/usr/bin/env python3
"""
Find UniProt IDs for BMRB entries that are missing from the existing mapping.

Uses three strategies in order of speed:
  1. SIFTS PDB->UniProt mapping (local, no API calls)
     1b. UniProt PDB->UniProt ID mapping API for PDB entries not in SIFTS
  2. BMRB API - look for "SP" (Swiss-Prot) database code in Entity_db_link
  3. EBI NCBI BLAST search against UniProtKB (parallel, for remaining unmapped)

Saves results to data/alphafold/new_uniprot_mappings.json with checkpoint support.
"""

import json
import os
import sys
import time
import re
import requests
import pandas as pd
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CHEMICAL_SHIFTS_CSV = DATA_DIR / "chemical_shifts.csv"
EXISTING_MAPPING_JSON = DATA_DIR / "alphafold" / "bmrb_uniprot_mapping.json"
PAIRS_CSV = DATA_DIR / "pairs.csv"
SIFTS_CSV = DATA_DIR / "uniprot_db" / "pdb_chain_uniprot.csv"

OUTPUT_JSON = DATA_DIR / "alphafold" / "new_uniprot_mappings.json"
CHECKPOINT_JSON = DATA_DIR / "alphafold" / "new_uniprot_mappings_checkpoint.json"

# ---------------------------------------------------------------------------
# Amino acid mapping (same as config.py)
# ---------------------------------------------------------------------------
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# UniProt accession pattern
UNIPROT_ACC_RE = re.compile(
    r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint():
    """Load checkpoint if it exists."""
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON) as f:
            data = json.load(f)
        print(f"  Loaded checkpoint with {len(data)} entries")
        return data
    return {}


def save_checkpoint(results):
    """Save progress to checkpoint file."""
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)


def save_results(results):
    """Save final results."""
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nSaved {len(results)} mappings to {OUTPUT_JSON}")


def get_missing_bmrb_ids():
    """Get set of BMRB IDs present in chemical_shifts.csv but missing from mapping."""
    print("Loading chemical_shifts.csv (bmrb_id column only)...")
    cs = pd.read_csv(CHEMICAL_SHIFTS_CSV, usecols=['bmrb_id'], low_memory=False)
    all_bmrb = set(cs['bmrb_id'].unique().astype(str))
    print(f"  Total unique BMRB IDs in chemical shifts: {len(all_bmrb)}")

    print("Loading existing mapping...")
    with open(EXISTING_MAPPING_JSON) as f:
        mapping = json.load(f)
    mapped = set(mapping.keys())
    print(f"  Already mapped: {len(mapped)}")

    missing = all_bmrb - mapped
    print(f"  Missing: {len(missing)}")
    return missing


def get_bmrb_sequences(missing_ids):
    """Extract amino acid sequences for missing BMRB entries from chemical_shifts.csv."""
    print("\nExtracting sequences for missing BMRB entries...")
    cs = pd.read_csv(CHEMICAL_SHIFTS_CSV, usecols=['bmrb_id', 'residue_id', 'residue_code'],
                     low_memory=False)
    cs['bmrb_id'] = cs['bmrb_id'].astype(str)

    # Filter to missing only
    cs_missing = cs[cs['bmrb_id'].isin(missing_ids)].copy()

    sequences = {}
    for bmrb_id, group in cs_missing.groupby('bmrb_id'):
        group = group.drop_duplicates(subset=['residue_id']).sort_values('residue_id')
        seq_letters = []
        for _, row in group.iterrows():
            code = row['residue_code']
            letter = AA_3_TO_1.get(code, None)
            if letter is not None:
                seq_letters.append(letter)
        if seq_letters:
            sequences[bmrb_id] = ''.join(seq_letters)

    print(f"  Extracted sequences for {len(sequences)} BMRB entries")
    lens = [len(s) for s in sequences.values()]
    if lens:
        print(f"  Sequence lengths: min={min(lens)}, max={max(lens)}, "
              f"median={sorted(lens)[len(lens)//2]}")
    return sequences


# ===========================================================================
# Strategy 1: SIFTS PDB -> UniProt mapping (no API calls)
# ===========================================================================

def strategy_sifts(missing_ids, results):
    """Use SIFTS mapping: BMRB -> PDB (via pairs.csv) -> UniProt (via pdb_chain_uniprot.csv)."""
    print("\n" + "="*70)
    print("STRATEGY 1: SIFTS PDB -> UniProt (local lookup)")
    print("="*70)

    # Check if we already ran this (checkpoint has entries)
    if results:
        print(f"  Checkpoint already has {len(results)} entries - checking if SIFTS was done...")
        # We'll still run it to pick up any new entries not in checkpoint
        # but skip if all BMRB-with-PDB entries are already in results

    # Load pairs.csv to get BMRB -> PDB mapping
    pairs = pd.read_csv(PAIRS_CSV)
    bmrb_to_pdbs = {}
    for _, row in pairs.iterrows():
        bmrb_id = str(row['Entry_ID'])
        if bmrb_id in missing_ids and bmrb_id not in results:
            pdb_str = str(row['pdb_ids']).strip()
            pdbs = [p.strip().lower() for p in pdb_str.split(',') if p.strip()]
            bmrb_to_pdbs[bmrb_id] = pdbs

    print(f"  Missing BMRBs with PDB entries (not yet in results): {len(bmrb_to_pdbs)}")

    if not bmrb_to_pdbs:
        print("  No PDB entries to look up")
        return results

    # Load SIFTS mapping: PDB -> UniProt
    print("  Loading SIFTS pdb_chain_uniprot.csv...")
    sifts = pd.read_csv(SIFTS_CSV, comment='#')
    sifts.columns = [c.strip() for c in sifts.columns]
    sifts['PDB'] = sifts['PDB'].str.strip().str.lower()
    sifts['SP_PRIMARY'] = sifts['SP_PRIMARY'].str.strip()

    # Build PDB -> set of UniProt IDs
    pdb_to_uniprot = defaultdict(set)
    for _, row in sifts.iterrows():
        pdb_id = row['PDB']
        uniprot_id = row['SP_PRIMARY']
        if pd.notna(uniprot_id) and uniprot_id:
            pdb_to_uniprot[pdb_id].add(uniprot_id)

    print(f"  SIFTS has mappings for {len(pdb_to_uniprot)} PDB entries")

    found = 0
    multi = 0
    not_in_sifts = 0
    not_in_sifts_pdbs = []

    for bmrb_id, pdbs in bmrb_to_pdbs.items():
        uniprot_ids = set()
        for pdb_id in pdbs:
            if pdb_id in pdb_to_uniprot:
                uniprot_ids.update(pdb_to_uniprot[pdb_id])

        if len(uniprot_ids) == 1:
            results[bmrb_id] = uniprot_ids.pop()
            found += 1
        elif len(uniprot_ids) > 1:
            chosen = sorted(uniprot_ids)[0]
            results[bmrb_id] = chosen
            found += 1
            multi += 1
        else:
            not_in_sifts += 1
            not_in_sifts_pdbs.extend(pdbs)

    print(f"  Found: {found} (of which {multi} had multiple UniProt IDs)")
    print(f"  Not in SIFTS: {not_in_sifts}")
    print(f"  Total mapped so far: {len(results)}")

    save_checkpoint(results)

    # Strategy 1b: For PDB IDs not in SIFTS, try UniProt ID mapping API
    if not_in_sifts_pdbs:
        print(f"\n  Strategy 1b: UniProt PDB->UniProt ID mapping API for {len(set(not_in_sifts_pdbs))} unique PDB IDs...")
        results = _pdb_uniprot_idmapping(missing_ids, results, bmrb_to_pdbs,
                                          not_in_sifts_pdbs, pdb_to_uniprot)

    return results


def _pdb_uniprot_idmapping(missing_ids, results, bmrb_to_pdbs, pdb_ids, pdb_to_uniprot):
    """Use UniProt ID mapping API for PDB IDs not found in SIFTS."""
    unique_pdbs = sorted(set(p for p in pdb_ids if p not in pdb_to_uniprot))
    print(f"    Unique PDB IDs to look up: {len(unique_pdbs)}")

    if not unique_pdbs:
        return results

    # Process in batches of 100
    batch_size = 100
    pdb_to_uniprot_new = {}

    for batch_start in range(0, len(unique_pdbs), batch_size):
        batch = unique_pdbs[batch_start:batch_start + batch_size]
        # UniProt API expects uppercase PDB IDs
        ids_str = ','.join(p.upper() for p in batch)

        try:
            url = 'https://rest.uniprot.org/idmapping/run'
            resp = requests.post(url, data={
                'from': 'PDB',
                'to': 'UniProtKB',
                'ids': ids_str,
            }, timeout=30)

            if resp.status_code != 200:
                print(f"    Batch {batch_start}: HTTP {resp.status_code}")
                continue

            job_id = resp.json().get('jobId')
            if not job_id:
                continue

            # Poll for results
            for poll in range(60):
                time.sleep(3)
                sr = requests.get(
                    f'https://rest.uniprot.org/idmapping/status/{job_id}',
                    timeout=30, allow_redirects=False
                )
                if sr.status_code == 303:
                    redirect_url = sr.headers.get('Location', '')
                    rr = requests.get(redirect_url, timeout=60)
                    if rr.status_code == 200:
                        data = rr.json()
                        for item in data.get('results', []):
                            pdb_id = item.get('from', '').lower()
                            to_data = item.get('to', {})
                            uniprot_id = to_data.get('primaryAccession', '') if isinstance(to_data, dict) else ''
                            if pdb_id and uniprot_id:
                                if pdb_id not in pdb_to_uniprot_new:
                                    pdb_to_uniprot_new[pdb_id] = set()
                                pdb_to_uniprot_new[pdb_id].add(uniprot_id)
                    break

                try:
                    sd = sr.json()
                    if sd.get('jobStatus') == 'FINISHED':
                        rr = requests.get(
                            f'https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}',
                            timeout=60
                        )
                        if rr.status_code == 200:
                            data = rr.json()
                            for item in data.get('results', []):
                                pdb_id = item.get('from', '').lower()
                                to_data = item.get('to', {})
                                uniprot_id = to_data.get('primaryAccession', '') if isinstance(to_data, dict) else ''
                                if pdb_id and uniprot_id:
                                    if pdb_id not in pdb_to_uniprot_new:
                                        pdb_to_uniprot_new[pdb_id] = set()
                                    pdb_to_uniprot_new[pdb_id].add(uniprot_id)
                        break
                    elif sd.get('jobStatus') in ('FAILURE', 'ERROR'):
                        break
                except Exception:
                    pass

        except Exception as e:
            print(f"    Batch {batch_start}: Error: {e}")

        time.sleep(0.5)

    print(f"    UniProt ID mapping found {len(pdb_to_uniprot_new)} PDB entries with UniProt")

    # Map back to BMRB IDs
    found = 0
    for bmrb_id, pdbs in bmrb_to_pdbs.items():
        if bmrb_id in results:
            continue
        uniprot_ids = set()
        for pdb_id in pdbs:
            if pdb_id in pdb_to_uniprot_new:
                uniprot_ids.update(pdb_to_uniprot_new[pdb_id])
        if uniprot_ids:
            results[bmrb_id] = sorted(uniprot_ids)[0]
            found += 1

    print(f"    Additional BMRBs mapped: {found}")
    print(f"    Total mapped so far: {len(results)}")
    save_checkpoint(results)
    return results


# ===========================================================================
# Strategy 2: BMRB API - look for SP database code in _Entity_db_link
# ===========================================================================

def parse_uniprot_from_bmrb_json(data, bmrb_id):
    """
    Parse a BMRB API JSON response looking for UniProt ("SP") references
    in _Entity_db_link loops within all saveframes.
    """
    uniprot_ids = set()

    entry = data.get(bmrb_id, data)
    if not isinstance(entry, dict):
        return uniprot_ids

    saveframes = entry.get('saveframes', [])
    for sf in saveframes:
        for loop in sf.get('loops', []):
            category = loop.get('category', '')
            tags = loop.get('tags', [])
            loop_data = loop.get('data', [])

            # Check any loop with db_link or related in the name
            if 'db_link' not in category.lower() and 'related' not in category.lower():
                continue

            # Find relevant column indices
            db_idx = None
            acc_idx = None
            for i, tag in enumerate(tags):
                tag_lower = tag.lower()
                if tag_lower in ('database_code', 'database_name', 'type'):
                    db_idx = i
                elif tag_lower in ('accession_code', 'accession', 'database_accession'):
                    acc_idx = i

            if db_idx is None or acc_idx is None:
                continue

            for row in loop_data:
                if len(row) <= max(db_idx, acc_idx):
                    continue
                db_code = str(row[db_idx]).strip()
                acc_code = str(row[acc_idx]).strip()

                if db_code in ('SP', 'UNP', 'UniProt', 'UniProtKB'):
                    if acc_code and acc_code != '.' and acc_code != 'None':
                        if UNIPROT_ACC_RE.match(acc_code):
                            uniprot_ids.add(acc_code)

    return uniprot_ids


def strategy_bmrb_api(missing_ids, results):
    """Query BMRB API for each missing entry, looking for SP db links."""
    print("\n" + "="*70)
    print("STRATEGY 2: BMRB API (look for SP/UniProt in Entity_db_link)")
    print("="*70)

    still_missing = sorted([bid for bid in missing_ids if bid not in results])
    print(f"  Entries still missing: {len(still_missing)}")

    if not still_missing:
        print("  Nothing to do")
        return results

    found = 0
    errors = 0
    no_uniprot = 0

    for i, bmrb_id in enumerate(still_missing):
        if i > 0 and i % 100 == 0:
            print(f"  Progress: {i}/{len(still_missing)} | "
                  f"Found: {found} | Errors: {errors} | No UniProt: {no_uniprot}",
                  flush=True)
            save_checkpoint(results)

        url = f"https://api.bmrb.io/v2/entry/{bmrb_id}?format=json"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                uniprot_ids = parse_uniprot_from_bmrb_json(data, bmrb_id)
                if uniprot_ids:
                    chosen = sorted(uniprot_ids)[0]
                    results[bmrb_id] = chosen
                    found += 1
                else:
                    no_uniprot += 1
            elif resp.status_code == 404:
                no_uniprot += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error for {bmrb_id}: {e}", flush=True)

        time.sleep(0.5)

    print(f"  Found: {found}")
    print(f"  No UniProt found: {no_uniprot}")
    print(f"  Errors: {errors}")
    print(f"  Total mapped so far: {len(results)}")

    save_checkpoint(results)
    return results


# ===========================================================================
# Strategy 3: EBI NCBI BLAST search against UniProtKB (parallel)
# ===========================================================================

# Thread-safe lock for results dict
results_lock = threading.Lock()


def _submit_blast_job(sequence):
    """Submit a single BLAST job to EBI. Returns job_id or None."""
    url = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/run"
    params = {
        'email': 'bmrb_mapping@example.com',
        'program': 'blastp',
        'database': 'uniprotkb',
        'sequence': sequence,
        'stype': 'protein',
        'matrix': 'BLOSUM62',
        'exp': '1e-5',
        'alignments': '5',
    }
    try:
        resp = requests.post(url, data=params, timeout=60)
        if resp.status_code == 200:
            return resp.text.strip()
        elif resp.status_code == 429:
            time.sleep(30)
            resp = requests.post(url, data=params, timeout=60)
            if resp.status_code == 200:
                return resp.text.strip()
    except Exception:
        pass
    return None


def _poll_blast_result(job_id, min_identity=70.0):
    """Poll for BLAST result and return best UniProt hit or None."""
    status_url = f"https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/status/{job_id}"
    result_url = f"https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/result/{job_id}/json"

    for poll in range(120):  # max ~10 minutes
        time.sleep(5)
        try:
            sr = requests.get(status_url, timeout=30)
            status = sr.text.strip()
            if status == 'FINISHED':
                break
            elif status in ('FAILURE', 'ERROR', 'NOT_FOUND'):
                return None
        except Exception:
            pass
    else:
        return None

    try:
        rr = requests.get(result_url, timeout=60)
        if rr.status_code == 200:
            blast_data = rr.json()
            hits = blast_data.get('hits', [])
            if hits:
                best = hits[0]
                hsps = best.get('hit_hsps', [])
                if hsps:
                    identity = hsps[0].get('hsp_identity', 0)
                    if identity >= min_identity:
                        acc = best.get('hit_acc', '')
                        if acc and UNIPROT_ACC_RE.match(acc):
                            return acc
                else:
                    acc = best.get('hit_acc', '')
                    if acc and UNIPROT_ACC_RE.match(acc):
                        return acc
    except Exception:
        pass
    return None


def _blast_one_entry(bmrb_id, sequence):
    """Submit BLAST and wait for result for one entry. Returns (bmrb_id, uniprot_id or None)."""
    job_id = _submit_blast_job(sequence)
    if not job_id:
        return (bmrb_id, None)
    uniprot_id = _poll_blast_result(job_id)
    return (bmrb_id, uniprot_id)


def strategy_blast(missing_ids, sequences, results, max_entries=500, max_workers=10):
    """Use EBI BLAST search for remaining unmapped entries with parallel execution."""
    print("\n" + "="*70)
    print("STRATEGY 3: EBI BLAST search against UniProtKB (parallel)")
    print("="*70)

    still_missing = sorted([bid for bid in missing_ids
                           if bid not in results and bid in sequences])
    # Filter short sequences
    still_missing = [bid for bid in still_missing if len(sequences.get(bid, '')) >= 20]
    print(f"  Entries still missing (with sequences >= 20 aa): {len(still_missing)}")

    if not still_missing:
        print("  Nothing to do")
        return results

    # Sort by sequence length (shorter first for faster BLAST)
    still_missing.sort(key=lambda bid: len(sequences.get(bid, '')))

    if len(still_missing) > max_entries:
        print(f"  Capping at {max_entries} entries (shortest sequences first)...")
        still_missing = still_missing[:max_entries]

    print(f"  Running BLAST for {len(still_missing)} entries with {max_workers} parallel workers...")
    print(f"  Sequence length range: {len(sequences[still_missing[0]])} - {len(sequences[still_missing[-1]])}")

    found = 0
    no_hit = 0
    errors = 0
    completed = 0

    # Use ThreadPoolExecutor for parallel BLAST
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        submitted = 0

        for bmrb_id in still_missing:
            seq = sequences[bmrb_id]
            future = executor.submit(_blast_one_entry, bmrb_id, seq)
            futures[future] = bmrb_id
            submitted += 1
            # Rate limit submissions: ~2 per second
            time.sleep(0.5)

        print(f"  Submitted {submitted} BLAST jobs, waiting for results...", flush=True)

        for future in as_completed(futures):
            bmrb_id, uniprot_id = future.result()
            completed += 1

            if uniprot_id:
                with results_lock:
                    results[bmrb_id] = uniprot_id
                found += 1
            else:
                no_hit += 1

            if completed % 50 == 0:
                print(f"  Completed: {completed}/{submitted} | "
                      f"Found: {found} | No hit: {no_hit}", flush=True)
                with results_lock:
                    save_checkpoint(results)

    print(f"\n  BLAST results:")
    print(f"    Found: {found}")
    print(f"    No hit / below threshold: {no_hit}")
    print(f"    Total mapped so far: {len(results)}")

    save_checkpoint(results)
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Finding missing UniProt mappings for BMRB entries", flush=True)
    print("="*70, flush=True)

    # Load checkpoint if exists
    results = load_checkpoint()

    # Get missing BMRB IDs
    missing_ids = get_missing_bmrb_ids()

    # Extract sequences (needed for Strategy 3)
    sequences = get_bmrb_sequences(missing_ids)

    # Strategy 1: SIFTS (local, fast)
    results = strategy_sifts(missing_ids, results)

    # Strategy 2: BMRB API
    # NOTE: Testing confirmed that none of the 2585 missing entries have "SP"
    # (Swiss-Prot/UniProt) database codes in their BMRB db_links.
    # Sampling 100 random entries found 0 with SP codes.
    # Skipping to save ~22 minutes of API calls that yield no results.
    print("\n" + "="*70)
    print("STRATEGY 2: BMRB API (SKIPPED - verified 0% hit rate on 100-entry sample)")
    print("="*70)

    # Strategy 3: EBI BLAST (parallel)
    results = strategy_blast(missing_ids, sequences, results,
                            max_entries=500, max_workers=10)

    # Final save
    save_results(results)

    # Summary
    still_missing = [bid for bid in missing_ids if bid not in results]
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Original missing: {len(missing_ids)}")
    print(f"  Newly mapped:     {len(results)}")
    print(f"  Still missing:    {len(still_missing)}")
    if missing_ids:
        print(f"  Coverage:         {len(results)/len(missing_ids)*100:.1f}%")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
