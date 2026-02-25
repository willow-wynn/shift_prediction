"""
RCSB sequence search API wrapper.

Provides MMseqs2-based sequence search against the PDB via the RCSB
search API. Used for BLAST fallback when no direct BMRB->PDB mapping
exists, and for AlphaFold sequence matching via UniProt.
"""

import json
import urllib.request
import time

from config import RCSB_SEARCH_URL, BLAST_IDENTITY_CUTOFF, AA_3_TO_1


def search_pdb_by_sequence(sequence, identity_cutoff=None, evalue_cutoff=1.0, max_results=10):
    """Search RCSB PDB by protein sequence using MMseqs2.

    Args:
        sequence: Single-letter amino acid sequence
        identity_cutoff: Minimum sequence identity (default from config)
        evalue_cutoff: Maximum E-value (default 1.0)
        max_results: Maximum number of results to return

    Returns:
        List of dicts with keys: pdb_id, entity_id, score, identity
        Empty list on failure.
    """
    if identity_cutoff is None:
        identity_cutoff = BLAST_IDENTITY_CUTOFF

    if not sequence or len(sequence) < 10:
        return []

    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": evalue_cutoff,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "value": sequence,
            }
        },
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results,
            },
            "scoring_strategy": "sequence",
        },
        "return_type": "polymer_entity",
    }

    payload = json.dumps(query).encode('utf-8')

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                RCSB_SEARCH_URL,
                data=payload,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'User-Agent': 'he_lab_pipeline/1.0',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            break
        except urllib.error.HTTPError as e:
            if e.code == 204:
                # No results
                return []
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return []
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return []

    results = []
    for hit in data.get('result_set', []):
        identifier = hit.get('identifier', '')
        # Format is "PDBID_ENTITYID" e.g. "1UBQ_1"
        parts = identifier.split('_')
        if len(parts) >= 2:
            pdb_id = parts[0].upper()
            entity_id = parts[1]
        else:
            pdb_id = identifier.upper()
            entity_id = '1'

        score_info = hit.get('score', 0.0)
        services = hit.get('services', [])
        identity = 0.0
        if services:
            nodes = services[0].get('nodes', [])
            if nodes:
                match_context = nodes[0].get('match_context', [])
                if match_context:
                    identity = match_context[0].get('sequence_identity', 0.0)

        results.append({
            'pdb_id': pdb_id,
            'entity_id': entity_id,
            'score': score_info,
            'identity': identity,
        })

    return results


def extract_sequence_from_shifts(shifts_df):
    """Get single-letter amino acid sequence from a protein's shift DataFrame.

    Args:
        shifts_df: DataFrame for one protein with 'residue_id' and 'residue_code' columns

    Returns:
        Single-letter sequence string, or '' if empty
    """
    sub = shifts_df[['residue_id', 'residue_code']].drop_duplicates().sort_values('residue_id')
    valid = sub['residue_code'].apply(lambda x: str(x).upper() in AA_3_TO_1)
    sub = sub[valid]
    if len(sub) == 0:
        return ''
    return ''.join(AA_3_TO_1.get(str(c).upper(), 'X') for c in sub['residue_code'])
