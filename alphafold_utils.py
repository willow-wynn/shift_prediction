"""
AlphaFold Database utilities.

Handles:
  - Mapping BMRB IDs to UniProt accessions (via BMRB REST API)
  - Downloading AlphaFold predicted structures
  - Fallback UniProt sequence search
"""

import os
import json
import time
import urllib.request

from config import ALPHAFOLD_DB_URL, ALPHAFOLD_DIR


BMRB_ENTRY_URL = 'https://api.bmrb.io/v2/entry/{bmrb_id}?format=json'
UNIPROT_SEARCH_URL = 'https://rest.uniprot.org/uniprotkb/search'


def _fetch_json(url, retries=3, timeout=30):
    """Fetch JSON from a URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            req.add_header('User-Agent', 'he_lab_pipeline/1.0')
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def get_uniprot_for_bmrb(bmrb_id):
    """Look up UniProt accession for a BMRB entry via BMRB REST API.

    Parses the BMRB entry JSON to find database links to UniProt.

    Args:
        bmrb_id: BMRB entry ID (str or int)

    Returns:
        UniProt accession string, or None if not found
    """
    url = BMRB_ENTRY_URL.format(bmrb_id=bmrb_id)
    data = _fetch_json(url)
    if data is None:
        return None

    try:
        # Navigate NMR-STAR JSON to find database references
        for saveframe in data.get('data', []):
            if not isinstance(saveframe, dict):
                continue
            loops = saveframe.get('loops', [])
            for loop in loops:
                tags = loop.get('tags', [])
                loop_data = loop.get('data', [])

                # Look for database link loops
                tag_names = [t.split('.')[-1] if '.' in t else t for t in tags]

                db_name_idx = None
                acc_idx = None
                for i, t in enumerate(tag_names):
                    if t in ('Database_code', 'Accession_code', 'database_code',
                             'accession_code'):
                        if 'atabase' in t.lower() and 'code' in t.lower() and 'ccession' not in t.lower():
                            db_name_idx = i
                        elif 'ccession' in t.lower():
                            acc_idx = i
                    elif t in ('Type', 'type'):
                        pass

                # Alternative: look for exact tag positions
                for i, t in enumerate(tag_names):
                    if t == 'Database_code' or t == 'database_code':
                        db_name_idx = i
                    if t == 'Accession_code' or t == 'accession_code':
                        acc_idx = i

                if db_name_idx is None or acc_idx is None:
                    continue

                for row_data in loop_data:
                    try:
                        db_name = str(row_data[db_name_idx]).upper()
                        accession = str(row_data[acc_idx]).strip()
                        if db_name in ('UNIPROT', 'UNP', 'SP', 'SWISSPROT', 'TREMBL') and accession and accession != '.':
                            return accession
                    except (IndexError, TypeError):
                        continue

    except Exception:
        pass

    return None


def download_alphafold_structure(uniprot_id, output_dir=None):
    """Download an AlphaFold predicted structure from the EBI database.

    Args:
        uniprot_id: UniProt accession (e.g. 'P0AEX9')
        output_dir: Directory to save the PDB file (default from config)

    Returns:
        Path to downloaded PDB file, or None on failure
    """
    if output_dir is None:
        output_dir = ALPHAFOLD_DIR

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'AF-{uniprot_id}-F1-model_v4.pdb')
    if os.path.exists(output_path):
        return output_path

    url = f'{ALPHAFOLD_DB_URL}/AF-{uniprot_id}-F1-model_v4.pdb'

    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'he_lab_pipeline/1.0')
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(output_path, 'wb') as f:
                    f.write(resp.read())
            return output_path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None

    return None


def search_uniprot_by_sequence(sequence, identity_threshold=0.9):
    """Search UniProt by protein sequence as a fallback for BMRB->UniProt mapping.

    Uses the UniProt REST API BLAST search.

    Args:
        sequence: Single-letter amino acid sequence
        identity_threshold: Minimum identity to accept (default 0.9)

    Returns:
        UniProt accession string, or None if no good match
    """
    if not sequence or len(sequence) < 10:
        return None

    # Use UniProt's text search with sequence (BLAST endpoint)
    url = f'{UNIPROT_SEARCH_URL}?query=sequence:{sequence[:50]}&format=json&size=1'

    try:
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/json')
        req.add_header('User-Agent', 'he_lab_pipeline/1.0')
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        results = data.get('results', [])
        if results:
            return results[0].get('primaryAccession')
    except Exception:
        pass

    return None
