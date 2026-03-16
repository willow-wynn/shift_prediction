"""
AlphaFold Database utilities.

Handles:
  - Mapping BMRB IDs to UniProt accessions (via BMRB REST API)
  - Downloading AlphaFold predicted structures
"""

import os
import json
import time
import urllib.request

from config import ALPHAFOLD_DB_URL, ALPHAFOLD_DIR


BMRB_ENTRY_URL = 'https://api.bmrb.io/v2/entry/{bmrb_id}?format=json'

# UniProt database name variants found in BMRB entries
UNIPROT_DB_NAMES = {'UNIPROT', 'UNP', 'SP', 'SWISSPROT', 'SWISS-PROT', 'TREMBL'}

# AlphaFold model version (v6 as of 2025)
AF_MODEL_VERSION = 'model_v6'


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


def _extract_saveframes(data, bmrb_id):
    """Extract saveframes list from BMRB API JSON response.

    The API returns {bmrb_id: {saveframes: [...]}}. This matches the
    parsing pattern used in 00_fetch_bmrb_shifts.py.
    """
    entry = data.get(str(bmrb_id)) or data.get('data')
    if entry is None:
        for v in data.values():
            if isinstance(v, dict) and 'saveframes' in v:
                entry = v
                break
    if entry is None:
        return []

    saveframes = entry.get('saveframes', []) if isinstance(entry, dict) else entry
    return saveframes if isinstance(saveframes, list) else []


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
        saveframes = _extract_saveframes(data, bmrb_id)

        for saveframe in saveframes:
            if not isinstance(saveframe, dict):
                continue
            loops = saveframe.get('loops', [])
            for loop in loops:
                tags = loop.get('tags', [])
                loop_data = loop.get('data', [])

                # Normalize tag names (strip NMR-STAR prefix like "Entity_db_link.")
                tag_names = [t.split('.')[-1] if '.' in t else t for t in tags]

                # Find database code and accession columns
                # BMRB uses both Database_code/Accession_code and Entry_code
                db_name_idx = None
                acc_idx = None
                for i, t in enumerate(tag_names):
                    if t in ('Database_code', 'database_code'):
                        db_name_idx = i
                    if t in ('Accession_code', 'accession_code',
                             'Entry_code', 'entry_code'):
                        acc_idx = i

                if db_name_idx is None or acc_idx is None:
                    continue

                for row_data in loop_data:
                    try:
                        db_name = str(row_data[db_name_idx]).strip().upper()
                        accession = str(row_data[acc_idx]).strip()
                        if db_name in UNIPROT_DB_NAMES and accession and accession != '.':
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

    output_path = os.path.join(output_dir, f'AF-{uniprot_id}-F1-{AF_MODEL_VERSION}.pdb')
    if os.path.exists(output_path):
        return output_path

    url = f'{ALPHAFOLD_DB_URL}/AF-{uniprot_id}-F1-{AF_MODEL_VERSION}.pdb'

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
