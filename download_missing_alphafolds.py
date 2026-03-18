#!/usr/bin/env python3
"""Download missing AlphaFold structures for BMRB entries with UniProt mappings."""
import json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ALPHAFOLD_DIR

AF_MODEL_VERSION = 'model_v6'
AF_URL = 'https://alphafold.ebi.ac.uk/files'

def download_af(uniprot_id, output_dir):
    path = os.path.join(output_dir, f'AF-{uniprot_id}-F1-{AF_MODEL_VERSION}.pdb')
    if os.path.exists(path):
        return path
    url = f'{AF_URL}/AF-{uniprot_id}-F1-{AF_MODEL_VERSION}.pdb'
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'he_lab_pipeline/1.0')
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(path, 'wb') as f:
                    f.write(resp.read())
            return path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < 2: time.sleep(2 ** attempt)
        except Exception:
            if attempt < 2: time.sleep(2 ** attempt)
    return None

def main():
    # Load all mappings
    with open(os.path.join(ALPHAFOLD_DIR, 'bmrb_uniprot_mapping.json')) as f:
        existing = json.load(f)
    new_path = os.path.join(ALPHAFOLD_DIR, 'new_uniprot_mappings.json')
    new = json.load(open(new_path)) if os.path.exists(new_path) else {}
    combined = {**existing, **new}

    # Find UniProt IDs that need downloading
    af_files = set()
    for fn in os.listdir(ALPHAFOLD_DIR):
        if fn.startswith('AF-') and fn.endswith('.pdb'):
            af_files.add(fn.split('-')[1])

    need = set(v for v in combined.values() if v and v not in af_files)
    print(f"UniProt IDs to download: {len(need)}")

    os.makedirs(ALPHAFOLD_DIR, exist_ok=True)
    downloaded = 0
    not_found = 0
    failed = 0

    for i, uid in enumerate(sorted(need)):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(need)} | downloaded={downloaded} not_found={not_found}")
        result = download_af(uid, ALPHAFOLD_DIR)
        if result:
            downloaded += 1
        else:
            not_found += 1
        time.sleep(0.3)

    print(f"\nDone: downloaded={downloaded}, not_found={not_found}, failed={failed}")
    print(f"Total AF files: {len(af_files) + downloaded}")

if __name__ == '__main__':
    main()
