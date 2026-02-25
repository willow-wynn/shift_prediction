"""
PDB file parsing and DSSP secondary structure assignment.

Handles:
- Reading PDB files (stop at first MODEL/TER for multi-model)
- Chain filtering
- DSSP wrapper with REMARK 800 cleaning
- Coordinate extraction per residue
"""

import os
import re
import subprocess
import tempfile
import numpy as np
import pandas as pd
from config import AA_3_TO_1, DSSP_PATH


def parse_pdb(pdb_path, chain_id=None):
    """Parse a PDB file and extract per-residue atom coordinates.

    Args:
        pdb_path: Path to PDB file
        chain_id: Chain to extract (None = all chains, '_' = no filter)

    Returns:
        Dict mapping (chain, residue_id) -> {
            'residue_name': str,
            'atoms': {atom_name: np.array([x, y, z])}
        }
    """
    residues = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            record = line[:6].strip()

            # Stop at ENDMDL (take first model only)
            if record == 'ENDMDL':
                break

            if record not in ('ATOM', 'HETATM'):
                continue

            # Parse ATOM/HETATM record
            atom_name = line[12:16].strip()
            alt_loc = line[16].strip()
            res_name = line[17:20].strip()
            chain = line[21].strip()

            # Filter by chain
            if chain_id is not None and chain_id != '_' and chain != chain_id:
                continue

            # Skip alternate conformations (keep 'A' or first)
            if alt_loc and alt_loc not in ('A', '1'):
                continue

            try:
                res_seq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            key = (chain, res_seq)
            if key not in residues:
                residues[key] = {
                    'residue_name': res_name,
                    'atoms': {},
                }

            residues[key]['atoms'][atom_name] = np.array([x, y, z])

    return residues


def clean_pdb_for_dssp(pdb_path, output_path):
    """Clean PDB file for DSSP compatibility.

    Removes problematic REMARK 800 lines and other issues that cause
    DSSP to fail on some PDB files.
    """
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        # Skip problematic REMARK 800 lines
        if line.startswith('REMARK 800'):
            continue
        cleaned.append(line)

    with open(output_path, 'w') as f:
        f.writelines(cleaned)


def run_dssp(pdb_path, dssp_path=None):
    """Run DSSP on a PDB file and parse results.

    Args:
        pdb_path: Path to PDB file
        dssp_path: Path to DSSP executable (default from config)

    Returns:
        Dict mapping (chain, residue_id) -> {
            'secondary_structure': str,
            'phi': float,
            'psi': float,
            'rel_acc': float,
            'nh_o_1_relidx': int, 'nh_o_1_energy': float,
            'o_nh_1_relidx': int, 'o_nh_1_energy': float,
            'nh_o_2_relidx': int, 'nh_o_2_energy': float,
            'o_nh_2_relidx': int, 'o_nh_2_energy': float,
        }
    """
    if dssp_path is None:
        dssp_path = DSSP_PATH

    # Try running DSSP directly first
    try:
        result = subprocess.run(
            [dssp_path, '-i', pdb_path],
            capture_output=True, text=True, timeout=60
        )
        dssp_output = result.stdout
    except Exception:
        dssp_output = None

    # If direct attempt failed or produced no output, try with cleaned PDB
    if not dssp_output:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as tmp:
            tmp_path = tmp.name
        try:
            clean_pdb_for_dssp(pdb_path, tmp_path)
            result = subprocess.run(
                [dssp_path, '-i', tmp_path],
                capture_output=True, text=True, timeout=60
            )
            dssp_output = result.stdout
        except Exception:
            dssp_output = None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not dssp_output:
        return {}

    return _parse_dssp_output(dssp_output)


def _parse_dssp_output(dssp_text):
    """Parse DSSP text output into per-residue data."""
    results = {}
    in_data = False

    for line in dssp_text.split('\n'):
        if '  #  RESIDUE AA' in line:
            in_data = True
            continue

        if not in_data or len(line) < 120:
            continue

        # Check for chain break
        if line[13] == '!':
            continue

        try:
            chain = line[11].strip()
            res_num = int(line[5:10].strip())
            ss = line[16]
            if ss == ' ':
                ss = 'C'  # Coil

            # Accessible surface area
            try:
                acc = float(line[34:38].strip())
            except (ValueError, IndexError):
                acc = 0.0

            # Phi/Psi
            try:
                phi = float(line[103:109].strip())
                psi = float(line[109:115].strip())
            except (ValueError, IndexError):
                phi = 360.0
                psi = 360.0

            # H-bond data (4 pairs: NH->O and O->NH, primary and secondary)
            hbonds = {}
            hbond_specs = [
                ('nh_o_1', 39, 50),
                ('o_nh_1', 50, 61),
                ('nh_o_2', 61, 72),
                ('o_nh_2', 72, 83),
            ]
            for name, start, end in hbond_specs:
                segment = line[start:end]
                try:
                    parts = segment.split(',')
                    relidx = int(parts[0].strip())
                    energy = float(parts[1].strip())
                    hbonds[f'{name}_relidx'] = relidx
                    hbonds[f'{name}_energy'] = energy
                except (ValueError, IndexError):
                    hbonds[f'{name}_relidx'] = 0
                    hbonds[f'{name}_energy'] = 0.0

            # Normalize accessible surface area to relative (0-1)
            # Using max ASA values from Tien et al. 2013
            MAX_ASA = {
                'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167,
                'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
                'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159,
                'S': 155, 'T': 172, 'V': 174, 'W': 285, 'Y': 263,
            }
            aa = line[13]
            max_asa = MAX_ASA.get(aa, 200)
            rel_acc = min(acc / max_asa, 1.0) if max_asa > 0 else 0.0

            results[(chain, res_num)] = {
                'secondary_structure': ss,
                'phi': phi if abs(phi) <= 180 else None,
                'psi': psi if abs(psi) <= 180 else None,
                'rel_acc': rel_acc,
                **hbonds,
            }
        except (ValueError, IndexError):
            continue

    return results


def extract_bfactors(pdb_path, chain_id=None):
    """Extract B-factors per residue from a PDB file.

    Returns:
        Dict mapping (chain, residue_id) -> mean B-factor for CA atom
    """
    bfactors = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if line[:6].strip() == 'ENDMDL':
                break
            if line[:6].strip() != 'ATOM':
                continue

            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue

            chain = line[21].strip()
            if chain_id is not None and chain_id != '_' and chain != chain_id:
                continue

            try:
                res_seq = int(line[22:26].strip())
                bfactor = float(line[60:66].strip())
                bfactors[(chain, res_seq)] = bfactor
            except (ValueError, IndexError):
                continue

    return bfactors


def resolve_pdb_path(pdb_id, search_dirs, chain_id=None, bmrb_id=None):
    """Find a PDB file across multiple directories and naming conventions.

    Args:
        pdb_id: PDB identifier (e.g. '1UBQ')
        search_dirs: List of directories to search
        chain_id: Optional chain ID for chain-suffixed filenames
        bmrb_id: Optional BMRB ID for rcsb_{bmrb_id}.pdb naming

    Returns:
        Path to PDB file, or None if not found
    """
    pdb_upper = pdb_id.upper()
    pdb_lower = pdb_id.lower()

    patterns = [
        f'{pdb_upper}.pdb',
        f'{pdb_lower}.pdb',
    ]
    if chain_id:
        patterns.extend([
            f'{pdb_upper}{chain_id}.pdb',
            f'{pdb_lower}{chain_id}.pdb',
        ])
    if bmrb_id:
        patterns.append(f'rcsb_{bmrb_id}.pdb')

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in patterns:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path

    return None


def lookup_with_chain_fallback(data_dict, chain, res_id):
    """Look up a (chain, res_id) key with empty-chain fallbacks.

    Tries (chain, res_id), then ('', res_id), then (' ', res_id).

    Args:
        data_dict: Dict keyed by (chain, residue_id)
        chain: Chain letter
        res_id: Residue ID

    Returns:
        Value from dict, or None if not found
    """
    for key in [(chain, res_id), ('', res_id), (' ', res_id)]:
        if key in data_dict:
            return data_dict[key]
    return None


def classify_atom(atom_name):
    """Classify a PDB atom name as 'heavy' or 'hydrogen'.

    In PDB format, hydrogen atoms have names starting with 'H' or
    a digit followed by 'H'. This handles standard naming conventions.

    Args:
        atom_name: Atom name string (e.g. 'CA', 'H', 'HB2', '1HG1')

    Returns:
        'hydrogen' if the atom is a hydrogen, 'heavy' otherwise
    """
    name = atom_name.strip()
    if not name:
        return 'heavy'
    # Standard: H atoms start with H (but not HG for mercury in rare cases)
    # In protein PDB files, all H* atoms are hydrogens
    if name[0] == 'H':
        return 'hydrogen'
    # Old-style PDB naming: digit prefix (e.g. 1HG1, 2HB)
    if len(name) >= 2 and name[0].isdigit() and name[1] == 'H':
        return 'hydrogen'
    return 'heavy'
