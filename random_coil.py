"""
Random coil chemical shift tables and correction functions.

Tables from Schwarzinger et al. (2001) J. Am. Chem. Soc. 123, 2970-2978
and Wishart et al. (1995) J. Biomol. NMR 5, 67-81.

Used for:
1. Computing secondary chemical shifts (delta = observed - random_coil)
2. Correcting retrieval transfer between different residue types
"""

# Random coil shifts in ppm at pH ~5-7, 25 degrees C
# Format: {residue_1letter: {atom_type: shift_ppm}}
# Sources: Schwarzinger 2001 (backbone), Wishart 1995 (supplementary)
RC_SHIFTS = {
    'A': {'CA': 52.5, 'CB': 19.1, 'C': 177.8, 'N': 123.8, 'H': 8.24, 'HA': 4.35},
    'R': {'CA': 56.0, 'CB': 30.7, 'C': 176.3, 'N': 120.5, 'H': 8.27, 'HA': 4.38},
    'N': {'CA': 53.1, 'CB': 38.7, 'C': 175.2, 'N': 118.7, 'H': 8.40, 'HA': 4.75},
    'D': {'CA': 54.2, 'CB': 41.1, 'C': 176.3, 'N': 120.4, 'H': 8.41, 'HA': 4.64},
    'C': {'CA': 58.2, 'CB': 28.0, 'C': 174.6, 'N': 118.8, 'H': 8.32, 'HA': 4.56},
    'Q': {'CA': 55.7, 'CB': 29.4, 'C': 176.0, 'N': 119.8, 'H': 8.32, 'HA': 4.37},
    'E': {'CA': 56.6, 'CB': 30.3, 'C': 176.6, 'N': 120.2, 'H': 8.36, 'HA': 4.29},
    'G': {'CA': 45.1, 'CB': None, 'C': 174.9, 'N': 108.8, 'H': 8.33, 'HA': 3.97},
    'H': {'CA': 55.0, 'CB': 29.0, 'C': 174.1, 'N': 118.2, 'H': 8.42, 'HA': 4.63},
    'I': {'CA': 61.1, 'CB': 38.8, 'C': 176.4, 'N': 119.9, 'H': 8.00, 'HA': 4.17},
    'L': {'CA': 55.1, 'CB': 42.4, 'C': 177.6, 'N': 121.8, 'H': 8.16, 'HA': 4.38},
    'K': {'CA': 56.5, 'CB': 33.1, 'C': 176.6, 'N': 120.4, 'H': 8.29, 'HA': 4.36},
    'M': {'CA': 55.4, 'CB': 32.9, 'C': 176.3, 'N': 119.6, 'H': 8.28, 'HA': 4.52},
    'F': {'CA': 57.7, 'CB': 39.6, 'C': 175.8, 'N': 120.3, 'H': 8.30, 'HA': 4.66},
    'P': {'CA': 63.3, 'CB': 32.1, 'C': 177.3, 'N': 137.4, 'H': None, 'HA': 4.44},
    'S': {'CA': 58.3, 'CB': 63.8, 'C': 174.6, 'N': 115.7, 'H': 8.31, 'HA': 4.50},
    'T': {'CA': 61.8, 'CB': 69.8, 'C': 174.7, 'N': 113.6, 'H': 8.15, 'HA': 4.35},
    'W': {'CA': 57.5, 'CB': 29.6, 'C': 176.1, 'N': 121.3, 'H': 8.25, 'HA': 4.70},
    'Y': {'CA': 57.9, 'CB': 38.8, 'C': 175.9, 'N': 120.3, 'H': 8.12, 'HA': 4.60},
    'V': {'CA': 62.2, 'CB': 32.9, 'C': 176.3, 'N': 119.2, 'H': 8.03, 'HA': 4.18},
}


def get_random_coil(residue_1letter, atom_type):
    """Get random coil shift for a residue and atom type.

    Args:
        residue_1letter: Single-letter amino acid code
        atom_type: One of 'CA', 'CB', 'C', 'N', 'H', 'HA'

    Returns:
        Random coil shift in ppm, or None if not available
    """
    if residue_1letter not in RC_SHIFTS:
        return None
    return RC_SHIFTS[residue_1letter].get(atom_type)


def get_secondary_shift(observed_shift, residue_1letter, atom_type):
    """Compute secondary chemical shift (observed - random_coil).

    Args:
        observed_shift: Observed chemical shift in ppm
        residue_1letter: Single-letter amino acid code
        atom_type: Atom type string

    Returns:
        Secondary shift in ppm, or None if RC not available
    """
    rc = get_random_coil(residue_1letter, atom_type)
    if rc is None or observed_shift is None:
        return None
    return observed_shift - rc


def correct_transfer(query_aa, ref_aa, ref_shift, atom_type):
    """Correct a transferred shift for residue-type differences.

    When transferring a chemical shift from a retrieved reference residue
    to the query residue, apply random coil correction:

        corrected = RC[query] + (ref_shift - RC[ref])

    This preserves the secondary shift (structural contribution) while
    adjusting for the intrinsic chemical shift difference between
    amino acid types.

    Args:
        query_aa: Query residue single-letter code
        ref_aa: Reference residue single-letter code
        ref_shift: Reference observed shift in ppm
        atom_type: Atom type ('CA', 'CB', etc.)

    Returns:
        Corrected shift in ppm, or ref_shift if correction not possible
    """
    rc_query = get_random_coil(query_aa, atom_type)
    rc_ref = get_random_coil(ref_aa, atom_type)

    if rc_query is None or rc_ref is None:
        return ref_shift  # Can't correct, return as-is

    return rc_query + (ref_shift - rc_ref)


def build_rc_tensor(residue_order, atom_types):
    """Build a tensor of random coil shifts for batch processing.

    Args:
        residue_order: List of 3-letter residue codes (e.g., STANDARD_RESIDUES)
        atom_types: List of atom type strings

    Returns:
        numpy array of shape (n_residues, n_atoms) with NaN for unavailable
    """
    import numpy as np
    from config import AA_3_TO_1

    n_res = len(residue_order)
    n_atoms = len(atom_types)
    rc_table = np.full((n_res, n_atoms), np.nan, dtype=np.float32)

    for i, res3 in enumerate(residue_order):
        res1 = AA_3_TO_1.get(res3)
        if res1 is None or res1 not in RC_SHIFTS:
            continue
        for j, atom in enumerate(atom_types):
            # Map shift column name to atom type
            atom_key = atom.replace('_shift', '').upper()
            val = RC_SHIFTS[res1].get(atom_key)
            if val is not None:
                rc_table[i, j] = val

    return rc_table
