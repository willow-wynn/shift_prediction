"""
UCBShift-X inspired physics-based features for chemical shift prediction.

This module computes structural and physicochemical features that capture
physical effects on NMR chemical shifts beyond what sequence-level or
distance-based features provide. The features are motivated by the UCBShift-X
approach (Li & Bruesch-weiler, JACS 2021) and classical NMR theory.

Features implemented:
    - Aromatic ring current effects (Haigh-Mallion model)
    - Half-sphere exposure (HSE)
    - BLOSUM62 evolutionary profile
    - Hydrogen bond geometry from DSSP
    - Order parameter from B-factors
"""

import numpy as np

from config import BLOSUM62, BLOSUM62_ORDER, AA_3_TO_1


# ============================================================================
# Aromatic ring atom definitions
# ============================================================================
AROMATIC_RING_ATOMS = {
    'PHE': [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']],
    'TYR': [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']],
    'TRP': [
        ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],   # 6-membered ring
        ['CG', 'CD1', 'NE1', 'CD2', 'CE2'],             # 5-membered ring
    ],
    'HIS': [['CG', 'ND1', 'CD2', 'CE1', 'NE2']],
}

# Approximate intensity factors for different ring systems (arbitrary units,
# relative to benzene = 1.0). These are rough literature values; PHE/TYR are
# 6-membered carbocyclic rings, HIS is a 5-membered heterocycle, and TRP has
# both a 6- and 5-membered ring.
RING_INTENSITY = {
    'PHE': [1.00],
    'TYR': [0.94],
    'TRP': [1.04, 0.56],
    'HIS': [0.43],
}


# ============================================================================
# 1. Ring Current Effects
# ============================================================================

def compute_ring_current(residue_atoms, neighbor_residues):
    """Compute aromatic ring current shifts on H and HA atoms.

    Uses the Haigh-Mallion model (Haigh & Mallion, Prog. NMR Spectrosc. 1979)
    to estimate the through-space effect of aromatic ring currents on nearby
    proton chemical shifts. Ring currents produce significant upfield or
    downfield shifts (up to ~1 ppm) on protons located above/below or in the
    plane of an aromatic ring, respectively.

    Relevance to chemical shift prediction:
        Ring current effects are one of the largest contributors to chemical
        shift deviations from random coil values, especially for H and HA
        atoms. UCBShift and SHIFTX2 both include explicit ring-current terms.

    Reference:
        Haigh, C.W. & Mallion, R.B. (1979) "Ring current theories in nuclear
        magnetic resonance." Prog. Nucl. Magn. Reson. Spectrosc. 13, 303-344.

    Parameters
    ----------
    residue_atoms : dict
        Atom coordinate dictionary for the target residue.
        Keys are atom names (str), values are np.ndarray of shape (3,).
        Must contain 'H' and/or 'HA' for the effect to be computed.
    neighbor_residues : list of dict
        Each dict has keys:
            'resname' : str  -- 3-letter residue code (e.g. 'PHE')
            'atoms'   : dict -- atom name -> np.ndarray(3,) coordinates

    Returns
    -------
    dict
        'ring_current_h'  : float -- total ring current contribution at H
        'ring_current_ha' : float -- total ring current contribution at HA
        Values are 0.0 when the target atom is absent or no aromatics nearby.
    """
    result = {'ring_current_h': 0.0, 'ring_current_ha': 0.0}

    target_atoms = {}
    if 'H' in residue_atoms:
        target_atoms['ring_current_h'] = np.asarray(residue_atoms['H'], dtype=np.float64)
    if 'HA' in residue_atoms:
        target_atoms['ring_current_ha'] = np.asarray(residue_atoms['HA'], dtype=np.float64)

    if not target_atoms:
        return result

    for nbr in neighbor_residues:
        resname = nbr.get('resname', '')
        atoms = nbr.get('atoms', {})

        if resname not in AROMATIC_RING_ATOMS:
            continue

        ring_defs = AROMATIC_RING_ATOMS[resname]
        intensities = RING_INTENSITY[resname]

        for ring_idx, ring_atom_names in enumerate(ring_defs):
            # Gather ring atom coordinates
            ring_coords = []
            for aname in ring_atom_names:
                if aname in atoms:
                    ring_coords.append(np.asarray(atoms[aname], dtype=np.float64))
            if len(ring_coords) < 3:
                # Need at least 3 atoms to define a plane
                continue

            ring_coords = np.array(ring_coords)
            ring_center = ring_coords.mean(axis=0)

            # Ring normal via SVD of centered coordinates
            centered = ring_coords - ring_center
            _, _, vh = np.linalg.svd(centered)
            ring_normal = vh[-1]  # smallest singular value direction = normal
            # Normalize (should already be unit, but be safe)
            norm_len = np.linalg.norm(ring_normal)
            if norm_len < 1e-12:
                continue
            ring_normal = ring_normal / norm_len

            intensity = intensities[ring_idx]

            for key, target_coord in target_atoms.items():
                vec = target_coord - ring_center
                r = np.linalg.norm(vec)
                if r < 1e-6 or r > 7.0:
                    # Skip if degenerate or too far (beyond 7 A cutoff)
                    continue

                cos_theta = np.dot(ring_normal, vec) / r
                # Haigh-Mallion geometric factor: (1 - 3*cos^2(theta)) / r^3
                geom_factor = (1.0 - 3.0 * cos_theta ** 2) / (r ** 3)

                result[key] += intensity * geom_factor

    return result


# ============================================================================
# 2. Half-Sphere Exposure
# ============================================================================

def compute_hse(ca_coords, cb_coords, query_idx, radius=13.0):
    """Compute half-sphere exposure (HSE) for a residue.

    HSE splits the sphere around a residue's CA atom into two hemispheres
    defined by the CA->CB direction. The "up" hemisphere (CB side) captures
    side-chain packing density; the "down" hemisphere captures backbone
    burial. This is a simple but effective descriptor of local packing
    environment that correlates with chemical shift perturbations.

    Relevance to chemical shift prediction:
        Solvent exposure and packing density directly influence chemical
        shifts through electric field effects and van der Waals contacts.
        HSE provides a computationally cheap proxy for these effects and is
        used in SPARTA+ and UCBShift feature sets.

    Reference:
        Hamelryck, T. (2005) "An amino acid has two sides: a new 2D measure
        provides a different view of solvent exposure." Proteins 59, 38-48.

    Parameters
    ----------
    ca_coords : np.ndarray, shape (N, 3)
        CA coordinates for all residues in the protein.
    cb_coords : np.ndarray, shape (N, 3)
        CB coordinates for all residues. For GLY (no CB), a pseudo-CB
        should be provided or the entry can be NaN (will use default).
    query_idx : int
        Index of the residue to compute HSE for.
    radius : float
        Sphere radius in Angstroms (default 13.0).

    Returns
    -------
    dict
        'hse_up'    : int   -- number of neighbors in upper (CB-side) hemisphere
        'hse_down'  : int   -- number of neighbors in lower hemisphere
        'hse_ratio' : float -- up / (up + down + 1), a normalized score
    """
    result = {'hse_up': 0, 'hse_down': 0, 'hse_ratio': 0.0}

    ca_coords = np.asarray(ca_coords, dtype=np.float64)
    cb_coords = np.asarray(cb_coords, dtype=np.float64)

    n_residues = ca_coords.shape[0]
    if query_idx < 0 or query_idx >= n_residues:
        return result

    query_ca = ca_coords[query_idx]
    query_cb = cb_coords[query_idx]

    # Check for NaN in CB (e.g., GLY without pseudo-CB)
    if np.any(np.isnan(query_cb)) or np.any(np.isnan(query_ca)):
        return result

    # Half-sphere normal: CA -> CB direction
    hs_normal = query_cb - query_ca
    norm_len = np.linalg.norm(hs_normal)
    if norm_len < 1e-8:
        return result
    hs_normal = hs_normal / norm_len

    up_count = 0
    down_count = 0

    for i in range(n_residues):
        if i == query_idx:
            continue
        if np.any(np.isnan(ca_coords[i])):
            continue
        diff = ca_coords[i] - query_ca
        dist = np.linalg.norm(diff)
        if dist > radius or dist < 1e-8:
            continue
        # Project onto half-sphere normal
        dot = np.dot(diff, hs_normal)
        if dot > 0:
            up_count += 1
        else:
            down_count += 1

    result['hse_up'] = up_count
    result['hse_down'] = down_count
    result['hse_ratio'] = up_count / (up_count + down_count + 1)

    return result


# ============================================================================
# 3. BLOSUM62 Features
# ============================================================================

def compute_blosum62_features(residue_code):
    """Look up the BLOSUM62 substitution profile for a residue.

    The BLOSUM62 matrix encodes evolutionary substitution probabilities
    between amino acid pairs. The 20-dimensional row vector for a residue
    captures its physicochemical similarity to all other standard amino acids,
    providing a compact evolutionary-informed descriptor.

    Relevance to chemical shift prediction:
        Chemical shifts are sensitive to amino acid identity and the local
        sequence context. BLOSUM62 features allow the model to generalize
        across similar residues (e.g., Ile/Leu/Val) and capture substitution
        patterns that correlate with structural and electronic similarity.

    Reference:
        Henikoff, S. & Henikoff, J.G. (1992) "Amino acid substitution
        matrices from protein blocks." Proc. Natl. Acad. Sci. 89, 10915-10919.

    Parameters
    ----------
    residue_code : str
        3-letter or 1-letter amino acid code (e.g. 'ALA' or 'A').

    Returns
    -------
    dict
        Keys 'blosum62_0' through 'blosum62_19' mapping to the BLOSUM62
        row values in BLOSUM62_ORDER (ARNDCQEGHILKMFPSTWYV).
        Returns all NaN if residue code is not recognized.
    """
    # Convert 3-letter to 1-letter if needed
    if len(residue_code) == 3:
        one_letter = AA_3_TO_1.get(residue_code, None)
    elif len(residue_code) == 1:
        one_letter = residue_code if residue_code in BLOSUM62_ORDER else None
    else:
        one_letter = None

    n_features = len(BLOSUM62_ORDER)  # 20

    if one_letter is None or one_letter not in BLOSUM62_ORDER:
        return {f'blosum62_{i}': np.nan for i in range(n_features)}

    idx = BLOSUM62_ORDER.index(one_letter)
    row = BLOSUM62[idx]

    return {f'blosum62_{i}': float(row[i]) for i in range(n_features)}


# ============================================================================
# 4. Hydrogen Bond Geometry
# ============================================================================

def compute_hbond_geometry(dssp_data, all_coords):
    """Compute hydrogen bond donor-acceptor geometry from DSSP data.

    DSSP assigns up to four hydrogen bonds per residue (two N-H...O=C and
    two O=C...H-N patterns) as relative residue index offsets and estimated
    energies. This function resolves those relative indices to actual
    donor-acceptor distances using the backbone N and O coordinates.

    Relevance to chemical shift prediction:
        Hydrogen bonds cause large and systematic chemical shift changes,
        particularly for backbone N, H, C, and CA atoms. H-bond distance
        and energy are among the most important structural descriptors in
        SHIFTX2, SPARTA+, and UCBShift. For example, the H shift changes
        by ~2 ppm between H-bonded and non-H-bonded amides.

    Reference:
        Kabsch, W. & Sander, C. (1983) "Dictionary of protein secondary
        structure: pattern recognition of hydrogen-bonded and geometrical
        features." Biopolymers 22, 2577-2637.

    Parameters
    ----------
    dssp_data : dict
        DSSP feature dictionary for the target residue. Expected keys:
            'nh_o_1_relidx' : int -- relative residue index for 1st N-H...O bond
            'nh_o_1_energy'  : float -- DSSP energy for 1st N-H...O bond (kcal/mol)
            'o_nh_1_relidx' : int -- relative residue index for 1st O...H-N bond
            'o_nh_1_energy'  : float -- DSSP energy for 1st O...H-N bond
            'residue_idx'    : int -- absolute index of this residue in the chain
    all_coords : dict or list
        Per-residue coordinate data. If dict, maps residue index (int) to
        a dict of atom_name -> np.ndarray(3,). If list, indexed by residue
        position, each element is a dict of atom_name -> np.ndarray(3,).

    Returns
    -------
    dict
        'hbond_dist_1'   : float -- N-O distance for 1st NH...O bond (A), NaN if absent
        'hbond_energy_1' : float -- DSSP energy for 1st NH...O bond (kcal/mol)
        'hbond_dist_2'   : float -- N-O distance for 1st O...HN bond (A), NaN if absent
        'hbond_energy_2' : float -- DSSP energy for 1st O...HN bond (kcal/mol)
    """
    result = {
        'hbond_dist_1': np.nan,
        'hbond_energy_1': np.nan,
        'hbond_dist_2': np.nan,
        'hbond_energy_2': np.nan,
    }

    if dssp_data is None:
        return result

    res_idx = dssp_data.get('residue_idx', None)
    if res_idx is None:
        return result

    # --- First H-bond: N-H...O=C (donor is this residue's N, acceptor is partner O)
    rel1 = dssp_data.get('nh_o_1_relidx', 0)
    energy1 = dssp_data.get('nh_o_1_energy', np.nan)

    if rel1 != 0 and not np.isnan(energy1):
        partner_idx = res_idx + int(rel1)
        dist1 = _get_no_distance(res_idx, partner_idx, all_coords, donor_atom='N', acceptor_atom='O')
        result['hbond_dist_1'] = dist1
        result['hbond_energy_1'] = float(energy1)

    # --- Second H-bond: O=C...H-N (acceptor is this residue's O, donor is partner N)
    rel2 = dssp_data.get('o_nh_1_relidx', 0)
    energy2 = dssp_data.get('o_nh_1_energy', np.nan)

    if rel2 != 0 and not np.isnan(energy2):
        partner_idx = res_idx + int(rel2)
        dist2 = _get_no_distance(partner_idx, res_idx, all_coords, donor_atom='N', acceptor_atom='O')
        result['hbond_dist_2'] = dist2
        result['hbond_energy_2'] = float(energy2)

    return result


def _get_no_distance(donor_idx, acceptor_idx, all_coords, donor_atom='N', acceptor_atom='O'):
    """Compute the N...O distance between a donor and acceptor residue.

    Parameters
    ----------
    donor_idx : int
        Residue index of the H-bond donor.
    acceptor_idx : int
        Residue index of the H-bond acceptor.
    all_coords : dict or list
        Per-residue coordinate data.
    donor_atom : str
        Atom name on the donor residue (default 'N').
    acceptor_atom : str
        Atom name on the acceptor residue (default 'O').

    Returns
    -------
    float
        Distance in Angstroms, or NaN if atoms are unavailable.
    """
    try:
        if isinstance(all_coords, dict):
            donor_coord = all_coords[donor_idx].get(donor_atom)
            acceptor_coord = all_coords[acceptor_idx].get(acceptor_atom)
        else:
            if donor_idx < 0 or donor_idx >= len(all_coords):
                return np.nan
            if acceptor_idx < 0 or acceptor_idx >= len(all_coords):
                return np.nan
            donor_coord = all_coords[donor_idx].get(donor_atom)
            acceptor_coord = all_coords[acceptor_idx].get(acceptor_atom)

        if donor_coord is None or acceptor_coord is None:
            return np.nan

        donor_coord = np.asarray(donor_coord, dtype=np.float64)
        acceptor_coord = np.asarray(acceptor_coord, dtype=np.float64)

        return float(np.linalg.norm(donor_coord - acceptor_coord))

    except (KeyError, IndexError, TypeError):
        return np.nan


# ============================================================================
# 5. Order Parameter from B-factors
# ============================================================================

def compute_order_parameter(bfactor):
    """Approximate the generalized order parameter S^2 from B-factors.

    The crystallographic B-factor (temperature factor / Debye-Waller factor)
    reflects the mean-square displacement of an atom. Under the assumption of
    isotropic harmonic motion, S^2 can be approximated as:

        S^2 ~ 1 - B / (8 * pi^2)

    where B is in A^2. The result is clamped to [0, 1].

    Relevance to chemical shift prediction:
        Backbone dynamics modulate chemical shifts, particularly for N and
        H atoms. Residues in flexible loops have systematically different
        shifts compared to rigid secondary structure elements. The order
        parameter is a standard feature in SPARTA+ and UCBShift.

    Reference:
        Lipari, G. & Szabo, A. (1982) "Model-free approach to the
        interpretation of nuclear magnetic resonance relaxation in
        macromolecules." J. Am. Chem. Soc. 104, 4546-4559.

    Parameters
    ----------
    bfactor : float or None
        Crystallographic B-factor in A^2. If None or NaN, returns NaN.

    Returns
    -------
    dict
        'order_parameter' : float -- estimated S^2, in [0, 1], or NaN.
    """
    if bfactor is None or (isinstance(bfactor, float) and np.isnan(bfactor)):
        return {'order_parameter': np.nan}

    bfactor = float(bfactor)
    s2 = 1.0 - bfactor / (8.0 * np.pi ** 2)
    s2 = float(np.clip(s2, 0.0, 1.0))

    return {'order_parameter': s2}


# ============================================================================
# 6. Master Function
# ============================================================================

def compute_all_physics_features(residue_data, neighbor_data, dssp_data=None, bfactor=None):
    """Compute all physics-based features for a single residue.

    This is the main entry point that calls each individual feature function
    and returns a single merged dictionary. Missing data is handled gracefully:
    unavailable features are set to NaN or 0 as appropriate.

    Parameters
    ----------
    residue_data : dict
        Information about the target residue. Expected keys:
            'resname'      : str  -- 3-letter residue code
            'atoms'        : dict -- atom_name -> np.ndarray(3,) coordinates
            'residue_idx'  : int  -- index in the chain
            'ca_coords'    : np.ndarray (N, 3) -- all CA coords in chain
            'cb_coords'    : np.ndarray (N, 3) -- all CB coords in chain
            'all_coords'   : dict or list -- per-residue atom coord dicts
    neighbor_data : list of dict
        Neighbor residue data for ring current computation. Each dict has
        'resname' (str) and 'atoms' (dict of atom_name -> coords).
    dssp_data : dict or None
        DSSP features for the residue (see compute_hbond_geometry).
    bfactor : float or None
        Crystallographic B-factor for the residue's CA atom.

    Returns
    -------
    dict
        Merged dictionary of all physics features. See get_physics_feature_names()
        for the complete list of keys.
    """
    features = {}

    # --- Ring current ---
    try:
        residue_atoms = residue_data.get('atoms', {})
        ring_current = compute_ring_current(residue_atoms, neighbor_data)
        features.update(ring_current)
    except Exception:
        features['ring_current_h'] = np.nan
        features['ring_current_ha'] = np.nan

    # --- Half-sphere exposure ---
    try:
        ca_coords = residue_data.get('ca_coords', None)
        cb_coords = residue_data.get('cb_coords', None)
        query_idx = residue_data.get('residue_idx', None)
        if ca_coords is not None and cb_coords is not None and query_idx is not None:
            hse = compute_hse(ca_coords, cb_coords, query_idx)
        else:
            hse = {'hse_up': 0, 'hse_down': 0, 'hse_ratio': np.nan}
        features.update(hse)
    except Exception:
        features['hse_up'] = 0
        features['hse_down'] = 0
        features['hse_ratio'] = np.nan

    # --- BLOSUM62 ---
    try:
        resname = residue_data.get('resname', '')
        blosum = compute_blosum62_features(resname)
        features.update(blosum)
    except Exception:
        for i in range(20):
            features[f'blosum62_{i}'] = np.nan

    # --- Hydrogen bond geometry ---
    try:
        if dssp_data is not None:
            all_coords = residue_data.get('all_coords', None)
            if all_coords is not None:
                hbond = compute_hbond_geometry(dssp_data, all_coords)
            else:
                hbond = {
                    'hbond_dist_1': np.nan, 'hbond_energy_1': np.nan,
                    'hbond_dist_2': np.nan, 'hbond_energy_2': np.nan,
                }
        else:
            hbond = {
                'hbond_dist_1': np.nan, 'hbond_energy_1': np.nan,
                'hbond_dist_2': np.nan, 'hbond_energy_2': np.nan,
            }
        features.update(hbond)
    except Exception:
        features['hbond_dist_1'] = np.nan
        features['hbond_energy_1'] = np.nan
        features['hbond_dist_2'] = np.nan
        features['hbond_energy_2'] = np.nan

    # --- Order parameter ---
    try:
        order = compute_order_parameter(bfactor)
        features.update(order)
    except Exception:
        features['order_parameter'] = np.nan

    return features


# ============================================================================
# 7. Feature Name Registry
# ============================================================================

def get_physics_feature_names():
    """Return the complete list of physics feature column names.

    These names correspond to the keys returned by
    compute_all_physics_features() and can be used to construct DataFrame
    columns or verify feature completeness.

    Returns
    -------
    list of str
        All physics feature names in a stable, deterministic order.
    """
    names = []

    # Ring current
    names.extend(['ring_current_h', 'ring_current_ha'])

    # HSE
    names.extend(['hse_up', 'hse_down', 'hse_ratio'])

    # BLOSUM62
    names.extend([f'blosum62_{i}' for i in range(20)])

    # H-bond geometry
    names.extend(['hbond_dist_1', 'hbond_energy_1', 'hbond_dist_2', 'hbond_energy_2'])

    # Order parameter
    names.append('order_parameter')

    return names
