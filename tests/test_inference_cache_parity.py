"""Parity test for cross-residue distance features.

This is the hard gate before any full cache rebuild. It verifies that the
cross-pair arrays computed by:

  1. distance_features.build_cross_arrays_for_residue   (will be called by
     05_build_training_cache.py during cache build)
  2. inference.extract_features_from_pdb                (lives in the
     inference path, calls build_cross_arrays_for_residue internally)

produce IDENTICAL bytes for every residue of a small test PDB.

If this test fails, training data and inference data diverge → predictions
during deployment use a different feature distribution than training, which
silently degrades model quality. Run before any cache rebuild and after any
change to:
  - distance_features.calc_cross_residue_distances
  - distance_features.build_cross_arrays_for_residue
  - inference.extract_features_from_pdb (cross-pair section)
  - 05_build_training_cache.py (cache builder cross-pair section, when added)

Run:
    python -m pytest tests/test_inference_cache_parity.py -v
"""
from __future__ import annotations
import os
import sys
import numpy as np
try:
    import pytest
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False
    # Minimal pytest shim so the @pytest.fixture decorators don't crash
    class _PytestShim:
        @staticmethod
        def fixture(*a, **k):
            def deco(fn):
                return fn
            return deco if not a else deco(a[0])
        @staticmethod
        def skip(msg):
            print(f'SKIP: {msg}')
            sys.exit(0)
    pytest = _PytestShim()
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from distance_features import build_cross_arrays_for_residue
from inference import extract_features_from_pdb
from pdb_utils import parse_pdb
from spatial_neighbors import find_neighbors
from config import (
    ATOM_TO_IDX, K_SPATIAL_NEIGHBORS, CONTEXT_WINDOW,
    MAX_CROSS_DISTANCES, N_CROSS_OFFSET_TYPES,
    CROSS_DIST_CUTOFF, CROSS_H_CUTOFF, AA_3_TO_1, RESIDUE_TO_IDX,
)


# A small PDB known to be valid in the data dir. Pick something with at least
# a few residues and some sidechains to exercise the cross-pair logic.
TEST_PDB = os.path.join(ROOT, 'data', 'pdbs', '1A2I.pdb')


@pytest.fixture(scope='module')
def parsed_pdb():
    """Parse the test PDB once and reuse across tests."""
    if not os.path.isfile(TEST_PDB):
        pytest.skip(f'Test PDB not found: {TEST_PDB}')
    pdb_data = parse_pdb(TEST_PDB, chain_id=None)
    aa_data = {}
    for (chain, res_id), res_info in pdb_data.items():
        if res_info['residue_name'] in AA_3_TO_1 or res_info['residue_name'] in RESIDUE_TO_IDX:
            aa_data[res_id] = res_info
    res_ids = sorted(aa_data.keys())
    spatial = {rid: aa_data[rid] for rid in res_ids}
    neighbors = find_neighbors(spatial, k=K_SPATIAL_NEIGHBORS)
    return aa_data, res_ids, neighbors


def test_cross_arrays_via_helper_directly(parsed_pdb):
    """Sanity check: helper produces the expected shapes and dtypes."""
    aa_data, res_ids, neighbors = parsed_pdb
    rid = res_ids[len(res_ids) // 2]
    nb_ids = neighbors.get(rid, {'ids': [-1] * K_SPATIAL_NEIGHBORS})['ids']

    a1, a2, off, vals, n = build_cross_arrays_for_residue(
        center_rid=rid,
        aa_data=aa_data,
        res_ids_in_order=res_ids,
        spatial_neighbor_ids=nb_ids,
        atom_to_idx=ATOM_TO_IDX,
        context_window=CONTEXT_WINDOW,
        max_cross_distances=MAX_CROSS_DISTANCES,
        n_cross_offset_types=N_CROSS_OFFSET_TYPES,
        heavy_cutoff=CROSS_DIST_CUTOFF,
        h_cutoff=CROSS_H_CUTOFF,
    )
    assert a1.dtype == np.int16 and a1.shape == (MAX_CROSS_DISTANCES,)
    assert a2.dtype == np.int16 and a2.shape == (MAX_CROSS_DISTANCES,)
    assert off.dtype == np.int8 and off.shape == (MAX_CROSS_DISTANCES,)
    assert vals.dtype == np.float16 and vals.shape == (MAX_CROSS_DISTANCES,)
    assert 0 <= n <= MAX_CROSS_DISTANCES

    # Padding region beyond n is the padding sentinel
    assert (a1[n:] == len(ATOM_TO_IDX)).all()
    assert (a2[n:] == len(ATOM_TO_IDX)).all()
    assert (off[n:] == N_CROSS_OFFSET_TYPES).all()


def test_cross_arrays_sorted_ascending_by_distance(parsed_pdb):
    """Pairs in the active region must be sorted ascending by distance."""
    aa_data, res_ids, neighbors = parsed_pdb
    failed = 0
    for rid in res_ids[:30]:
        nb_ids = neighbors.get(rid, {'ids': [-1] * K_SPATIAL_NEIGHBORS})['ids']
        _, _, _, vals, n = build_cross_arrays_for_residue(
            center_rid=rid, aa_data=aa_data,
            res_ids_in_order=res_ids,
            spatial_neighbor_ids=nb_ids,
            atom_to_idx=ATOM_TO_IDX,
            context_window=CONTEXT_WINDOW,
            max_cross_distances=MAX_CROSS_DISTANCES,
            n_cross_offset_types=N_CROSS_OFFSET_TYPES,
            heavy_cutoff=CROSS_DIST_CUTOFF,
            h_cutoff=CROSS_H_CUTOFF,
        )
        if n > 1:
            active = vals[:n].astype(np.float32)
            if not np.all(np.diff(active) >= 0):
                failed += 1
    assert failed == 0, f'{failed}/30 residues had unsorted distances'


def test_inference_matches_cache_helper(parsed_pdb):
    """Hard parity gate: extract_features_from_pdb's cross arrays must match
    build_cross_arrays_for_residue exactly, residue-by-residue.

    This pins the contract: the inference pipeline cannot drift from the
    cache builder pipeline. If we ever change the sort order, cutoff,
    offset encoding, or atom-vocab interaction in one path without the
    other, this test fails immediately.
    """
    aa_data, res_ids, neighbors = parsed_pdb

    # Run the inference path
    inf_residues, inf_res_ids = extract_features_from_pdb(
        TEST_PDB, chain_id=None, atom_to_idx=ATOM_TO_IDX, dssp_stats=None
    )
    assert inf_res_ids == res_ids, 'Residue ordering must match'

    pad_atom = len(ATOM_TO_IDX)
    pad_off = N_CROSS_OFFSET_TYPES

    mismatches = 0
    n_with_pairs = 0
    for inf_res, rid in zip(inf_residues, inf_res_ids):
        nb_ids = neighbors.get(rid, {'ids': [-1] * K_SPATIAL_NEIGHBORS})['ids']
        ref_a1, ref_a2, ref_off, ref_v, ref_n = build_cross_arrays_for_residue(
            center_rid=rid, aa_data=aa_data,
            res_ids_in_order=res_ids,
            spatial_neighbor_ids=nb_ids,
            atom_to_idx=ATOM_TO_IDX,
            context_window=CONTEXT_WINDOW,
            max_cross_distances=MAX_CROSS_DISTANCES,
            n_cross_offset_types=N_CROSS_OFFSET_TYPES,
            heavy_cutoff=CROSS_DIST_CUTOFF,
            h_cutoff=CROSS_H_CUTOFF,
        )

        inf_a1 = inf_res['cross_atom1_idx'].numpy()
        inf_a2 = inf_res['cross_atom2_idx'].numpy()
        inf_off = inf_res['cross_offset_idx'].numpy()
        inf_v = inf_res['cross_distances'].numpy()
        inf_mask = inf_res['cross_dist_mask'].numpy()
        inf_n = int(inf_mask.sum())

        # n must match
        if inf_n != ref_n:
            mismatches += 1
            continue
        if ref_n == 0:
            continue
        n_with_pairs += 1

        # Active region must match exactly
        if not np.array_equal(inf_a1[:ref_n], ref_a1[:ref_n].astype(np.int64)):
            mismatches += 1; continue
        if not np.array_equal(inf_a2[:ref_n], ref_a2[:ref_n].astype(np.int64)):
            mismatches += 1; continue
        if not np.array_equal(inf_off[:ref_n], ref_off[:ref_n].astype(np.int64)):
            mismatches += 1; continue
        # float16 round-trip via float32 — exact match expected because
        # both paths apply the same /10 + clip + cast
        if not np.array_equal(inf_v[:ref_n], ref_v[:ref_n].astype(np.float32)):
            mismatches += 1; continue

    assert n_with_pairs > 0, 'Expected at least some residues to have cross pairs'
    assert mismatches == 0, (
        f'{mismatches} of {len(inf_residues)} residues differ between '
        f'inference and cache helper'
    )


def test_offset_codes_in_valid_range(parsed_pdb):
    """Every active offset code must be in [0, N_CROSS_OFFSET_TYPES) and never
    equal to the center-self code (CONTEXT_WINDOW+1)."""
    aa_data, res_ids, neighbors = parsed_pdb
    self_code = CONTEXT_WINDOW + 1
    saw_self = 0
    out_of_range = 0
    for rid in res_ids[:50]:
        nb_ids = neighbors.get(rid, {'ids': [-1] * K_SPATIAL_NEIGHBORS})['ids']
        _, _, off, _, n = build_cross_arrays_for_residue(
            center_rid=rid, aa_data=aa_data,
            res_ids_in_order=res_ids,
            spatial_neighbor_ids=nb_ids,
            atom_to_idx=ATOM_TO_IDX,
            context_window=CONTEXT_WINDOW,
            max_cross_distances=MAX_CROSS_DISTANCES,
            n_cross_offset_types=N_CROSS_OFFSET_TYPES,
            heavy_cutoff=CROSS_DIST_CUTOFF,
            h_cutoff=CROSS_H_CUTOFF,
        )
        if n == 0:
            continue
        active_off = off[:n]
        if (active_off == self_code).any():
            saw_self += 1
        if ((active_off < 0) | (active_off >= N_CROSS_OFFSET_TYPES)).any():
            out_of_range += 1
    assert saw_self == 0, 'Self-offset code (CW+1) should never appear in cross arrays'
    assert out_of_range == 0, 'All offset codes must be in [0, N_CROSS_OFFSET_TYPES)'


if __name__ == '__main__':
    # Allow running without pytest installed.
    # Build the fixture once and call each test fn directly.
    if not os.path.isfile(TEST_PDB):
        print(f'SKIP: test PDB not found: {TEST_PDB}')
        sys.exit(0)
    pdb_data = parse_pdb(TEST_PDB, chain_id=None)
    aa_data = {}
    for (chain, res_id), res_info in pdb_data.items():
        if res_info['residue_name'] in AA_3_TO_1 or res_info['residue_name'] in RESIDUE_TO_IDX:
            aa_data[res_id] = res_info
    res_ids = sorted(aa_data.keys())
    spatial = {rid: aa_data[rid] for rid in res_ids}
    neighbors = find_neighbors(spatial, k=K_SPATIAL_NEIGHBORS)
    fixture = (aa_data, res_ids, neighbors)

    tests = [
        ('cross arrays via helper', test_cross_arrays_via_helper_directly),
        ('sorted ascending',         test_cross_arrays_sorted_ascending_by_distance),
        ('inference matches cache',  test_inference_matches_cache_helper),
        ('offset codes in range',    test_offset_codes_in_valid_range),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn(fixture)
            print(f'  PASS  {name}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {name}: {e}')
        except Exception as e:
            failed += 1
            print(f'  ERROR {name}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} tests passed.')
    sys.exit(0 if failed == 0 else 1)
