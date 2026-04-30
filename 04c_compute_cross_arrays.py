#!/usr/bin/env python3
"""04c — Add cross-residue distance arrays to an EXISTING fold cache.

This is the lightweight Phase-1 augmentation path. Instead of rebuilding
the entire fold cache from scratch (which 05_build_training_cache.py
does), we read the existing cache's structural metadata (bmrb_mapping,
global_to_resid, spatial_ids) and only compute + save the 5 new
cross_*.npy files alongside it.

Useful for:
  - Validating the cross-array computation on existing v2 caches
    without a full rebuild.
  - Smoke-testing Phase-1 inference + training before committing to
    a full rebuild on the new identity-90 fold split.

CAVEAT: The intra-distances in the existing cache came from a specific
PDB chosen at 01_build_datasets time. The cross-distances we compute
here come from the FIRST resolvable PDB for each BMRB. If those differ,
intra and cross become slightly inconsistent. For the v2 hybrid cache
(BMRB-hash split, ~50% UniProt leakage) the intra/cross consistency
matters less because we're not deploying that cache anyway. For the
final production cache, do the full rebuild via 05_build_training_cache.py
(see claude/cross_features_full_rebuild_plan.md).

Usage:
  python 04c_compute_cross_arrays.py \
      --cache_dir data/struct_retrieval_v2/cache/fold_1/structural \
      --pdb_search_dirs data/pdbs /home/brooks/1TB/Wynn/data_archive/alphafold \
      --uniprot_map data/alphafold/bmrb_uniprot_mapping.json \
      [--limit 50]   # process only N proteins (smoke test)

Outputs (under cache_dir/):
  cross_atom1.npy   int16   (N_residues, M_CR=200)
  cross_atom2.npy   int16   (N_residues, M_CR=200)
  cross_offset.npy  int8    (N_residues, M_CR=200)
  cross_values.npy  float16 (N_residues, M_CR=200)
  cross_count.npy   int16   (N_residues,)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    ATOM_TO_IDX, K_SPATIAL_NEIGHBORS, CONTEXT_WINDOW,
    MAX_CROSS_DISTANCES, N_CROSS_OFFSET_TYPES,
    CROSS_DIST_CUTOFF, CROSS_H_CUTOFF, AA_3_TO_1, RESIDUE_TO_IDX,
)
from distance_features import build_cross_arrays_for_residue
from pdb_utils import parse_pdb, resolve_pdb_path


def load_pairs_csv(path: str) -> Dict[str, List[str]]:
    """data/pairs.csv: Entry_ID, pdb_ids  -> bmrb -> [pdb1, pdb2, ...]"""
    if not os.path.isfile(path):
        return {}
    import pandas as pd
    df = pd.read_csv(path, dtype={'Entry_ID': str})
    out: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        b = str(r['Entry_ID'])
        raw = '' if (isinstance(r['pdb_ids'], float) and np.isnan(r['pdb_ids'])) else str(r['pdb_ids'])
        out[b] = [p.strip().upper() for p in raw.split(',') if p.strip()]
    return out


def resolve_bmrb_pdb(bmrb_id: str,
                      pairs: Dict[str, List[str]],
                      uniprot_map: Dict[str, str],
                      pdb_search_dirs: List[str]) -> Optional[str]:
    """Try experimental first, then AlphaFold."""
    # Experimental
    for pdb_id in pairs.get(bmrb_id, []):
        p = resolve_pdb_path(pdb_id, pdb_search_dirs)
        if p:
            return p
    # AlphaFold
    up = uniprot_map.get(bmrb_id)
    if up:
        for d in pdb_search_dirs:
            for pat in [f'AF-{up}-F1-model_v6.pdb',
                         f'AF-{up}-F1-model_v6_fix.pdb',
                         f'AF-{up}-F1-model_v4.pdb']:
                p = os.path.join(d, pat)
                if os.path.isfile(p):
                    return p
    return None


def parse_pdb_aa(path: str) -> Dict[int, dict]:
    """Parse PDB and filter to standard amino acids, indexed by res_id."""
    pdb_data = parse_pdb(path, chain_id=None)
    aa = {}
    for (chain, res_id), info in pdb_data.items():
        rn = info['residue_name']
        if rn in AA_3_TO_1 or rn in RESIDUE_TO_IDX:
            # If the same res_id appears in multiple chains, keep the first.
            aa.setdefault(res_id, info)
    return aa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', required=True,
                    help='Path to <fold_cache>/structural/')
    ap.add_argument('--pdb_search_dirs', nargs='+',
                    default=['data/pdbs', '/home/brooks/1TB/Wynn/data_archive/alphafold'])
    ap.add_argument('--pairs_csv', default='data/pairs.csv')
    ap.add_argument('--uniprot_map', default='data/alphafold/bmrb_uniprot_mapping.json')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only N proteins (smoke test)')
    ap.add_argument('--dry_run', action='store_true',
                    help='Compute but do not save .npy files')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f'Augmenting cache: {cache_dir}')

    # Load existing arrays
    print('Loading existing cache metadata...')
    bmrb_mapping = json.load(open(cache_dir / 'bmrb_mapping.json'))
    global_to_resid = json.load(open(cache_dir / 'global_to_resid.json'))
    spatial_ids = np.load(cache_dir / 'spatial_ids.npy')   # (N, K_SPATIAL)
    residue_idx = np.load(cache_dir / 'residue_idx.npy')   # (N,) -- to get N
    n_residues = residue_idx.shape[0]
    print(f'  n_residues = {n_residues:,}')
    print(f'  spatial_ids shape = {spatial_ids.shape}')

    # Group residues by BMRB so we parse each PDB once
    print('Grouping residues by BMRB...')
    bmrb_to_globals: Dict[str, List[int]] = {}
    for g_str, b in bmrb_mapping.items():
        g = int(g_str)
        bmrb_to_globals.setdefault(str(b), []).append(g)
    bmrbs = sorted(bmrb_to_globals.keys(), key=int)
    print(f'  unique BMRBs: {len(bmrbs):,}')
    if args.limit:
        bmrbs = bmrbs[:args.limit]
        print(f'  --limit set: processing {len(bmrbs)}')

    # PDB resolver state
    pairs = load_pairs_csv(args.pairs_csv)
    uniprot_map = json.load(open(args.uniprot_map)) if os.path.isfile(args.uniprot_map) else {}
    print(f'  pairs.csv entries: {len(pairs):,}')
    print(f'  bmrb→uniprot entries: {len(uniprot_map):,}')

    # Allocate output arrays
    pad_atom = len(ATOM_TO_IDX)
    pad_off = N_CROSS_OFFSET_TYPES
    M_CR = MAX_CROSS_DISTANCES
    out_a1 = np.full((n_residues, M_CR), pad_atom, dtype=np.int16)
    out_a2 = np.full((n_residues, M_CR), pad_atom, dtype=np.int16)
    out_off = np.full((n_residues, M_CR), pad_off, dtype=np.int8)
    out_v = np.zeros((n_residues, M_CR), dtype=np.float16)
    out_n = np.zeros((n_residues,), dtype=np.int16)

    # Walk proteins
    n_pdb_ok = 0; n_pdb_miss = 0; n_residues_filled = 0
    pdb_failures: List[str] = []
    t0 = time.time()
    for bmrb in tqdm(bmrbs, desc='proteins'):
        pdb_path = resolve_bmrb_pdb(bmrb, pairs, uniprot_map, args.pdb_search_dirs)
        if pdb_path is None:
            n_pdb_miss += 1
            pdb_failures.append(bmrb)
            continue
        try:
            aa_data = parse_pdb_aa(pdb_path)
        except Exception as e:
            n_pdb_miss += 1
            pdb_failures.append(f'{bmrb} parse-error: {e}')
            continue
        if not aa_data:
            n_pdb_miss += 1
            pdb_failures.append(f'{bmrb} no-aa-residues')
            continue
        n_pdb_ok += 1
        res_ids_in_order = sorted(aa_data.keys())

        for g in bmrb_to_globals[bmrb]:
            rid = global_to_resid.get(str(g))
            if rid is None or int(rid) not in aa_data:
                continue
            rid = int(rid)
            # Spatial neighbor residue IDs from existing cache
            sp_ids = spatial_ids[g].tolist()
            a1, a2, off, vals, n = build_cross_arrays_for_residue(
                center_rid=rid, aa_data=aa_data,
                res_ids_in_order=res_ids_in_order,
                spatial_neighbor_ids=sp_ids,
                atom_to_idx=ATOM_TO_IDX,
                context_window=CONTEXT_WINDOW,
                max_cross_distances=MAX_CROSS_DISTANCES,
                n_cross_offset_types=N_CROSS_OFFSET_TYPES,
                heavy_cutoff=CROSS_DIST_CUTOFF,
                h_cutoff=CROSS_H_CUTOFF,
            )
            out_a1[g] = a1
            out_a2[g] = a2
            out_off[g] = off
            out_v[g] = vals
            out_n[g] = n
            n_residues_filled += 1

    elapsed = time.time() - t0
    print(f'\n  PDB resolved/parsed: {n_pdb_ok:,} / {len(bmrbs):,}  '
          f'(missing: {n_pdb_miss:,})')
    print(f'  residues with cross arrays filled: {n_residues_filled:,} / {n_residues:,}')
    print(f'  elapsed: {elapsed/60:.1f} min  '
          f'({elapsed/max(n_pdb_ok,1):.2f} s/protein)')

    # Distribution of cross_count
    if n_residues_filled > 0:
        nz = out_n[out_n > 0]
        print(f'  cross_count: mean={nz.mean():.1f}  '
              f'median={np.median(nz):.0f}  p95={np.percentile(nz, 95):.0f}  '
              f'p99={np.percentile(nz, 99):.0f}  max={nz.max()}  '
              f'fraction at cap M_CR={M_CR}: {(nz==M_CR).mean()*100:.2f}%')

    if pdb_failures and len(pdb_failures) <= 20:
        print(f'  failures: {pdb_failures}')
    elif pdb_failures:
        print(f'  failures (first 20 of {len(pdb_failures)}): {pdb_failures[:20]}')

    if args.dry_run:
        print('\n--dry_run set; not writing arrays.')
        return 0

    print('\nSaving cross arrays...')
    np.save(cache_dir / 'cross_atom1.npy', out_a1)
    np.save(cache_dir / 'cross_atom2.npy', out_a2)
    np.save(cache_dir / 'cross_offset.npy', out_off)
    np.save(cache_dir / 'cross_values.npy', out_v)
    np.save(cache_dir / 'cross_count.npy', out_n)
    for f in ['cross_atom1.npy', 'cross_atom2.npy', 'cross_offset.npy',
              'cross_values.npy', 'cross_count.npy']:
        sz = (cache_dir / f).stat().st_size / 1e6
        print(f'  {f}: {sz:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
