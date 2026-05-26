#!/usr/bin/env python3
"""Re-assign the 5-fold split with two stricter dedup policies than the
existing UniProt-only deduper:

  1. Use 90% sequence-identity CLUSTERS (data/identity_clusters_90.json)
     as the primary grouping key, falling back to UniProt, then to
     BMRB-hash. This catches twins that share a sequence but lack a
     mapped UniProt.
  2. Hard-exclude every BMRB whose paired PDB is in the UCBShift-200
     test set. Those go to fold 0 = "always test, never train", so
     the UCBShift-200 benchmark is a legitimately unseen test set.

Assignment policy (applied in order, first match wins):
  - If BMRB's paired PDB ∈ UCBShift-200 test set: fold = 0
  - Elif BMRB has a 90%-identity cluster component (size ≥ 2):
      fold = MD5(min(component)) % 5 + 1
  - Elif BMRB has a UniProt: fold = MD5(uniprot) % 5 + 1
  - Else: fold = MD5(bmrb_id) % 5 + 1   (singleton, no twins to leak)

Outputs (all under data/):
  fold_assignments_identity90.json   {"<bmrb_id>": fold_int 0..5}
  excluded_ucbshift200.json          {"bmrbs": [...], "pdbs": [...]}
  identity90_components.json         {"<canonical_bmrb>": [members]}

Outputs (under data_alphafold/):
  structure_data_hybrid_fold_{0..5}_id90.csv  (only the BMRBs in that fold)

Usage:
  python experiments/dedupe_folds_by_identity90.py            # build
  python experiments/dedupe_folds_by_identity90.py --verify   # only check existing assignment
  python experiments/dedupe_folds_by_identity90.py --no_csv   # skip writing per-fold CSVs
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

N_FOLDS = 5
EXCLUDED_FOLD = 0   # fold-0 = held out from training entirely

DATA_DIR    = os.path.join(_ROOT, 'data')
AF_DIR      = os.path.join(_ROOT, 'data_alphafold')
CLUSTERS    = os.path.join(DATA_DIR, 'identity_clusters_90.json')
UP_MAP_PATH = os.path.join(DATA_DIR, 'alphafold/bmrb_uniprot_mapping.json')
PAIRS_CSV   = os.path.join(DATA_DIR, 'pairs.csv')
BENCHMARK_DIR = os.path.join(_ROOT, 'results/benchmark')
UCB_TEST_DIR = '/home/brooks/Work/Wynn/UCBShift_testing/CSpred/train_model/pdbs/test'

OUT_FOLD_MAP = os.path.join(DATA_DIR, 'fold_assignments_identity90.json')
OUT_EXCLUDED = os.path.join(DATA_DIR, 'excluded_ucbshift200.json')
OUT_COMPS    = os.path.join(DATA_DIR, 'identity90_components.json')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fold_by_hash(key: str, n_folds: int = N_FOLDS) -> int:
    """Stable 5-fold assignment by MD5 of the key. Returns 1..5."""
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % n_folds + 1


class UnionFind:
    """Iterative union-find with path compression + rank."""

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_combined_components(
    clusters: Dict[str, List[str]],
    bmrb_to_uniprot: Dict[str, str],
    universe: Set[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Build connected components from BOTH 90%-identity edges AND UniProt
    co-membership, over a universe of BMRBs. Two BMRBs are in the same
    component if there is any path through:
      - identity_clusters_90 adjacency, OR
      - shared UniProt accession.

    Returns:
        bmrb_to_canonical: bmrb_id -> canonical (== min member of its component)
        canonical_to_members: canonical -> sorted list of member bmrb_ids
    """
    # Pool = universe ∪ all clusters' BMRBs ∪ all uniprot-mapping BMRBs
    pool: Set[str] = set(universe)
    pool.update(map(str, clusters.keys()))
    for v in clusters.values():
        pool.update(map(str, v))
    pool.update(map(str, bmrb_to_uniprot.keys()))

    uf = UnionFind(pool)

    # Identity-90 edges
    for k, twins in clusters.items():
        for t in twins:
            uf.union(str(k), str(t))

    # UniProt edges: group all BMRBs sharing a UniProt
    up_groups: Dict[str, List[str]] = defaultdict(list)
    for b, up in bmrb_to_uniprot.items():
        if up:
            up_groups[up].append(str(b))
    for up, bs in up_groups.items():
        if len(bs) >= 2:
            anchor = bs[0]
            for b in bs[1:]:
                uf.union(anchor, b)

    # Materialize components and pick canonical = min(member)
    comp: Dict[str, List[str]] = defaultdict(list)
    for b in pool:
        comp[uf.find(b)].append(b)
    bmrb_to_canonical: Dict[str, str] = {}
    canonical_to_members: Dict[str, List[str]] = {}
    for members in comp.values():
        members_sorted = sorted(members)
        canonical = min(members)
        canonical_to_members[canonical] = members_sorted
        for m in members_sorted:
            bmrb_to_canonical[m] = canonical
    return bmrb_to_canonical, canonical_to_members


def load_bmrb_to_pdbs(pairs_csv: str) -> Dict[str, Set[str]]:
    """data/pairs.csv: Entry_ID, pdb_ids (comma-separated, 4-char each).
    Returns: bmrb_id -> set of UPPERCASED 4-char PDB IDs.
    """
    df = pd.read_csv(pairs_csv, dtype={'Entry_ID': str})
    out: Dict[str, Set[str]] = {}
    for _, r in df.iterrows():
        bmrb = str(r['Entry_ID'])
        raw = str(r['pdb_ids']) if not pd.isna(r['pdb_ids']) else ''
        pdbs = {p.strip().upper() for p in raw.split(',') if p.strip()}
        if pdbs:
            out[bmrb] = pdbs
    return out


def load_ucb_test_pdb4(test_dir: str) -> Set[str]:
    """List 4-char PDB IDs from the UCBShift test directory.
    Filenames are <PDB4><CHAIN>.pdb where CHAIN is 1 char or '_'.
    Returns the unique 4-char PDB IDs (uppercased).
    """
    pdbs: Set[str] = set()
    if not os.path.isdir(test_dir):
        print(f'WARN: UCB test dir not found: {test_dir}')
        return pdbs
    for fn in sorted(os.listdir(test_dir)):
        if not fn.endswith('.pdb'):
            continue
        stem = fn[:-4]                  # e.g. "1AM7A" or "109M_"
        if len(stem) >= 4:
            pdbs.add(stem[:4].upper())
    return pdbs


def load_bmrb_to_pdbs_from_benchmark(benchmark_dir: str) -> Dict[str, Set[str]]:
    """Scan results/benchmark/ filenames matching <PDB><CHAIN>_<BMRB>_*.json
    to recover the BMRB → 4-char-PDB pairing used for benchmarking. Covers
    proteins not present in pairs.csv (e.g. UCBShift test PDBs).
    """
    out: Dict[str, Set[str]] = {}
    if not os.path.isdir(benchmark_dir):
        return out
    for fn in os.listdir(benchmark_dir):
        if not fn.endswith('.json'):
            continue
        stem = (fn.replace('_excluded', '')
                  .replace('_result', '')
                  .replace('.json', ''))
        parts = stem.split('_')
        if len(parts) < 2:
            continue
        pdb_str = parts[0]
        bmrb_str = parts[1]
        if not pdb_str or not bmrb_str.isdigit():
            continue
        out.setdefault(bmrb_str, set()).add(pdb_str[:4].upper())
    return out


def merge_bmrb_pdb_dicts(*dicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Merge several BMRB→{PDB} dicts taking the union per BMRB."""
    out: Dict[str, Set[str]] = {}
    for d in dicts:
        for b, pdbs in d.items():
            out.setdefault(b, set()).update(pdbs)
    return out


# ---------------------------------------------------------------------------
# Main fold assignment
# ---------------------------------------------------------------------------

def assign_folds(
    bmrbs: Iterable[str],
    bmrb_to_canonical: Dict[str, str],
    multi_member_canonicals: Set[str],
    excluded_bmrbs: Set[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Returns:
        bmrb_to_fold:  bmrb -> fold_int (0..5)
        policy_count:  dict counting which policy chose the fold

    Policy (first match wins):
      1. excluded (paired to UCBShift-200 test PDB) → fold 0
      2. canonical ∈ multi_member_canonicals → MD5("comp_"+canon) % 5 + 1
         (combined identity-90 + UniProt component; applies to ALL members)
      3. fallback → MD5("bmrb_"+bmrb) % 5 + 1   (true singletons)
    """
    out: Dict[str, int] = {}
    policy: Dict[str, int] = {'excluded': 0, 'component': 0, 'bmrb_hash': 0}
    for b in bmrbs:
        b = str(b)
        if b in excluded_bmrbs:
            out[b] = EXCLUDED_FOLD
            policy['excluded'] += 1
            continue
        canon = bmrb_to_canonical.get(b)
        if canon is not None and canon in multi_member_canonicals:
            out[b] = fold_by_hash(f'comp_{canon}')
            policy['component'] += 1
            continue
        out[b] = fold_by_hash(f'bmrb_{b}')
        policy['bmrb_hash'] += 1
    return out, policy


def verify_no_leakage(
    bmrb_to_fold: Dict[str, int],
    bmrb_to_canonical: Dict[str, str],
    bmrb_to_uniprot: Dict[str, str],
    excluded_bmrbs: Set[str],
) -> Dict[str, int]:
    """Three checks:
    1. component_leaks: combined identity-90/UniProt component spans 2+ training folds
    2. uniprot_leaks: any UniProt group spans 2+ training folds (subset of #1; sanity)
    3. excluded_in_train_folds: any UCB-excluded BMRB ended up in fold 1..5
    """
    # Component check
    comp_to_folds: Dict[str, Set[int]] = defaultdict(set)
    for b, f in bmrb_to_fold.items():
        if f == EXCLUDED_FOLD:
            continue
        canon = bmrb_to_canonical.get(b)
        if canon is None:
            continue
        comp_to_folds[canon].add(f)
    comp_leaks = sum(1 for s in comp_to_folds.values() if len(s) > 1)

    # UniProt check (should be 0 if comp_leaks is 0, since UniProt edges
    # are folded into the component graph).
    up_to_folds: Dict[str, Set[int]] = defaultdict(set)
    for b, f in bmrb_to_fold.items():
        if f == EXCLUDED_FOLD:
            continue
        up = bmrb_to_uniprot.get(b)
        if up is None:
            continue
        up_to_folds[up].add(f)
    up_leaks = sum(1 for s in up_to_folds.values() if len(s) > 1)

    excluded_in_train = sum(
        1 for b in excluded_bmrbs if bmrb_to_fold.get(b, EXCLUDED_FOLD) != EXCLUDED_FOLD
    )

    return {
        'component_leaks': comp_leaks,
        'uniprot_leaks': up_leaks,
        'excluded_in_train_folds': excluded_in_train,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true',
                    help='Only verify the existing assignment; do not write')
    ap.add_argument('--no_csv', action='store_true',
                    help='Skip writing per-fold CSVs (faster smoke test)')
    args = ap.parse_args()

    print('Loading inputs...')
    clusters = json.load(open(CLUSTERS))
    print(f'  identity_clusters_90: {len(clusters):,} BMRBs covered')

    up_map = json.load(open(UP_MAP_PATH))
    up_map = {str(k): v for k, v in up_map.items() if v}
    print(f'  bmrb_uniprot_mapping: {len(up_map):,} entries')

    bmrb_to_pdbs_pairs = load_bmrb_to_pdbs(PAIRS_CSV)
    bmrb_to_pdbs_bench = load_bmrb_to_pdbs_from_benchmark(BENCHMARK_DIR)
    bmrb_to_pdbs = merge_bmrb_pdb_dicts(bmrb_to_pdbs_pairs, bmrb_to_pdbs_bench)
    print(f'  pairs.csv:           {len(bmrb_to_pdbs_pairs):,} BMRBs with paired PDBs')
    print(f'  results/benchmark/:  {len(bmrb_to_pdbs_bench):,} BMRBs with paired PDBs')
    print(f'  union:               {len(bmrb_to_pdbs):,} BMRBs')

    ucb_test_pdb4 = load_ucb_test_pdb4(UCB_TEST_DIR)
    print(f'  UCBShift-200 test PDBs: {len(ucb_test_pdb4)} unique 4-char IDs')

    # --- Build BMRB pool from existing per-fold CSVs (deferred to after
    # PDB/UCB exclusion; we need the universe to feed combined-components) ---
    print('\nLoading BMRB pool from existing per-fold CSVs...')
    fold_csvs = [os.path.join(AF_DIR, f'structure_data_hybrid_fold_{f}.csv')
                 for f in range(1, N_FOLDS + 1)]
    bmrb_pool: Set[str] = set()
    for path in fold_csvs:
        if not os.path.isfile(path):
            print(f'  MISSING: {path}'); continue
        df = pd.read_csv(path, usecols=['bmrb_id'], dtype={'bmrb_id': str})
        bmrb_pool.update(df['bmrb_id'].astype(str).unique())
    print(f'  total BMRBs in pool: {len(bmrb_pool):,}')

    # --- Build COMBINED (identity-90 + UniProt) components ---
    print('\nBuilding combined components (identity-90 ∪ UniProt) via union-find...')
    bmrb_to_canonical, canonical_to_members = build_combined_components(
        clusters, up_map, bmrb_pool)
    n_components = len(canonical_to_members)
    multi_member_canonicals: Set[str] = {
        c for c, m in canonical_to_members.items() if len(m) > 1
    }
    print(f'  components: {n_components:,}  '
          f'(of which multi-member: {len(multi_member_canonicals):,})')

    # --- Identify BMRBs to exclude ---
    print('\nIdentifying UCBShift-200 BMRBs for exclusion...')
    excluded_bmrbs: Set[str] = set()
    excluded_pdbs_observed: Set[str] = set()
    for b, pdbs in bmrb_to_pdbs.items():
        hits = pdbs & ucb_test_pdb4
        if hits:
            excluded_bmrbs.add(b)
            excluded_pdbs_observed.update(hits)
    in_pool_excluded = excluded_bmrbs & bmrb_pool
    print(f'  BMRBs directly paired to a test PDB: {len(excluded_bmrbs):,}')
    print(f'    of which in the training pool: {len(in_pool_excluded):,}')
    print(f'  test PDBs matched: {len(excluded_pdbs_observed)} / {len(ucb_test_pdb4)}')
    unmatched = ucb_test_pdb4 - excluded_pdbs_observed
    if unmatched:
        print(f'  test PDBs with NO BMRB pairing (orphaned): {len(unmatched)}')

    # Propagate exclusion through the combined component: any BMRB in the
    # same identity-90 / UniProt component as a directly-excluded BMRB is
    # also excluded. Otherwise a UniProt-twin of a UCB-200 protein would
    # remain in training and leak through retrieval.
    excluded_canonicals = {bmrb_to_canonical.get(b) for b in excluded_bmrbs
                           if bmrb_to_canonical.get(b) is not None}
    propagated = set()
    for canon in excluded_canonicals:
        for m in canonical_to_members.get(canon, []):
            if m not in excluded_bmrbs:
                propagated.add(m)
    excluded_bmrbs |= propagated
    print(f'  +{len(propagated)} BMRBs excluded by component propagation '
          f'(identity-90 / UniProt twins of test-paired BMRBs)')
    in_pool_excluded = excluded_bmrbs & bmrb_pool
    print(f'  Total excluded: {len(excluded_bmrbs):,} '
          f'(in training pool: {len(in_pool_excluded):,})')

    # --- Assign folds ---
    print('\nAssigning folds...')
    bmrb_to_fold, policy = assign_folds(
        bmrb_pool,
        bmrb_to_canonical=bmrb_to_canonical,
        multi_member_canonicals=multi_member_canonicals,
        excluded_bmrbs=excluded_bmrbs,
    )
    print('  policy counts:')
    for k, v in policy.items():
        print(f'    {k:<12s} {v:>7,}')

    # --- Distribution ---
    from collections import Counter
    dist = Counter(bmrb_to_fold.values())
    print('\nFold distribution:')
    for f in sorted(dist):
        label = '(EXCLUDED)' if f == EXCLUDED_FOLD else ''
        print(f'  fold {f}: {dist[f]:>5,} BMRBs {label}')

    # --- Verify ---
    print('\nLeakage verification...')
    leaks = verify_no_leakage(
        bmrb_to_fold,
        bmrb_to_canonical=bmrb_to_canonical,
        bmrb_to_uniprot=up_map,
        excluded_bmrbs=excluded_bmrbs,
    )
    for k, v in leaks.items():
        ok = v == 0
        print(f'  {k:<28s} {v:>5}   {"✓" if ok else "✗ FAIL"}')
    if any(v != 0 for v in leaks.values()):
        print('\nLEAKAGE DETECTED — refusing to write outputs.')
        return 1

    # --- Compare to existing splits (informational) ---
    try:
        old_v2_split: Dict[str, int] = {}
        for f in range(1, N_FOLDS + 1):
            sp = os.path.join(_ROOT, 'data/struct_retrieval_v2/cache',
                              f'fold_{f}/structural/bmrb_mapping.json')
            if os.path.isfile(sp):
                bmap = json.load(open(sp))
                for b in set(bmap.values()):
                    old_v2_split[str(b)] = f
        if old_v2_split:
            common = set(old_v2_split) & set(bmrb_to_fold)
            moved = sum(1 for b in common if old_v2_split[b] != bmrb_to_fold[b])
            print(f'\nVs v2 hybrid split: {moved:,} of {len(common):,} BMRBs '
                  f'changed fold ({100*moved/len(common):.1f}%)')
    except Exception as e:
        print(f'  (skipping v2-comparison: {e})')

    if args.verify:
        print('\n--verify only; not writing outputs.')
        return 0

    # --- Save canonical assignment + excluded list + components ---
    with open(OUT_FOLD_MAP, 'w') as fh:
        json.dump(bmrb_to_fold, fh)
    print(f'\nWrote: {OUT_FOLD_MAP}')

    with open(OUT_EXCLUDED, 'w') as fh:
        json.dump({
            'bmrbs': sorted(excluded_bmrbs),
            'pdbs': sorted(excluded_pdbs_observed),
            'all_test_pdbs': sorted(ucb_test_pdb4),
            'unmatched_test_pdbs': sorted(unmatched),
        }, fh, indent=2)
    print(f'Wrote: {OUT_EXCLUDED}')

    with open(OUT_COMPS, 'w') as fh:
        # Only write multi-member components to keep this readable
        json.dump({c: m for c, m in canonical_to_members.items() if len(m) > 1},
                  fh, indent=2)
    print(f'Wrote: {OUT_COMPS}  (multi-member components only)')

    if args.no_csv:
        print('\n--no_csv set; skipping per-fold CSV regrouping.')
        return 0

    # --- Save per-fold CSVs (re-grouped by new fold) ---
    print('\nRe-grouping per-fold CSVs...')
    dfs = []
    for f, path in zip(range(1, N_FOLDS + 1), fold_csvs):
        if not os.path.isfile(path):
            continue
        d = pd.read_csv(path, dtype={'bmrb_id': str})
        d['_orig_fold'] = f
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df['split'] = df['bmrb_id'].astype(str).map(bmrb_to_fold).fillna(-1).astype(int)
    df = df.drop(columns=['_orig_fold'])
    out_dir = AF_DIR
    for f in [EXCLUDED_FOLD] + list(range(1, N_FOLDS + 1)):
        sub = df[df['split'] == f]
        if len(sub) == 0:
            continue
        out_path = os.path.join(out_dir, f'structure_data_hybrid_fold_{f}_id90.csv')
        sub.to_csv(out_path, index=False)
        print(f'  fold {f}: {len(sub):>9,} rows, '
              f'{sub.bmrb_id.nunique():>5} unique BMRBs → '
              f'{os.path.basename(out_path)}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
