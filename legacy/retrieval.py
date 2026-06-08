#!/usr/bin/env python3
"""
Retrieval Module for Chemical Shift Prediction (Better Data Pipeline)

Adapted from homologies/retrieval_module.py with the following modifications:

1. Imports FAISS_NPROBE and K_RETRIEVED from config instead of hardcoding

2. Keeps Retriever and EmbeddingLookup classes identical to original

Usage:
    retriever = Retriever(index_dir='/path/to/indices', exclude_fold=1)
    results = retriever.retrieve(query_embeddings, query_bmrb_ids)
"""

import json
import os
import pickle
import sys
from typing import Optional

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss
import h5py
import numpy as np

from config import FAISS_NPROBE, K_RETRIEVED, STANDARD_RESIDUES, AA_3_TO_1, RESIDUE_TO_IDX


# ============================================================================
# Retriever (identical to original)
# ============================================================================

class Retriever:
    """
    Handles retrieval of similar residues using FAISS indices.

    Two supported modes (set via `mode` kwarg):
      - 'only' (new, default): retrieve from a single fold's residues. The
        index file naming is index_only_fold_{k}.faiss. Used by the current
        cache design: each fold_k/ cache retrieves from fold k's residues.
        When train.py loads the OTHER folds' caches, it automatically gets
        retrievals from different folds than the test fold — no leakage.
      - 'exclude' (legacy): retrieve from all folds except fold k. File is
        index_exclude_fold_{k}.faiss. Kept for back-compat with older caches.

    Always excludes same-protein matches (by bmrb_id).
    """

    def __init__(
        self,
        index_dir: str,
        exclude_fold: int = None,
        only_fold: int = None,
        k: int = K_RETRIEVED,
        nprobe: int = FAISS_NPROBE,
        device: str = 'cpu',
        mode: str = None,
    ):
        """
        Initialize retriever. Exactly one of (exclude_fold, only_fold) must be set,
        or pass `mode` along with the fold number via exclude_fold.

        Args:
            index_dir: Directory containing FAISS indices and metadata
            exclude_fold: (legacy) fold to exclude from retrieval pool
            only_fold: (new) fold to retrieve from exclusively
            k, nprobe, device: as before
            mode: 'only' or 'exclude' (auto-detected from which kwarg is set)
        """
        self.index_dir = index_dir
        self.k = k
        self.nprobe = nprobe
        self.device = device

        # Resolve mode
        if only_fold is not None:
            self.mode = 'only'
            self.fold = only_fold
        elif exclude_fold is not None:
            self.mode = mode or 'exclude'
            self.fold = exclude_fold
        else:
            raise ValueError("Must pass either only_fold or exclude_fold")
        self.exclude_fold = exclude_fold  # kept for back-compat with callers
        self.only_fold = only_fold

        # Resolve index + metadata paths
        prefix = 'only' if self.mode == 'only' else 'exclude'
        index_path = os.path.join(index_dir, f'index_{prefix}_fold_{self.fold}.faiss')
        metadata_path = os.path.join(index_dir, f'metadata_{prefix}_fold_{self.fold}.pkl')

        print(f"Loading FAISS index from {index_path}... (mode: {self.mode})")
        self.index = faiss.read_index(index_path)
        self.index.nprobe = nprobe

        # Optionally move to GPU (uses cuVS backend if available — required
        # for Blackwell / sm_120 since FAISS' native CUDA kernels don't yet
        # target RTX 50-series).
        if device == 'cuda' and faiss.get_num_gpus() > 0:
            print("  Moving index to GPU...")
            self._gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            if hasattr(co, 'use_cuvs'):
                co.use_cuvs = True
            self.index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self.index, co)

        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load shift columns
        with open(os.path.join(index_dir, 'shift_cols.json'), 'r') as f:
            self.shift_cols = json.load(f)

        # Only exclude exact same-protein matches (no identity clustering)
        print("  Exclusion mode: same-protein only (exact match)")

        # Build protein ID lookup for fast exclusion
        self._build_protein_lookup()

        print(f"  Index contains {self.index.ntotal:,} residues")
        print(f"  Shift columns: {len(self.shift_cols)}")

    def _build_protein_lookup(self):
        """Flatten metadata into dense numpy arrays for vectorized retrieve()."""
        n = len(self.metadata)
        n_shifts = len(self.shift_cols)

        self.meta_bmrb_ids = np.array([m['bmrb_id'] for m in self.metadata], dtype=object)
        self.meta_residue_ids = np.array([m['residue_id'] for m in self.metadata], dtype=np.int32)
        self.meta_residue_codes = np.array([m['residue_code'] for m in self.metadata], dtype=np.int32)

        self.meta_shifts = np.empty((n, n_shifts), dtype=np.float32)
        self.meta_shift_masks = np.empty((n, n_shifts), dtype=bool)
        for i, m in enumerate(self.metadata):
            self.meta_shifts[i] = m['shifts']
            self.meta_shift_masks[i] = m['shift_mask']

        self.idx_to_bmrb = self.meta_bmrb_ids  # kept for any legacy callers

    def retrieve(
        self,
        query_embeddings: np.ndarray,
        query_bmrb_ids: list[str],
        k: Optional[int] = None,
    ) -> dict:
        """
        Retrieve similar residues for query embeddings.

        Args:
            query_embeddings: (n_queries, embed_dim) normalized embeddings
            query_bmrb_ids: List of BMRB IDs for each query (for same-protein exclusion)
            k: Number of neighbors (overrides init if provided)

        Returns:
            Dictionary with:
                'indices': (n_queries, k) FAISS indices
                'distances': (n_queries, k) cosine similarities
                'bmrb_ids': (n_queries, k) BMRB IDs of retrieved residues
                'residue_ids': (n_queries, k) residue IDs
                'residue_codes': (n_queries, k) residue type indices
                'shifts': (n_queries, k, n_shifts) chemical shifts
                'shift_masks': (n_queries, k, n_shifts) validity masks
        """
        k = k or self.k
        n_queries = len(query_embeddings)
        n_shifts = len(self.shift_cols)

        # Normalize queries (FAISS inner product = cosine similarity for normalized vectors)
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(query_embeddings)

        # Retrieve more than k to allow for same-protein filtering. Same-protein
        # hits are rare in a large pool (a ~200-residue protein in a ~2M-residue
        # index yields 0–5 same-protein hits in the top ~40 typically), so k+10
        # is plenty.
        k_extra = k + 10

        distances, indices = self.index.search(query_embeddings, k_extra)

        # Vectorized filter: mark invalid / same-protein positions.
        query_bmrb_arr = np.asarray([str(b) for b in query_bmrb_ids], dtype=object)
        safe_idx = np.where(indices >= 0, indices, 0)  # clip -1 for safe gather
        retrieved_bmrb_raw = self.meta_bmrb_ids[safe_idx]           # (nq, ke)
        invalid = indices < 0
        same_protein = retrieved_bmrb_raw == query_bmrb_arr[:, None]
        keep = ~(invalid | same_protein)

        # FAISS returns candidates in distance-sorted order, so "top-k valid" is
        # just "the first k True entries in each row of `keep`".
        rank = np.cumsum(keep, axis=1)  # (nq, ke), 1-indexed rank among valids
        select = keep & (rank <= k)     # (nq, ke)
        dest_col = rank - 1             # 0-indexed destination column

        # Scatter selected (row, col) -> out[row, dest_col[row, col]]
        r, c = np.nonzero(select)
        dc = dest_col[r, c]
        sel_idx = indices[r, c]  # source FAISS index

        out_indices = np.full((n_queries, k), -1, dtype=np.int64)
        out_distances = np.zeros((n_queries, k), dtype=np.float32)
        out_indices[r, dc] = sel_idx
        out_distances[r, dc] = distances[r, c]

        # Gather metadata via fancy indexing (use safe_out for -1 positions)
        safe_out = np.where(out_indices >= 0, out_indices, 0)
        out_residue_ids = self.meta_residue_ids[safe_out].astype(np.int32)
        out_residue_codes = self.meta_residue_codes[safe_out].astype(np.int32)
        out_shifts = self.meta_shifts[safe_out]
        out_shift_masks = self.meta_shift_masks[safe_out]
        out_bmrb_ids = self.meta_bmrb_ids[safe_out]

        # Zero out positions where the slot is invalid (fewer than k valid hits).
        invalid_out = out_indices < 0
        if invalid_out.any():
            out_residue_ids[invalid_out] = 0
            out_residue_codes[invalid_out] = 0
            out_shifts[invalid_out] = 0
            out_shift_masks[invalid_out] = False
            out_bmrb_ids[invalid_out] = ''

        return {
            'indices': out_indices,
            'distances': out_distances,
            'bmrb_ids': out_bmrb_ids,
            'residue_ids': out_residue_ids,
            'residue_codes': out_residue_codes,
            'shifts': out_shifts,
            'shift_masks': out_shift_masks,
        }

    def retrieve_batch(
        self,
        query_embeddings: np.ndarray,
        query_bmrb_ids: list[str],
        batch_size: int = 10000,
        k: Optional[int] = None,
    ) -> dict:
        """
        Retrieve in batches for very large query sets.
        """
        k = k or self.k
        n_queries = len(query_embeddings)
        n_shifts = len(self.shift_cols)

        # Initialize output arrays
        out_indices = np.full((n_queries, k), -1, dtype=np.int64)
        out_distances = np.zeros((n_queries, k), dtype=np.float32)
        out_bmrb_ids = np.empty((n_queries, k), dtype=object)
        out_residue_ids = np.zeros((n_queries, k), dtype=np.int32)
        out_residue_codes = np.zeros((n_queries, k), dtype=np.int32)
        out_shifts = np.zeros((n_queries, k, n_shifts), dtype=np.float32)
        out_shift_masks = np.zeros((n_queries, k, n_shifts), dtype=bool)

        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            batch_result = self.retrieve(
                query_embeddings[start:end],
                query_bmrb_ids[start:end],
                k=k,
            )

            out_indices[start:end] = batch_result['indices']
            out_distances[start:end] = batch_result['distances']
            out_bmrb_ids[start:end] = batch_result['bmrb_ids']
            out_residue_ids[start:end] = batch_result['residue_ids']
            out_residue_codes[start:end] = batch_result['residue_codes']
            out_shifts[start:end] = batch_result['shifts']
            out_shift_masks[start:end] = batch_result['shift_masks']

        return {
            'indices': out_indices,
            'distances': out_distances,
            'bmrb_ids': out_bmrb_ids,
            'residue_ids': out_residue_ids,
            'residue_codes': out_residue_codes,
            'shifts': out_shifts,
            'shift_masks': out_shift_masks,
        }


# ============================================================================
# EmbeddingLookup (identical to original)
# ============================================================================

class EmbeddingLookup:
    """
    Memory-efficient lookup of ESM embeddings by (bmrb_id, residue_id).
    Uses HDF5 file access instead of loading everything into RAM.
    """

    def __init__(self, h5_path: str, cache_size: int = 500):
        """
        Initialize embedding lookup with lazy loading.

        Args:
            h5_path: Path to ESM embeddings HDF5 file
            cache_size: Number of proteins to cache in memory (LRU)
        """
        self.h5_path = h5_path
        self.cache_size = cache_size

        # Keep file handle open for fast access
        self.h5f = h5py.File(h5_path, 'r')

        # Build index: bmrb_id -> list of residue_ids (lightweight)
        print(f"Indexing ESM embeddings from {h5_path}...")
        self.protein_ids = list(self.h5f['embeddings'].keys())
        self.residue_index = {}  # bmrb_id -> {residue_id -> local_idx}

        for bmrb_id in self.protein_ids:
            residue_ids = self.h5f['embeddings'][bmrb_id]['residue_ids'][:]
            self.residue_index[bmrb_id] = {int(rid): i for i, rid in enumerate(residue_ids)}

        print(f"  Indexed {len(self.protein_ids)} proteins")

        # Get embedding dimension from first protein
        sample_emb = self.h5f['embeddings'][self.protein_ids[0]]['embeddings'][0]
        self.embed_dim = sample_emb.shape[0]
        print(f"  Embedding dimension: {self.embed_dim}")

        # Simple LRU cache for recently accessed proteins
        self._cache = {}
        self._cache_order = []

    def _load_protein(self, bmrb_id: str) -> np.ndarray:
        """Load embeddings for a protein, using cache."""
        if bmrb_id in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(bmrb_id)
            self._cache_order.append(bmrb_id)
            return self._cache[bmrb_id]

        # Load from file
        embs = self.h5f['embeddings'][bmrb_id]['embeddings'][:].astype(np.float32)

        # Add to cache
        self._cache[bmrb_id] = embs
        self._cache_order.append(bmrb_id)

        # Evict oldest if cache full
        while len(self._cache_order) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return embs

    def get(self, bmrb_id: str, residue_id: int) -> Optional[np.ndarray]:
        """Get embedding for a single residue."""
        bmrb_str = str(bmrb_id)
        if bmrb_str not in self.residue_index:
            return None
        if residue_id not in self.residue_index[bmrb_str]:
            return None

        local_idx = self.residue_index[bmrb_str][residue_id]
        embs = self._load_protein(bmrb_str)
        return embs[local_idx]

    def get_protein(self, bmrb_id: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Get all embeddings for a protein.

        Returns:
            (residue_ids, embeddings) or None if protein not found
        """
        bmrb_str = str(bmrb_id)
        if bmrb_str not in self.residue_index:
            return None

        residue_ids = np.array(list(self.residue_index[bmrb_str].keys()), dtype=np.int32)
        embs = self._load_protein(bmrb_str)
        return residue_ids, embs

    def get_batch(
        self,
        bmrb_ids: list[str],
        residue_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for a batch of residues.

        Optimized to group by protein and use vectorized numpy operations.

        Returns:
            embeddings: (n, embed_dim) array
            valid: (n,) boolean mask indicating which queries had valid embeddings
        """
        from collections import defaultdict

        n = len(bmrb_ids)
        embeddings = np.zeros((n, self.embed_dim), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        # Group requests by protein to minimize HDF5 reads
        by_protein = defaultdict(list)
        for i, (bmrb_id, rid) in enumerate(zip(bmrb_ids, residue_ids)):
            by_protein[str(bmrb_id)].append((i, rid))

        # Load each protein once, extract all needed residues with vectorized ops
        for bmrb_str, requests in by_protein.items():
            if bmrb_str not in self.residue_index:
                continue

            # Load all embeddings for this protein (single HDF5 read)
            embs = self._load_protein(bmrb_str)
            residue_idx_map = self.residue_index[bmrb_str]

            # Vectorized extraction: build index arrays
            orig_indices = []
            local_indices = []
            for orig_idx, rid in requests:
                if rid in residue_idx_map:
                    orig_indices.append(orig_idx)
                    local_indices.append(residue_idx_map[rid])

            if orig_indices:
                orig_indices = np.array(orig_indices, dtype=np.int64)
                local_indices = np.array(local_indices, dtype=np.int64)
                # Single vectorized copy operation
                embeddings[orig_indices] = embs[local_indices]
                valid[orig_indices] = True

        return embeddings, valid

    def close(self):
        """Close the HDF5 file handle."""
        if hasattr(self, 'h5f') and self.h5f is not None:
            self.h5f.close()
            self.h5f = None

    def __del__(self):
        self.close()


def test_retrieval():
    """Test retrieval with synthetic data."""
    print("Testing retrieval module (better data pipeline)...")

    # This would need actual data to test properly
    print("  (Requires pre-built indices - skipping live test)")
    print("  Module loaded successfully!")


if __name__ == "__main__":
    test_retrieval()
