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

    The retriever automatically excludes:
    1. Residues from the held-out test fold
    2. Residues from the same protein as the query
    3. Residues from proteins with >90% sequence identity to the query protein
    """

    def __init__(
        self,
        index_dir: str,
        exclude_fold: int,
        k: int = K_RETRIEVED,
        nprobe: int = FAISS_NPROBE,
        device: str = 'cpu',
    ):
        """
        Initialize retriever for a specific fold configuration.

        Args:
            index_dir: Directory containing FAISS indices and metadata
            exclude_fold: Which fold to exclude (the test fold)
            k: Number of neighbors to retrieve
            nprobe: Number of clusters to search (higher = more accurate but slower)
            device: 'cpu' or 'cuda'
        """
        self.index_dir = index_dir
        self.exclude_fold = exclude_fold
        self.k = k
        self.nprobe = nprobe
        self.device = device

        # Load index
        index_path = os.path.join(index_dir, f'index_exclude_fold_{exclude_fold}.faiss')
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        self.index.nprobe = nprobe

        # Optionally move to GPU
        if device == 'cuda' and faiss.get_num_gpus() > 0:
            print("  Moving index to GPU...")
            gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)

        # Load metadata
        metadata_path = os.path.join(index_dir, f'metadata_exclude_fold_{exclude_fold}.pkl')
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load shift columns
        with open(os.path.join(index_dir, 'shift_cols.json'), 'r') as f:
            self.shift_cols = json.load(f)

        # Load identity exclusion map (>90% sequence identity clusters)
        self.identity_exclusion = {}
        exclusion_path = os.path.join(index_dir, 'identity_clusters_90.json')
        if os.path.exists(exclusion_path):
            with open(exclusion_path, 'r') as f:
                self.identity_exclusion = json.load(f)
            n_with = sum(1 for v in self.identity_exclusion.values() if v)
            print(f"  Loaded identity exclusion map: {n_with} proteins with >90% identity neighbors")
        else:
            print("  WARNING: No identity exclusion map found. Only same-protein exclusion active.")

        # Build protein ID lookup for fast exclusion
        self._build_protein_lookup()

        print(f"  Index contains {self.index.ntotal:,} residues")
        print(f"  Shift columns: {len(self.shift_cols)}")

    def _build_protein_lookup(self):
        """Build lookup table mapping FAISS index -> protein ID for fast exclusion."""
        self.idx_to_bmrb = np.array([
            m['bmrb_id'] for m in self.metadata
        ])

        # Create mapping from bmrb_id to set of indices (for fast exclusion)
        self.bmrb_to_indices = {}
        for idx, bmrb_id in enumerate(self.idx_to_bmrb):
            if bmrb_id not in self.bmrb_to_indices:
                self.bmrb_to_indices[bmrb_id] = []
            self.bmrb_to_indices[bmrb_id].append(idx)

        for k in self.bmrb_to_indices:
            self.bmrb_to_indices[k] = set(self.bmrb_to_indices[k])

    def _get_exclusion_set(self, query_bmrb):
        """Get the full set of FAISS indices to exclude for a query protein.

        Includes the query protein itself plus all proteins with >90% identity.
        """
        # Start with same-protein exclusion
        exclude_set = set(self.bmrb_to_indices.get(query_bmrb, set()))

        # Add proteins with >90% sequence identity
        similar_proteins = self.identity_exclusion.get(query_bmrb, [])
        for similar_bmrb in similar_proteins:
            exclude_set.update(self.bmrb_to_indices.get(similar_bmrb, set()))

        return exclude_set

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
        query_embeddings = query_embeddings.copy()
        faiss.normalize_L2(query_embeddings)

        # Retrieve more than k to allow for same-protein filtering
        k_extra = k + 50  # Get extras to filter

        distances, indices = self.index.search(query_embeddings, k_extra)

        # Initialize output arrays
        out_indices = np.full((n_queries, k), -1, dtype=np.int64)
        out_distances = np.zeros((n_queries, k), dtype=np.float32)
        out_bmrb_ids = np.empty((n_queries, k), dtype=object)
        out_residue_ids = np.zeros((n_queries, k), dtype=np.int32)
        out_residue_codes = np.zeros((n_queries, k), dtype=np.int32)
        out_shifts = np.zeros((n_queries, k, n_shifts), dtype=np.float32)
        out_shift_masks = np.zeros((n_queries, k, n_shifts), dtype=bool)

        # Filter same-protein and >90% identity matches
        for q in range(n_queries):
            query_bmrb = str(query_bmrb_ids[q])
            exclude_set = self._get_exclusion_set(query_bmrb)

            count = 0
            for i in range(k_extra):
                idx = indices[q, i]
                if idx == -1:  # Invalid
                    continue
                if idx in exclude_set:  # Same protein
                    continue

                # Valid neighbor
                out_indices[q, count] = idx
                out_distances[q, count] = distances[q, i]

                meta = self.metadata[idx]
                out_bmrb_ids[q, count] = meta['bmrb_id']
                out_residue_ids[q, count] = meta['residue_id']
                out_residue_codes[q, count] = meta['residue_code']
                out_shifts[q, count] = meta['shifts']
                out_shift_masks[q, count] = meta['shift_mask']

                count += 1
                if count >= k:
                    break

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
