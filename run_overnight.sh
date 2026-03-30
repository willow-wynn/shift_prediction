#!/bin/bash
set -e

echo "=========================================="
echo "OVERNIGHT PIPELINE: Structure-Bootstrapped Retrieval"
echo "=========================================="
echo "Started: $(date)"

# Phase 1: Train structure-only model (~3-4 hrs)
echo ""
echo "=== PHASE 1: Train structure-only model ==="
python train_structure_only.py \
    --fold 1 \
    --cache_dir data/cache \
    --output_dir data/struct_only/checkpoints \
    --epochs 200 \
    --batch_size 512

# Phase 2: Extract embeddings (~10 min)
echo ""
echo "=== PHASE 2: Extract structure embeddings ==="
python 03b_extract_struct_embeddings.py \
    --checkpoint data/struct_only/checkpoints/best_struct_fold1.pt \
    --cache_dir data/cache \
    --output data/struct_retrieval_v2/struct_embeddings.h5

# Phase 3: Build FAISS indices (~1 hr)
echo ""
echo "=== PHASE 3: Build FAISS indices ==="
# Create minimal CSV for 04
mkdir -p data/struct_retrieval_v2
if [ ! -f data/struct_retrieval_v2/structure_data_index.csv ]; then
    cp data/struct_retrieval/structure_data_index.csv data/struct_retrieval_v2/
fi
ln -sf "$(pwd)/data/struct_retrieval_v2/structure_data_index.csv" \
    data/struct_retrieval_v2/structure_data_hybrid.csv

python 04_build_retrieval_index.py \
    --data_dir data/struct_retrieval_v2 \
    --embeddings data/struct_retrieval_v2/struct_embeddings.h5 \
    --output_dir data/struct_retrieval_v2/retrieval_indices

# Copy identity clusters
cp data/retrieval_indices/identity_clusters_90.json \
    data/struct_retrieval_v2/retrieval_indices/ 2>/dev/null || true

# Phase 4: Build training caches (~2.5 hrs)
echo ""
echo "=== PHASE 4: Build training caches ==="
# Symlink full CSV for 05 (needs distance columns)
ln -sf "$(pwd)/data/structure_data_hybrid.csv" \
    data/struct_retrieval_v2/structure_data_hybrid.csv

# Create per-fold CSVs if they don't exist (avoid pandas OOM)
for fold in 1 2 3 4 5; do
    src="data/struct_retrieval/structure_data_hybrid_fold_${fold}.csv"
    dst="data/struct_retrieval_v2/structure_data_hybrid_fold_${fold}.csv"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        ln -s "$(readlink -f $src)" "$dst"
    fi
done

python 05_build_training_cache.py \
    --data_dir data/struct_retrieval_v2 \
    --embeddings data/struct_retrieval_v2/struct_embeddings.h5 \
    --index_dir data/struct_retrieval_v2/retrieval_indices \
    --output_dir data/struct_retrieval_v2/cache \
    --device cuda

# Phase 5: Train retrieval with frozen base encoder (~3 hrs)
echo ""
echo "=== PHASE 5: Train retrieval (frozen base encoder) ==="
python train_retrieval_frozen.py \
    --struct_checkpoint data/struct_only/checkpoints/best_struct_fold1.pt \
    --cache_dir data/struct_retrieval_v2/cache \
    --output_dir data/struct_retrieval_v2/checkpoints \
    --fold 1 \
    --epochs 150 \
    --batch_size 512

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE: $(date)"
echo "=========================================="
