# Manuscript Draft vs. Code: Discrepancies and Architecture Description

## Part 1: Differences Between Manuscript Draft and Code

### Things the Manuscript Describes That the Code Does NOT Do

1. **BLAST search for missing PDB entries**
   The manuscript says: "For BMRB entries with no corresponding PDB entry listed in the BMRB's bulk data, we use BLAST on the PDB database for appropriate entries."
   The code does NOT do this. Instead, the code (`01_build_datasets.py`) uses the BMRB's own BMRB-to-PDB pairing data, and for proteins without a PDB match, it falls back to AlphaFold structures fetched via UniProt ID mapping. There is a separate utility script (`find_missing_uniprots.py`) that uses EBI BLAST to find UniProt IDs for BMRB entries, but this is for AlphaFold lookup, not for finding PDB entries directly.

2. **30% sequence identity threshold for BLAST hits**
   The manuscript draft mentions "30% sequence identity" for BLAST results, then immediately notes "(30% is not enough, do not do this)." This is not implemented in code. The code uses an 80% minimum sequence identity threshold (`config.py:159`, `MIN_SEQUENCE_IDENTITY = 0.80`) for BMRB-to-PDB alignment acceptance.

3. **"Distance matrix" description is slightly misleading**
   The manuscript says: "We next compute a distance matrix for each residue, consisting of the distances between each heavy atom and each other heavy atom and the distances between each hydrogen atom and the heavy atom it is directly bonded to."
   The code computes pairwise distances between atoms within a residue, but stores them as a flat list of (atom1, atom2, distance) triples, not as a matrix. The distinction matters because the model processes these as a list of pairs with learned atom embeddings, not as a 2D matrix. Also, the H-to-heavy restriction (hydrogens only paired with their bonded heavy atom) is implemented in `distance_features.py`.

4. **Three datasets described incompletely**
   The manuscript says: "The first dataset consists of all proteins from the BMRB for which there are no nonstandard residues or nucleotides present, and their corresponding AlphaFold structures. The second dataset consists only of data for which there is an experimental counterpart. The third dataset consists exclusively of AlphaFold structures."
   In the code, the three datasets are:
   - **hybrid** (`data/`): experimental PDB where available, AlphaFold where not (this is the primary training set)
   - **experimental** (`data_experimental/`): experimental PDB structures only
   - **alphafold** (`data_alphafold/`): AlphaFold structures only
   The manuscript's first dataset description ("all proteins... and their corresponding AlphaFold structures") sounds like it describes the alphafold-only dataset, but the primary training dataset is actually the hybrid one.

### Things the Code Does That the Manuscript Does NOT Mention

1. **Per-amino-acid normalization**
   The code (`05_build_training_cache.py`, `dataset.py`) implements per-(amino_acid, shift_type) z-score normalization. Targets are re-normalized on-the-fly from global z-scores to per-AA z-scores. This is critical for sidechain shifts whose distributions vary dramatically across amino acid types (e.g., CD1 in leucine ~24 ppm vs. CD1 in tyrosine ~132 ppm). The manuscript does not mention this.

2. **ESM-2 embeddings and FAISS retrieval**
   The manuscript mentions "vector search in embedding space" and "transfer from retrieved chemical shifts" but does not describe the specific tools: ESM-2 3B (`esm2_t36_3B_UR50D`, 2560-dim embeddings, `03_extract_esm_embeddings.py`) for embeddings, and FAISS IVF indices (`04_build_retrieval_index.py`) with K=32 neighbors and nprobe=64 for retrieval. The manuscript should describe this.

3. **Fold-aware retrieval exclusion**
   The code builds separate FAISS indices for each of the 5 folds, excluding the test fold from the retrieval pool to prevent data leakage. Additionally, proteins with >90% sequence identity (computed via MMseqs2 clustering, `cluster_sequences.py`) are excluded from each other's retrieval results. The manuscript mentions the 90% threshold but not the fold-aware indexing.

4. **Alignment mismatch encoding**
   The code uses 6 mismatch types: `match`, `gap_in_cs`, `gap_in_structure`, `mismatch`, `protein_edge`, `UNK` (`config.py:42`). The manuscript only describes 4 of these, omitting `protein_edge` (positions at the N/C terminus) and `UNK`.

5. **DSSP features**
   The code extracts 9 DSSP features per residue: relative accessibility, 4 hydrogen bond relative indices and 4 hydrogen bond energies. These are used both in the CNN input and in the 49-dim structural feature vector. The manuscript mentions DSSP but does not enumerate which features are extracted.

6. **Sidechain geometry features**
   The code computes 5 sidechain geometry descriptors per residue: number of resolved atoms, mean distance from CA, compactness (std of distances), max extent, and centroid distance. These are part of the 49-dim structural feature vector. The manuscript does not mention these.

7. **NMR model selection**
   The code selects the median-representative NMR model: superimposes all chains onto Model 1 via Kabsch, computes median CA coordinates across all models, and selects the model with lowest RMSD from the median (`01_build_datasets.py:584-668`). The manuscript describes this correctly.

8. **Joint distance attention for spatial neighbors**
   The code concatenates the query residue's intra-residue distance pairs + a CA-CA inter-residue distance pair + the neighbor's intra-residue distance pairs into a single set of 2M+1 pairs, and processes them jointly through the same DistanceAttention module. The manuscript does not describe this.

9. **Retrieval dropout**
   During training, all retrieval data for a sample is dropped with probability 0.3 (`RETRIEVAL_DROPOUT`), forcing the model to learn structure-only predictions as well. The manuscript does not mention this.

10. **Huber loss**
    The code uses Huber loss with delta=0.5 (`HUBER_DELTA`), which is robust to outliers. Backbone shifts are weighted 2x (`BACKBONE_LOSS_WEIGHT`). The manuscript does not specify the loss function.

11. **CosineAnnealingWarmRestarts scheduler**
    The code uses AdamW with cosine annealing warm restarts (T_0=50, T_mult=2). The manuscript does not mention the learning rate schedule.

12. **Gradient clipping**
    The code clips gradient norms to 1.0. Not mentioned in the manuscript.

13. **Canonical atom vocabulary**
    The code uses a fixed 88-atom vocabulary (`config.py:60-84`) shared across all datasets, ensuring model/cache interchangeability. The manuscript does not describe this.

14. **49-dim structural feature vector**
    The code constructs a 49-dim feature vector for each residue: 21 backbone pairwise distances + 5 sidechain geometry + 4 backbone angles (sin/cos phi/psi) + 9 DSSP features + 10 secondary structure one-hot. This vector is used in both the retrieval pathway (for query and neighbor description) and structure extraction. The manuscript does not describe this.

15. **Inference without retrieval**
    At inference time (`inference.py`), there is no FAISS index available. The model falls back to pure structure-only prediction, with the retrieval gate learning to go to zero when no valid neighbors are provided. The manuscript does not discuss inference.

### Things Where the Manuscript and Code Agree

- Downloading all BMRB entries (16,246)
- PairwiseAligner for best PDB chain selection
- Discarding proteins with nonstandard residues
- Masking shifts >4 standard deviations from the mean
- Kabsch superimposition of NMR chains
- Median model selection for NMR ensembles
- 80% minimum sequence identity for BMRB-PDB pairs
- Gap/mismatch encoding (partially — code has 6 types, manuscript 4)
- K=5 spatial neighbors by CA distance
- K=32 retrieval neighbors
- Context window of 11 residues (5+1+5)
- CNN-based architecture
- Pairwise distance attention with learned atom embeddings
- Retrieval-augmented prediction with shift transfer

### Manuscript Issues to Fix

- Remove the BLAST/30% identity paragraph entirely (not implemented, noted as bad idea)
- Clarify the three dataset descriptions (hybrid is primary, not alphafold-only)
- Add per-AA normalization to Methods
- Describe ESM-2 and FAISS explicitly
- Describe the full 49-dim structural feature vector
- Add mismatch types `protein_edge` and `UNK`
- Describe DSSP features extracted (9 features, not just "backbone angles, RSA, H-bond energies")
- Describe sidechain geometry features
- Add loss function (Huber, delta=0.5, backbone 2x weight)
- Add training details (AdamW, cosine warm restarts, gradient clipping)
- Add retrieval dropout
- Describe inference mode (structure-only fallback)
- Complete the architecture description (cuts off mid-sentence)
- The model predicts 49 shift types (6 backbone + 43 sidechain), not just 6

---

## Part 2: Model Architecture (Plain Language)

### How the Model Predicts Chemical Shifts: From Structure to Prediction

Our model takes a protein's three-dimensional structure as input and predicts the NMR chemical shift of every atom in every residue. The basic idea is simple: the chemical shift of an atom depends on its local three-dimensional environment. An atom buried deep inside a helix will resonate at a different frequency than the same type of atom sitting on the surface of a loop. Our model learns to read a protein's structure and infer what each atom's chemical shift should be.

The model makes its predictions using two complementary strategies, then blends them together:

1. **Structure-based prediction**: Looking directly at the local 3D geometry around each residue.
2. **Retrieval-based prediction**: Finding similar residues elsewhere in the training data and borrowing their known chemical shifts.

We describe each strategy below, then explain how the model combines them.

### Input: What the Model Sees

For each residue whose shifts we want to predict, the model receives:

- **A window of 11 residues** centered on the target (5 before, the target, 5 after in sequence). For each residue in this window, the model gets every pairwise distance between heavy atoms within that residue, plus the distance from each hydrogen to the heavy atom it is bonded to. These distances are stored as a list of (atom 1, atom 2, distance in Angstroms) triples. There are up to 400 such pairs per residue.

- **The amino acid type** of each residue in the window (one of 20 standard types, or "unknown").

- **Secondary structure** (helix, sheet, coil, turn, etc.) as determined by the DSSP algorithm.

- **An alignment flag** indicating whether the residue was a perfect match between the BMRB sequence and the PDB structure, a mutation, a gap (insertion/deletion in either dataset), or a terminal position.

- **DSSP-derived features**: relative solvent accessibility (how exposed the residue is to solvent), and hydrogen bond energies and partner positions for up to two backbone hydrogen bonds.

- **Five spatial neighbors**: the five closest residues in 3D space (by CA-CA distance, excluding residues within 4 positions in sequence). For each neighbor, the model receives all of the same structural information listed above — distances, amino acid type, secondary structure, backbone angles, and DSSP features.

- **32 retrieval neighbors**: the 32 residues from the rest of the training data that are most similar to the target residue in a learned embedding space (described below). For each retrieved neighbor, the model receives the neighbor's known chemical shifts, amino acid type, a structural feature summary, and a similarity score.

### Strategy 1: Structure-Based Prediction

The first strategy processes the local 3D geometry of the target residue and its sequential and spatial neighbors to predict chemical shifts directly from structure.

**Step 1: Learning what distances mean.** Raw atom-to-atom distances are just numbers — 2.4 Angstroms between a CA and a CB doesn't mean much to a neural network on its own. So the model first learns an "embedding" (a short numerical description) for each atom type. When the model sees a distance pair like (CA, CB, 2.4 A), it looks up its learned descriptions of "CA" and "CB", concatenates them with the distance value, and passes this through a small neural network. It does this for every distance pair in the residue, then uses an attention mechanism to combine all the pairs into a single 256-dimensional summary of that residue's geometry. This is called the Distance Attention module, and it runs independently for each of the 11 residues in the window.

**Step 2: Building a sequence-context encoding.** The distance attention summaries (256 dimensions each) are combined with embeddings for the amino acid type (64-dim), secondary structure (32-dim), alignment status (16-dim), a validity flag (16-dim), and a projection of the DSSP features (32-dim). This gives a 416-dimensional vector for each of the 11 positions in the window.

These 11 vectors are then passed through a 5-layer convolutional neural network (CNN). Each layer uses 1D convolutions with a kernel width of 3, meaning each position can see its immediate neighbors. With 5 layers stacked, the center position can "see" information from all 11 positions in the window. The CNN progressively increases the representation size from 416 to 1280 dimensions, with residual connections (shortcuts that help training), group normalization, and dropout at each layer. After the CNN, we extract only the center position's representation (1280 dimensions), which now encodes information about the target residue in the context of its sequential neighbors.

**Step 3: Incorporating spatial neighbors.** Proteins fold in three dimensions, so residues that are far apart in sequence can end up right next to each other in space. To capture this, the model processes the 5 nearest spatial neighbors using a separate attention mechanism.

For each spatial neighbor, the model does something clever: it concatenates the target residue's own distance pairs, a single CA-CA distance between the target and the neighbor (capturing how far apart they are in space), and the neighbor's own distance pairs, into one combined set of distance pairs. This entire set is processed through the same Distance Attention module used in Step 1, allowing the model to jointly learn how the geometry of two nearby residues relates to each other.

Each spatial neighbor is also described by its amino acid type, secondary structure, backbone angles, and the log of the sequence separation (how far apart the residues are in the sequence, which helps distinguish "close because of a hairpin" from "close because of a domain interface"). These features are combined with the joint distance attention output and fed through a multi-head attention mechanism that weights the contribution of each spatial neighbor by its relevance.

The spatial attention output (192 dimensions) is concatenated with the CNN output (1280 dimensions) to form the **base encoding** (1472 dimensions). This base encoding is the model's full understanding of the target residue's local structural environment.

**Step 4: Predicting shifts from structure.** The base encoding is passed through a 4-layer feed-forward network (1472 to 1024 to 512 to 256 to 49 shift predictions), with GELU activations and dropout. This produces the **structure-only prediction**: the model's best guess at all 49 chemical shifts given only the structural data.

### Strategy 2: Retrieval-Based Prediction

The second strategy leverages the observation that similar residues in similar structural environments tend to have similar chemical shifts. Instead of predicting shifts from scratch, the model finds similar residues in the training data and transfers their known shifts.

**Step 1: Finding similar residues.** Before training, we run each protein's sequence through ESM-2, a large pretrained protein language model (3 billion parameters). ESM-2 produces a 2560-dimensional embedding for each residue that captures rich information about that residue's sequence context, evolutionary relationships, and likely structural role. We use FAISS (a fast vector search library) to build an index of all residue embeddings in the training set. For each target residue, we retrieve the 32 most similar residues by cosine similarity. To prevent data leakage, we exclude any residue from a protein with greater than 90% sequence identity to the target protein, and we use separate indices for each cross-validation fold so that test proteins are never in the retrieval pool.

**Step 2: Encoding the retrieved neighbors.** Each of the 32 retrieved neighbors is described by a rich set of features: its amino acid type, its rank in the retrieval results, the cosine similarity score, whether it is the same amino acid type as the target, its known chemical shifts, how those shifts deviate from the consensus of all retrieved neighbors, the fraction of its shifts that are measured, and a 49-dimensional structural feature summary for both the query and the neighbor (plus the L2 distance between these summaries). All of these features are concatenated and passed through a 3-layer neural network to produce a 320-dimensional encoding of each neighbor.

**Step 3: Letting neighbors talk to each other.** The 32 neighbor encodings are passed through 2 self-attention layers — standard transformer-style attention where each neighbor can attend to every other neighbor. This allows the model to learn contextual relationships between neighbors: for example, if 30 of 32 neighbors agree on a shift value but 2 are outliers, the self-attention layers can learn to downweight the outliers.

**Step 4: Per-shift querying.** Different chemical shifts care about different things. The CA shift is sensitive to backbone geometry, while a sidechain shift like CD1 depends on sidechain packing. To handle this, the model maintains a separate learned "query" embedding for each of the 49 shift types. These queries attend to the neighbor encodings through 3 layers of cross-attention: each shift type independently decides which neighbors are most informative for that particular shift. The result is a 320-dimensional "shift context" vector for each of the 49 shifts.

**Step 5: Weighted shift transfer.** A scoring network examines each neighbor's encoding and produces a per-shift importance weight. These weights are normalized via softmax (so they sum to 1) and used to compute a weighted average of the retrieved neighbors' known chemical shifts. Neighbors with missing shifts are masked out. This produces the **retrieval prediction**: a weighted average of the most relevant known shifts from the training set.

### Blending the Two Strategies

The model learns a **retrieval gate** for each shift type: a value between 0 and 1 that controls how much to trust the retrieval prediction versus the structure-only prediction. The gate takes as input the shift-specific context from the retrieval pathway, the structural base encoding, and a count of how many valid retrieved neighbors exist. When retrieval data is abundant and reliable, the gate tends toward 1 (trust the retrieved shifts). When few neighbors are available or they disagree, the gate tends toward 0 (fall back to structure).

The final prediction is simply:

    prediction = gate * retrieval_prediction + (1 - gate) * structure_prediction

During training, the retrieval pathway is randomly disabled for 30% of samples (all retrieved data is zeroed out), which forces the model to maintain a strong structure-only predictor as a fallback. This is important because at inference time on a novel protein, retrieval data may not be available.

### Training

The model is trained on 491,345 residues (for the experimental dataset) using 5-fold cross-validation. We minimize Huber loss (a variant of mean squared error that is less sensitive to outlier predictions) with delta=0.5. Backbone shifts (C, CA, CB, N, H, HA) are weighted 2x in the loss to emphasize the most commonly measured and biologically important shifts. We use the AdamW optimizer with learning rate 2e-4, weight decay 0.05, cosine annealing warm restarts (T_0=50 epochs, T_mult=2), gradient clipping at norm 1.0, and mixed-precision (FP16) training for efficiency.

Chemical shift targets are normalized per amino acid type and per shift type: for each (amino acid, atom) combination, shifts are z-scored using the training set's mean and standard deviation for that combination. This is important because the same atom type can have very different chemical shift ranges depending on the amino acid (for example, CD1 in leucine is around 24 ppm while CD1 in tyrosine is around 132 ppm).

### Output

The model outputs 49 predicted chemical shifts per residue (6 backbone: C, CA, CB, N, H, HA; and 43 sidechain shifts), each accompanied by a validity mask indicating whether that shift type is expected to exist for the residue's amino acid type. Predictions are denormalized back to ppm using the per-amino-acid statistics before reporting.
