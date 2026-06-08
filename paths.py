"""Overridable filesystem roots for the chemical-shift pipeline.

Finding #15a: the pipeline hardcoded build-machine paths (the big-storage root
`/home/brooks/1TB/Wynn`, the dataset directories, the main `data/` root). On any
other machine these are wrong and there was no way to override them without
editing source. This module centralizes every such root behind an environment
variable, falling back to the historical default when the variable is unset, so:

  * `import config` on a fresh checkout (no env vars) yields byte-identical
    defaults to before this change (backward compatible), and
  * a different machine can point the pipeline at its own disks purely via
    environment variables, no source edits.

Environment variables (all optional; default = previous hardcoded value):

  SHIFT_DATA_ROOT        main dataset dir              default: 'data'
  SHIFT_PDB_DIR          PDB files dir                 default: '<DATA_ROOT>/pdbs'
  SHIFT_ALPHAFOLD_DIR    AlphaFold mmCIF cache         default: '<DATA_ROOT>/alphafold'
  SHIFT_BIG_STORAGE      big-output root               default: '/home/brooks/1TB/Wynn'
  SHIFT_DATA_ALPHAFOLD   alphafold dataset dir         default: 'data_alphafold'
  SHIFT_DATA_EXPERIMENTAL experimental dataset dir     default: 'data_experimental'
  SHIFT_DATA_REFDB       refdb dataset dir             default: 'data_refdb'
  SHIFT_DATA_REREFERENCED rereferenced dataset dir     default: 'data_rereferenced'

Keep this module dependency-free (stdlib only) so `config` can import it cheaply.
"""

import os


def _env(name, default):
    """Return the env override for `name`, or `default` if unset/blank.

    A set-but-empty variable is treated as unset so an accidental `export
    SHIFT_DATA_ROOT=` does not silently point the pipeline at the empty string.
    """
    val = os.environ.get(name)
    if val is None or val == '':
        return default
    return val


# ----------------------------------------------------------------------------
# Main dataset roots
# ----------------------------------------------------------------------------
DATA_DIR = _env('SHIFT_DATA_ROOT', 'data')
PDB_DIR = _env('SHIFT_PDB_DIR', os.path.join(DATA_DIR, 'pdbs'))
ALPHAFOLD_DIR = _env('SHIFT_ALPHAFOLD_DIR', os.path.join(DATA_DIR, 'alphafold'))

# ----------------------------------------------------------------------------
# Big-output storage root (caches, embeddings, FAISS indices, runs, results)
# ----------------------------------------------------------------------------
BIG_STORAGE = _env('SHIFT_BIG_STORAGE', '/home/brooks/1TB/Wynn')
BIG_RUNS_DIR = os.path.join(BIG_STORAGE, 'runs')
BIG_CACHES_DIR = os.path.join(BIG_STORAGE, 'caches')
BIG_RESULTS_DIR = os.path.join(BIG_STORAGE, 'results')

# ----------------------------------------------------------------------------
# Per-dataset directories (the `--data` flag maps to these).
# `hybrid` always resolves to the main DATA_DIR so an overridden data root keeps
# `--data hybrid` working.
# ----------------------------------------------------------------------------
DATASET_DIRS = {
    'hybrid': DATA_DIR,
    'alphafold': _env('SHIFT_DATA_ALPHAFOLD', 'data_alphafold'),
    'experimental': _env('SHIFT_DATA_EXPERIMENTAL', 'data_experimental'),
    'refdb': _env('SHIFT_DATA_REFDB', 'data_refdb'),
    'rereferenced': _env('SHIFT_DATA_REREFERENCED', 'data_rereferenced'),
}
