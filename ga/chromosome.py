"""
ga/chromosome.py
================
Defines the search space, chromosome encoding/decoding, and random initialisation.

Each architecture is encoded as a list of 13 integers, where each integer is an
index into the corresponding list of values in SEARCH_SPACE.
"""

import random
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
SEARCH_SPACE: Dict[str, List[Any]] = {
    "num_blocks":   [2, 3, 4, 5],           # Gene 0 — number of conv blocks
    "filters_1":    [32, 64, 128],          # Gene 1 — output channels block 1
    "filters_2":    [64, 128, 256],         # Gene 2 — output channels block 2
    "filters_3":    [128, 256, 512],        # Gene 3 — output channels block 3
    "filters_4":    [256, 512, 1024],       # Gene 4 — output channels block 4
    "filters_5":    [512, 1024, 2048],      # Gene 5 — output channels block 5
    "kernel_size":  [3, 5],                 # Gene 6 — conv kernel size
    "activation":   ["relu", "leaky_relu", "elu"],  # Gene 7
    "dropout":      [0.0, 0.2, 0.4, 0.5],   # Gene 8 — dropout before FC
    "batch_norm":   [True, False],          # Gene 9 — use batch normalisation
    "fc_hidden":    [128, 256, 512],        # Gene 10 — FC hidden units
    "pooling":      ["max", "avg", "mixed"],# Gene 11 — pooling type
    "use_residual": [True, False],          # Gene 12 — whether architecture uses skip connections
}

GENE_NAMES: List[str] = list(SEARCH_SPACE.keys())
GENE_VALUES: List[List[Any]] = list(SEARCH_SPACE.values())
NUM_GENES: int = len(GENE_NAMES)
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Chromosome operations
# ---------------------------------------------------------------------------

def random_chromosome(rng=None) -> List[int]:
    """Return a random chromosome: a list of integer indices into SEARCH_SPACE."""
    rng = rng or random
    return [rng.randrange(len(values)) for values in GENE_VALUES]


def decode(chromosome: List[int]) -> Dict[str, Any]:
    """Convert integer-index chromosome to a human-readable architecture dict."""
    if not isinstance(chromosome, (list, tuple)) or len(chromosome) != NUM_GENES:
        raise ValueError(f"Expected {NUM_GENES} genes (schema {SCHEMA_VERSION}); migrate legacy records by architecture name")
    for i, value in enumerate(chromosome):
        if type(value) is not int or not 0 <= value < len(GENE_VALUES[i]):
            raise ValueError(f"Invalid index {value!r} for {GENE_NAMES[i]}")
    return {GENE_NAMES[i]: GENE_VALUES[i][chromosome[i]] for i in range(NUM_GENES)}


def chromosome_to_str(chromosome: List[int]) -> str:
    """Compact string representation of a chromosome for logging."""
    return "[" + ",".join(str(g) for g in chromosome) + "]"


def describe(chromosome: List[int]) -> str:
    """Pretty-print decoded architecture."""
    arch = decode(chromosome)
    lines = ["Architecture:"]
    for k, v in arch.items():
        lines.append(f"  {k:12s}: {v}")
    return "\n".join(lines)
