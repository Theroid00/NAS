"""Versioned small MLP architecture encoding for tabular classification."""
import random
from typing import List, Dict, Any

SEARCH_SPACE: Dict[str, List[Any]] = {
    "num_layers": [1, 2, 3, 4],
    "width_1": [16, 32, 64, 128],
    "width_2": [16, 32, 64, 128],
    "width_3": [16, 32, 64, 128],
    "width_4": [16, 32, 64, 128],
    "activation": ["relu", "leaky_relu", "elu"],
    "dropout": [0.0, 0.1, 0.3],
    "layer_norm": [False, True],
    "use_residual": [False, True],
}

GENE_NAMES: List[str] = list(SEARCH_SPACE.keys())
GENE_VALUES: List[List[Any]] = list(SEARCH_SPACE.values())
NUM_GENES: int = len(GENE_NAMES)
SCHEMA_VERSION = 3


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
