"""nichefate package.

The package root stays dependency-light so lineage-aware, barcode-aware, and
niche-model subpackages remain importable in minimal environments.
"""

PROJECT_NAME = "nichefate"
DATASET_ID = "merfish_colitis_moffitt_2024"

__all__ = ["DATASET_ID", "PROJECT_NAME"]

__version__ = "0.1.0"
