"""SpatialDatasetAdapter facade for legacy M0 input preparation.

M0 remains the stable spatial transcriptomics adapter layer. This facade
re-exports the validated dataset IO, metadata, and spatial-coordinate helpers
without changing behavior.
"""

from nichefate.io import *  # noqa: F401,F403
from nichefate.metadata import *  # noqa: F401,F403
from nichefate.spatial import *  # noqa: F401,F403
