"""PlanA-K M0-M2.5 backbone helper modules.

This package export surface is intentionally limited to the stable spatial
niche construction, representation, and metaniche coarsening backbone. Later
Kmix_A, GPCCA, terminal-state, absorption, and DARLIN integration layers remain
outside this M0-M2.5 freeze.
"""

from .schemas import *
from .io import *
from .reporting import *
from .validation import *
from .kernel_qc import *
from .m2_inventory import *
from .metaniche import *
from .coordinates import *
from .rare_state import *
from .production_preflight import *
from .full_m2_5_production import *

__all__ = [name for name in globals() if not name.startswith("__")]
