"""Backward-compatible facade for the PlanA-K M0-M2.5 backbone.

The implementation has been split into focused ``nichefate.planA_k`` modules.
This facade deliberately exposes only the stable spatial niche construction,
representation, and metaniche coarsening helpers included in the M0-M2.5
backbone freeze. Downstream Kmix_A, GPCCA, terminal-state, absorption, and
DARLIN integration layers are separate and are not exported here.
"""

from __future__ import annotations

from nichefate.planA_k.schemas import *
from nichefate.planA_k.io import *
from nichefate.planA_k.reporting import *
from nichefate.planA_k.validation import *
from nichefate.planA_k.kernel_qc import *
from nichefate.planA_k.m2_inventory import *
from nichefate.planA_k.metaniche import *
from nichefate.planA_k.coordinates import *
from nichefate.planA_k.rare_state import *
from nichefate.planA_k.production_preflight import *
from nichefate.planA_k.full_m2_5_production import *

__all__ = [name for name in globals() if not name.startswith("__")]
