from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DynamicsInterfaceDesign:
    """Frozen design contract for future clone-aware dynamics modules."""

    cellbin_clone_matrix: str = "C_cellbin_clone"
    tile_clone_matrix: str = "C_tile_clone"
    niche_clone_matrix: str = "C_niche_clone"
    direction_requirement: str = "time_or_perturbation_or_biological_prior"
    l126_limitation: str = "serial_sections_are_not_timepoints"


def freeze_dynamics_interface_design() -> dict[str, Any]:
    """Return the frozen lineage-aware dynamics interface contract."""

    design = DynamicsInterfaceDesign()
    return {
        "design_only": True,
        "objects": [
            design.cellbin_clone_matrix,
            design.tile_clone_matrix,
            design.niche_clone_matrix,
        ],
        "direction_requirement": design.direction_requirement,
        "l126_limitation": design.l126_limitation,
        "as_dict": asdict(design),
    }


__all__ = ["DynamicsInterfaceDesign", "freeze_dynamics_interface_design"]
