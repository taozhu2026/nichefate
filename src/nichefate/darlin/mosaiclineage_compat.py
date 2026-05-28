from __future__ import annotations

from typing import Any


def mosaiclineage_available() -> bool:
    try:
        import mosaiclineage  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def compatibility_summary() -> dict[str, Any]:
    return {
        "mosaiclineage_available": mosaiclineage_available(),
        "joint_clone_unit": "validated_darlin_style_joint_clone",
        "reference_bank_role": "allele_level_qc_metadata",
        "de_novo_role": "allele_level_qc_metadata",
    }


__all__ = ["compatibility_summary", "mosaiclineage_available"]
