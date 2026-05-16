from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (
    audit_coordinate_join_keys,
    compute_spatial_compactness_qc,
    load_m1_coordinates_for_slices,
    rare_state_preservation_audit,
    run_coordinate_join_preview,
    stratified_pilot_design_payload,
)


def write_tiny_pilot_and_m1(tmp_path: Path) -> tuple[Path, Path, Path]:
    pilot_root = tmp_path / "pilot_outputs"
    pilot_root.mkdir(parents=True)
    m1_root = tmp_path / "m1" / "by_slice"
    rows = []
    coord_rows = []
    for slice_id, day, x_offset in [("slice_D0", 0, 0.0), ("slice_D3", 3, 100.0)]:
        slice_dir = m1_root / slice_id
        slice_dir.mkdir(parents=True)
        for idx in range(12):
            cell_id = f"{slice_id}_cell_{idx}"
            metaniche = "MN_A" if idx < 6 else "MN_B"
            rows.append(
                {
                    "anchor_id": f"{slice_id}::{idx}",
                    "metaniche_id": metaniche,
                    "slice_id": slice_id,
                    "time": f"D{day}",
                    "time_day": day,
                    "mouse_id": f"m{day}",
                    "anchor_index": idx,
                    "anchor_cell_id": cell_id,
                    "cell_type_l1": "rare" if idx == 0 else "common",
                    "cell_type_l2": "rare" if idx == 0 else "common",
                    "cell_type_l3": "rare" if idx == 0 else "common",
                    "source_m2_path": "tiny",
                }
            )
            for scale in ["radius_x2", "radius_x4", "radius_x8"]:
                coord_rows.append(
                    {
                        "slice_id": slice_id,
                        "scale": scale,
                        "anchor_index": idx,
                        "anchor_cell_id": cell_id,
                        "time": f"D{day}",
                        "time_day": day,
                        "mouse_id": f"m{day}",
                        "cell_type_l1": "rare" if idx == 0 else "common",
                        "cell_type_l2": "rare" if idx == 0 else "common",
                        "cell_type_l3": "rare" if idx == 0 else "common",
                        "x": x_offset + idx,
                        "y": float(idx % 3),
                    }
                )
        pd.DataFrame(coord_rows).query("slice_id == @slice_id").to_parquet(
            slice_dir / f"niche_features_{slice_id}.parquet",
            index=False,
        )
    pd.DataFrame(rows).to_csv(pilot_root / "anchor_to_metaniche.tsv", sep="\t", index=False)
    pd.DataFrame({"metaniche_id": ["MN_A", "MN_B"], "anchor_count": [12, 12]}).to_csv(
        pilot_root / "metaniche_table.tsv",
        sep="\t",
        index=False,
    )
    return pilot_root, m1_root, tmp_path / "reports"


def test_m1_coordinate_loading_deduplicates_scales(tmp_path: Path) -> None:
    _, m1_root, _ = write_tiny_pilot_and_m1(tmp_path)

    coords = load_m1_coordinates_for_slices(["slice_D0", "slice_D3"], m1_root=m1_root)

    assert len(coords) == 24
    assert {"x", "y", "anchor_index", "anchor_cell_id"}.issubset(coords.columns)
    assert set(coords["scale"]) == {"radius_x2"}


def test_join_key_audit_and_coordinate_preview(tmp_path: Path) -> None:
    pilot_root, m1_root, output_dir = write_tiny_pilot_and_m1(tmp_path)
    join_frame, join_summary = audit_coordinate_join_keys(pilot_root=pilot_root, m1_root=m1_root)
    payload = run_coordinate_join_preview(
        output_dir=output_dir,
        pilot_root=pilot_root,
        m1_root=m1_root,
        overwrite=True,
        dry_run=False,
    )

    assert join_summary["safe_join_identified"] is True
    assert "slice_id+anchor_index+anchor_cell_id" in set(join_frame["candidate_key"])
    assert payload["safe_join_identified"] is True
    assert (output_dir / "coordinate_join_preview" / "anchor_coordinates.preview.tsv").exists()


def test_spatial_compactness_and_rare_state_audit(tmp_path: Path) -> None:
    pilot_root, m1_root, output_dir = write_tiny_pilot_and_m1(tmp_path)
    run_coordinate_join_preview(
        output_dir=output_dir,
        pilot_root=pilot_root,
        m1_root=m1_root,
        overwrite=True,
        dry_run=False,
    )
    spatial_frame, spatial_payload = compute_spatial_compactness_qc(output_dir)
    rare_frame, rare_payload = rare_state_preservation_audit(pilot_root=pilot_root)

    assert spatial_payload["spatial_compactness_available"] is True
    assert len(spatial_frame) == 2
    assert rare_payload["rare_state_audit_available"] is True
    assert {"label_column", "label_value", "collapsed_warning"}.issubset(rare_frame.columns)


def test_stratified_design_is_per_slice() -> None:
    text, payload = stratified_pilot_design_payload()

    assert "per-slice" in payload["recommended_strategy"]
    assert payload["max_slices"] == 4
    assert "per-slice coarsening" in text
