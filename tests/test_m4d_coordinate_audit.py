import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4d_00_audit_coordinate_availability.py"
SPEC = importlib.util.spec_from_file_location("m4d_coordinate_audit", SCRIPT_PATH)
m4d = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4d
SPEC.loader.exec_module(m4d)


def toy_node_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_node_index": [0, 1, 2],
            "anchor_id": ["s0::0", "s0::1", "s1::0"],
            "slice_id": ["s0", "s0", "s1"],
            "anchor_index": [0, 1, 0],
            "anchor_cell_id": ["c0", "c1", "c2"],
        }
    )


def toy_m1_rows(nonidentical: bool = False) -> pd.DataFrame:
    rows = []
    for slice_id, anchor_index, cell_id, x, y in [
        ("s0", 0, "c0", 1.0, 10.0),
        ("s0", 1, "c1", 2.0, 20.0),
        ("s1", 0, "c2", 4.0, 40.0),
    ]:
        for scale in ["radius_x2", "radius_x4", "radius_x8"]:
            dx = 0.5 if nonidentical and slice_id == "s0" and anchor_index == 1 and scale == "radius_x4" else 0.0
            rows.append(
                {
                    "slice_id": slice_id,
                    "anchor_index": anchor_index,
                    "anchor_cell_id": cell_id,
                    "scale": scale,
                    "x": x + dx,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def test_m1_coordinates_reduce_only_after_cross_scale_identity_check() -> None:
    reduced, summary = m4d.reduce_m1_coordinates(
        toy_m1_rows(),
        Path("/tmp/niche_features_s0.parquet"),
        "radius_x2",
        1e-9,
    )

    assert len(reduced) == 3
    assert summary["nonidentical_coordinate_anchors"] == 0
    assert reduced["coordinate_scale_used"].unique().tolist() == ["radius_x2"]
    assert {"coordinate_source", "coordinate_source_path", "coordinate_join_key"} <= set(reduced.columns)


def test_nonidentical_scale_coordinates_fail_clearly() -> None:
    with pytest.raises(ValueError, match="not identical across scale rows"):
        m4d.reduce_m1_coordinates(
            toy_m1_rows(nonidentical=True),
            Path("/tmp/niche_features_s0.parquet"),
            "radius_x2",
            1e-9,
        )


def test_coordinate_cache_has_one_row_per_m4a_node_and_provenance_columns() -> None:
    reduced, _ = m4d.reduce_m1_coordinates(
        toy_m1_rows(),
        Path("/tmp/niche_features_s0.parquet"),
        "radius_x2",
        1e-9,
    )

    cache = m4d.build_coordinate_cache(toy_node_table(), reduced)
    qc = m4d.cache_qc(cache, expected_nodes=3)

    assert len(cache) == 3
    assert cache["global_node_index"].tolist() == [0, 1, 2]
    assert cache["global_node_index"].is_unique
    assert qc["matched_nodes"] == 3
    assert qc["missing_coordinates"] == 0
    assert qc["tissue_space_maps_enabled"]
    assert not qc["cross_time_physical_arrows_allowed"]
    assert not qc["tissue_space_arrows_allowed"]
    assert {
        "coordinate_source",
        "coordinate_source_path",
        "coordinate_scale_used",
        "coordinate_join_key",
        "coordinate_join_status",
    } <= set(cache.columns)


def test_missing_coordinates_are_reported_without_changing_row_count() -> None:
    reduced, _ = m4d.reduce_m1_coordinates(
        toy_m1_rows().query("slice_id != 's1'"),
        Path("/tmp/niche_features_s0.parquet"),
        "radius_x2",
        1e-9,
    )

    cache = m4d.build_coordinate_cache(toy_node_table(), reduced)
    qc = m4d.cache_qc(cache, expected_nodes=3)

    assert len(cache) == 3
    assert qc["matched_nodes"] == 2
    assert qc["missing_coordinates"] == 1
    assert not qc["tissue_space_maps_enabled"]
    assert qc["state_space_only_visualization"]
    assert cache.loc[cache["global_node_index"] == 2, "coordinate_join_status"].iloc[0] == "missing"


def test_per_slice_centered_and_scaled_coordinates_are_computed() -> None:
    reduced, _ = m4d.reduce_m1_coordinates(
        toy_m1_rows(),
        Path("/tmp/niche_features_s0.parquet"),
        "radius_x2",
        1e-9,
    )
    cache = m4d.build_coordinate_cache(toy_node_table(), reduced)

    s0 = cache.loc[cache["slice_id"] == "s0"].sort_values("anchor_index")
    assert s0["x_centered_by_slice"].tolist() == pytest.approx([-0.5, 0.5])
    assert np.isfinite(cache["x_scaled_by_slice"].dropna()).all()
    assert np.isfinite(cache["y_scaled_by_slice"].dropna()).all()


def test_validate_m4a_node_table_requires_expected_identity() -> None:
    table = toy_node_table()
    validated = m4d.validate_m4a_node_table(table, expected_nodes=3)

    assert validated["global_node_index"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="Expected 4 M4A nodes"):
        m4d.validate_m4a_node_table(table, expected_nodes=4)


def test_inspect_sources_prefers_m1_and_skips_m0_h5ad_when_m1_has_coordinates(tmp_path: Path) -> None:
    m4a = tmp_path / "m4a.parquet"
    m4c = tmp_path / "m4c.parquet"
    m2_dir = tmp_path / "m2"
    m1_dir = tmp_path / "m1"
    m2_slice = m2_dir / "slice"
    m1_slice = m1_dir / "slice"
    m2_slice.mkdir(parents=True)
    m1_slice.mkdir(parents=True)
    pd.DataFrame({"global_node_index": [0], "slice_id": ["s0"], "anchor_index": [0]}).to_parquet(m4a)
    pd.DataFrame({"global_node_index": [0], "slice_id": ["s0"], "anchor_index": [0]}).to_parquet(m4c)
    pd.DataFrame({"slice_id": ["s0"], "anchor_index": [0], "anchor_cell_id": ["c0"]}).to_parquet(
        m2_slice / "m2_representation_slice.parquet"
    )
    toy_m1_rows().to_parquet(m1_slice / "niche_features_slice.parquet")
    h5ad = tmp_path / "heavy.h5ad"
    h5ad.write_text("not loaded\n", encoding="utf-8")

    audit = m4d.inspect_sources(
        {
            "m4a_node_table": m4a,
            "m4c_node_summary": m4c,
            "m2_by_slice_root": m2_dir,
            "m1_by_slice_root": m1_dir,
            "m0_final_h5ad": h5ad,
        }
    )

    m1_row = audit.loc[audit["source_priority"] == "4_m1_by_slice_niche_features"].iloc[0]
    m0_row = audit.loc[audit["source_priority"] == "5_m0_h5ad_fallback"].iloc[0]
    assert m1_row["selected_coordinate_source"]
    assert m1_row["available_coordinate_columns"] == "x;y"
    assert m0_row["status"] == "skipped_cheaper_m1_coordinates_available"


def test_cache_contract_validation_rejects_non_unique_or_misaligned_index() -> None:
    cache = pd.DataFrame({"global_node_index": [0, 0], "coordinate_join_status": ["matched", "matched"]})
    qc = {
        "expected_rows": 2,
        "cache_rows": 2,
        "global_node_index_unique": False,
        "global_node_index_aligned": False,
    }

    with pytest.raises(ValueError, match="global_node_index must be unique"):
        m4d.validate_cache_contract(cache, qc)


def test_no_forbidden_downstream_flags_are_declared() -> None:
    assert m4d.NO_DOWNSTREAM_FLAGS["no_gpcca"] is True
    assert m4d.NO_DOWNSTREAM_FLAGS["no_branched_nicheflow_training"] is True
    assert m4d.NO_DOWNSTREAM_FLAGS["no_m5"] is True
    assert m4d.NO_DOWNSTREAM_FLAGS["no_regulator_analysis"] is True
