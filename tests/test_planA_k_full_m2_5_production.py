from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k.full_m2_5_production import (
    FullM25Params,
    adaptive_metaniche_count,
    discover_slice_inputs,
    load_feature_lock,
    run_full_m2_5,
    validate_coordinate_join_for_slice,
    validate_no_ssd_path,
    validate_output_root,
)


def write_feature_lock(path: Path, feature_columns: list[str]) -> Path:
    payload = {
        "config_version": "test_feature_lock",
        "metadata_columns": [
            "slice_id",
            "slice_file",
            "time",
            "time_day",
            "mouse_id",
            "anchor_index",
            "anchor_cell_id",
            "cell_type_l1",
            "cell_type_l2",
            "cell_type_l3",
        ],
        "feature_columns": feature_columns,
        "feature_column_count": len(feature_columns),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_tiny_inputs(tmp_path: Path, rows_per_slice: int = 24) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(7)
    m1_root = tmp_path / "m1" / "by_slice"
    m2_root = tmp_path / "m2" / "by_slice"
    feature_columns = [
        "radius_x2__ct_l1__epithelial",
        "radius_x2__ct_l1__immune",
        "radius_x2__entropy",
        "radius_x2__emb_mean_pc001",
    ]
    for idx, day in enumerate([0, 3]):
        slice_id = f"tiny_D{day}_slice_{idx}"
        m1_dir = m1_root / slice_id
        m2_dir = m2_root / slice_id
        m1_dir.mkdir(parents=True)
        m2_dir.mkdir(parents=True)
        anchor_index = np.arange(rows_per_slice)
        anchor_cell_id = [f"{slice_id}_cell_{i}" for i in anchor_index]
        labels = np.where(anchor_index < rows_per_slice // 2, "epithelial", "immune")
        m2 = pd.DataFrame(
            {
                "slice_id": slice_id,
                "slice_file": f"{slice_id}.h5ad",
                "time": f"D{day}",
                "time_day": day,
                "mouse_id": f"m{idx}",
                "anchor_index": anchor_index,
                "anchor_cell_id": anchor_cell_id,
                "cell_type_l1": labels,
                "cell_type_l2": labels,
                "cell_type_l3": np.where(anchor_index == 0, "rare", "common"),
                feature_columns[0]: (labels == "epithelial").astype(float),
                feature_columns[1]: (labels == "immune").astype(float),
                feature_columns[2]: rng.random(rows_per_slice),
                feature_columns[3]: rng.normal(size=rows_per_slice),
            }
        )
        coords = []
        for scale in ["radius_x2", "radius_x4", "radius_x8"]:
            coords.append(
                pd.DataFrame(
                    {
                        "slice_id": slice_id,
                        "scale": scale,
                        "anchor_index": anchor_index,
                        "anchor_cell_id": anchor_cell_id,
                        "x": idx * 100 + anchor_index,
                        "y": anchor_index % 5,
                    }
                )
            )
        m2.to_parquet(m2_dir / f"m2_representation_{slice_id}.parquet", index=False)
        pd.concat(coords, ignore_index=True).to_parquet(
            m1_dir / f"niche_features_{slice_id}.parquet",
            index=False,
        )
    lock = write_feature_lock(tmp_path / "feature_lock.json", feature_columns)
    return m1_root, m2_root, lock


def params(tmp_path: Path, **overrides) -> FullM25Params:
    m1_root, m2_root, lock = write_tiny_inputs(tmp_path)
    data = {
        "feature_lock": lock,
        "output_root": tmp_path / "outputs",
        "seed": 11,
        "m1_root": m1_root,
        "m2_root": m2_root,
        "dry_run": True,
        "smoke_test": False,
        "max_slices": None,
        "max_anchors_per_slice": None,
        "overwrite": False,
        "resume": False,
        "n_pca_components": 2,
        "target_mode": "adaptive",
        "min_metaniches_per_slice": 2,
        "max_metaniches_per_slice": 4,
        "base_metaniches_per_slice": 3,
        "tmp_dir": tmp_path / "tmp",
    }
    data.update(overrides)
    return FullM25Params(**data)


def test_feature_lock_loading(tmp_path: Path) -> None:
    p = params(tmp_path)
    lock = load_feature_lock(p.feature_lock)

    assert lock["feature_column_count"] == 4
    assert "slice_id" not in lock["feature_columns"]


def test_adaptive_metaniche_count_estimation() -> None:
    assert adaptive_metaniche_count(100, 100, base=100, min_count=50, max_count=150) == 100
    assert adaptive_metaniche_count(25, 100, base=100, min_count=50, max_count=150) == 50
    assert adaptive_metaniche_count(400, 100, base=100, min_count=50, max_count=150) == 150
    assert adaptive_metaniche_count(400, 100, base=77, min_count=50, max_count=150, target_mode="fixed") == 77


def test_output_root_and_ssd_guards(tmp_path: Path) -> None:
    p = params(tmp_path, dry_run=False, output_root=tmp_path / "not_production")
    with pytest.raises(ValueError, match="Full production output root"):
        validate_output_root(p)
    with pytest.raises(ValueError, match="Refusing /ssd path"):
        validate_no_ssd_path(Path("/ssd/nichefate/test"), "output-root")


def test_coordinate_join_validation(tmp_path: Path) -> None:
    p = params(tmp_path)
    inventory = discover_slice_inputs(p.m1_root, p.m2_root)
    row = inventory.iloc[0]
    result = validate_coordinate_join_for_slice(Path(row["m2_path"]), Path(row["m1_path"]))

    assert result["valid"] is True
    assert result["join_coverage"] == pytest.approx(1.0)


def test_coordinate_join_duplicate_blocks(tmp_path: Path) -> None:
    p = params(tmp_path)
    inventory = discover_slice_inputs(p.m1_root, p.m2_root)
    row = inventory.iloc[0]
    coords = pd.read_parquet(row["m1_path"])
    duplicated = pd.concat([coords, coords.head(1)], ignore_index=True)
    duplicated.to_parquet(row["m1_path"], index=False)

    result = validate_coordinate_join_for_slice(Path(row["m2_path"]), Path(row["m1_path"]))

    assert result["valid"] is False
    assert result["duplicate_join_key_rows"] > 0


def test_dry_run_does_not_create_output_root(tmp_path: Path) -> None:
    p = params(tmp_path, dry_run=True, output_root=tmp_path / "production_like")
    payload = run_full_m2_5(p)

    assert payload["production_will_run"] is False
    assert not (tmp_path / "production_like").exists()


def test_smoke_test_output_schema(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke"
    p = params(
        tmp_path,
        output_root=output_root,
        dry_run=True,
        smoke_test=True,
        max_slices=2,
        max_anchors_per_slice=12,
        overwrite=True,
    )
    payload = run_full_m2_5(p)

    assert payload["smoke_test"] is True
    assert payload["production_executed"] is False
    assert (output_root / "run_manifest.json").exists()
    assert (output_root / "anchor_to_metaniche.parquet").exists()
    assert (output_root / "metaniche_table.parquet").exists()
    assert (output_root / "metaniche_feature_centroids.parquet").exists()
    assert (output_root / "metaniche_coordinates.tsv").exists()
    assert pd.read_parquet(output_root / "anchor_to_metaniche.parquet")["slice_id"].nunique() == 2


def test_existing_output_overwrite_guard(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("x", encoding="utf-8")
    p = params(tmp_path, output_root=output_root, dry_run=False, smoke_test=True, overwrite=False)

    with pytest.raises(FileExistsError):
        validate_output_root(p)


def test_missing_input_handling(tmp_path: Path) -> None:
    lock = write_feature_lock(tmp_path / "feature_lock.json", ["a", "b"])
    p = FullM25Params(
        feature_lock=lock,
        output_root=tmp_path / "out",
        seed=1,
        m1_root=tmp_path / "missing_m1",
        m2_root=tmp_path / "missing_m2",
        dry_run=True,
        tmp_dir=tmp_path / "tmp",
    )
    payload = run_full_m2_5(p)

    assert "No M2 inputs discovered." in payload["blockers"]
