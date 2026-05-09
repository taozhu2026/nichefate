import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "m3_v2_06_run_full_by_shard.py"
SPEC = importlib.util.spec_from_file_location("m3_v2_full_runner", RUNNER_PATH)
m3_v2_full = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m3_v2_full
SPEC.loader.exec_module(m3_v2_full)


def base_config(tmp_path: Path) -> dict:
    mode_schema = tmp_path / "mode_schema.json"
    mode_schema.write_text(
        json.dumps(
            {
                "mode_name": "constrained_v1prior_sharpening",
                "validated_pseudo_only_parameters": {
                    "lambda": 1.0,
                    "tau_scale": 0.5,
                    "top_k": 10,
                    "G_barcode": 1.0,
                    "row_normalization": "per_source_anchor",
                },
            }
        )
    )
    output_root = tmp_path / "m3_v2"
    return {
        "mode": {
            "schema": str(mode_schema),
            "mode_name": "constrained_v1prior_sharpening",
            "locked_parameters": {
                "lambda": 1.0,
                "tau_scale": 0.5,
                "top_k": 10,
                "G_barcode": 1.0,
                "row_normalization": "per_source_anchor",
            },
        },
        "paths": {
            "output_root": str(output_root),
            "full_by_shard_dir": str(output_root / "full_by_shard"),
            "reports_dir": str(output_root / "reports"),
            "logs_dir": str(output_root / "logs"),
            "figures_dir": str(output_root / "reports" / "figures"),
        },
        "inputs": {},
        "protected_roots": [
            str(tmp_path / "m3"),
            str(tmp_path / "m4a"),
            str(tmp_path / "m4b"),
            str(tmp_path / "m4c"),
            str(tmp_path / "m3_v2_pilot"),
        ],
        "expected": {
            "candidate_k": 30,
            "full_shards": 1,
            "full_sources": 2,
            "v1_candidate_edges": 60,
            "retained_top10_edges": 20,
            "time_pairs": ["D0_to_D3"],
        },
    }


def toy_edges() -> pd.DataFrame:
    rows = []
    for source_idx in range(2):
        for target_idx, distance in enumerate([0.0, 1.0, 2.0]):
            rows.append(
                {
                    "source_anchor_id": f"s{source_idx}",
                    "target_anchor_id": f"t{source_idx}_{target_idx}",
                    "source_slice_id": "source_slice",
                    "target_slice_id": f"target_slice_{target_idx}",
                    "source_mouse_id": "m0",
                    "target_mouse_id": f"m{target_idx}",
                    "row_normalized_transition_prob": [0.6, 0.3, 0.1][target_idx],
                    "v2_d_state": distance,
                    "v2_tau_state": 1.0,
                    "v2_g_composition": [1.0, 0.8, 0.2][target_idx],
                    "v2_g_spatial_topology": [1.0, 0.7, 0.2][target_idx],
                    "v2_g_slice_mouse": 1.0,
                }
            )
    return pd.DataFrame(rows)


def toy_v1_edges() -> pd.DataFrame:
    rows = []
    for source_idx in range(2):
        for target_idx in range(3):
            rows.append(
                {
                    "source_anchor_id": f"s{source_idx}",
                    "target_anchor_id": f"t{source_idx}_{target_idx}",
                    "source_time": "D0",
                    "target_time": "D3",
                    "source_slice_id": "source_slice",
                    "target_slice_id": f"target_slice_{target_idx}",
                    "source_mouse_id": "m0",
                    "target_mouse_id": f"m{target_idx}",
                    "row_normalized_transition_prob": [0.6, 0.3, 0.1][target_idx],
                }
            )
    return pd.DataFrame(rows)[m3_v2_full.M3_V1_EDGE_COLUMNS]


def toy_plan(tmp_path: Path) -> pd.DataFrame:
    output = tmp_path / "m3_v2" / "full_by_shard" / "D0_to_D3" / "source_slice"
    return pd.DataFrame(
        [
            {
                "shard_id": "m3_v2_full_0001",
                "time_pair": "D0_to_D3",
                "source_time": "D0",
                "target_time": "D3",
                "source_slice_id": "source_slice",
                "m3_v1_shard_path": str(tmp_path / "v1.parquet"),
                "m3_v1_row_count": 60,
                "source_count": 2,
                "retained_v2_edge_count": 20,
                "candidate_k": 30,
                "top_k": 10,
                "m3_v2_output_dir": str(output),
                "m3_v2_output_parquet": str(output / "candidate_edges_D0_to_D3__source_slice.parquet"),
                "m3_v2_shard_qc_json": str(output / "candidate_edges_D0_to_D3__source_slice_qc.json"),
            }
        ]
    )


def toy_shard(tmp_path: Path) -> pd.Series:
    output = tmp_path / "m3_v2" / "full_by_shard" / "D0_to_D3" / "source_slice"
    v1 = tmp_path / "v1.parquet"
    toy_v1_edges().to_parquet(v1, index=False)
    return pd.Series(
        {
            "shard_id": "m3_v2_full_0001",
            "time_pair": "D0_to_D3",
            "source_time": "D0",
            "target_time": "D3",
            "source_slice_id": "source_slice",
            "m3_v1_shard_path": str(v1),
            "m3_v1_row_count": 6,
            "source_count": 2,
            "retained_v2_edge_count": 4,
            "candidate_k": 3,
            "top_k": 2,
            "m3_v2_output_dir": str(output),
            "m3_v2_output_parquet": str(output / "candidate_edges_D0_to_D3__source_slice.parquet"),
            "m3_v2_shard_qc_json": str(tmp_path / "m3_v2" / "logs" / "shard_qc" / "qc.json"),
        }
    )


def test_config_parsing_and_mode_schema_loading(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    schema, params = m3_v2_full.load_mode_schema(config)

    assert schema["mode_name"] == "constrained_v1prior_sharpening"
    assert params.lambda_value == pytest.approx(1.0)
    assert params.tau_scale == pytest.approx(0.5)
    assert params.top_k == 10
    assert params.g_barcode == pytest.approx(1.0)


def test_toy_top10_reweighting_and_row_normalization() -> None:
    params = m3_v2_full.LockedParameters(1.0, 0.5, 2, 1.0, "per_source_anchor")
    retained, qc = m3_v2_full.reweight_edges_with_components(toy_edges(), params)

    assert qc["row_sum_pass"]
    assert len(retained) == 4
    assert retained.groupby("source_anchor_id").size().eq(2).all()
    sums = retained.groupby("source_anchor_id")["v2_row_normalized_transition_prob"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_guarded_production_writes_only_versioned_v2_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = base_config(tmp_path)
    shard = toy_shard(tmp_path)
    params = m3_v2_full.LockedParameters(1.0, 0.5, 2, 1.0, "per_source_anchor")
    args = Namespace(resume=True, overwrite=False, stop_on_error=True)

    def fake_components(edges: pd.DataFrame, _config: dict) -> tuple[pd.DataFrame, dict]:
        out = edges.copy()
        out["v2_d_state"] = [0.0, 1.0, 2.0] * 2
        out["v2_tau_state"] = 1.0
        out["v2_g_composition"] = [1.0, 0.8, 0.2] * 2
        out["v2_g_spatial_topology"] = [1.0, 0.7, 0.2] * 2
        out["v2_g_slice_mouse"] = 1.0
        return out, {"unique_source_anchors": 2, "unique_target_anchors": 6}

    monkeypatch.setattr(m3_v2_full, "compute_v2_components", fake_components)

    record, failure = m3_v2_full.run_one_shard(shard, config, params, args)

    assert failure is None
    assert record is not None
    output = Path(shard["m3_v2_output_parquet"])
    assert output.is_file()
    assert str(output).startswith(str((tmp_path / "m3_v2").resolve()))
    assert Path(shard["m3_v2_shard_qc_json"]).is_file()
    frame = pd.read_parquet(output)
    assert len(frame) == 4
    assert frame.groupby("source_anchor_id").size().max() <= 2


def test_dry_run_outputs_do_not_write_shard_parquet(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    schema, params = m3_v2_full.load_mode_schema(config)
    plan = m3_v2_full.add_resume_status(toy_plan(tmp_path), resume=False, overwrite=False)
    args = Namespace(dry_run=True, max_shards=1)

    outputs = m3_v2_full.write_dryrun_outputs(config, plan, plan, schema, params, {}, args)

    assert outputs["plan_csv"].is_file()
    assert outputs["summary_json"].is_file()
    assert not list((tmp_path / "m3_v2" / "full_by_shard").glob("**/candidate_edges_*.parquet"))


def test_path_separation_uses_real_ancestors_not_string_prefix(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    config["paths"]["output_root"] = "/home/zhutao/scratch/nichefate/m3_v2"
    config["paths"]["full_by_shard_dir"] = "/home/zhutao/scratch/nichefate/m3_v2/full_by_shard"
    config["paths"]["reports_dir"] = "/home/zhutao/scratch/nichefate/m3_v2/reports"
    config["paths"]["logs_dir"] = "/home/zhutao/scratch/nichefate/m3_v2/logs"
    config["paths"]["figures_dir"] = "/home/zhutao/scratch/nichefate/m3_v2/reports/figures"
    config["protected_roots"] = ["/home/zhutao/scratch/nichefate/m3"]

    m3_v2_full.validate_output_path_separation(config)

    config["paths"]["output_root"] = "/home/zhutao/scratch/nichefate/m3/full_by_shard"
    with pytest.raises(ValueError):
        m3_v2_full.validate_output_path_separation(config)


def test_resume_plan_behavior_reports_valid_existing_shard(tmp_path: Path) -> None:
    plan = toy_plan(tmp_path)
    out = Path(plan.iloc[0]["m3_v2_output_parquet"])
    out.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "source_anchor_id": [f"s{i // 10}" for i in range(20)],
            "target_anchor_id": [f"t{i}" for i in range(20)],
            "source_slice_id": "source_slice",
            "target_slice_id": "target_slice",
            "source_mouse_id": "m0",
            "target_mouse_id": "m1",
            "v1_row_normalized_transition_prob": 0.1,
            "v2_row_normalized_transition_prob": 0.1,
            "v2_unnormalized_weight": 1.0,
            "v2_rank_within_source": list(range(1, 11)) * 2,
        }
    )
    frame.to_parquet(out, index=False)

    result = m3_v2_full.add_resume_status(plan, resume=True, overwrite=False)

    assert result.iloc[0]["resume_status"] == "SKIP_VALID_EXISTING"


def test_missing_required_columns_failure(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"source_anchor_id": ["s1"]}).to_parquet(path, index=False)

    with pytest.raises(ValueError):
        m3_v2_full.validate_parquet_columns(path, m3_v2_full.M3_V1_REQUIRED_COLUMNS, "toy")


def test_no_forbidden_downstream_outputs_are_declared(tmp_path: Path) -> None:
    config = base_config(tmp_path)

    serialized = json.dumps(config["paths"], sort_keys=True).lower()
    assert "m4a_v2" not in serialized
    assert "m4c_v2" not in serialized
    m3_v2_full.validate_output_path_separation(config)


def test_ssd_output_path_rejection(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    config["paths"]["output_root"] = "/ssd/nichefate/m3_v2"
    config["paths"]["full_by_shard_dir"] = "/ssd/nichefate/m3_v2/full_by_shard"
    config["paths"]["reports_dir"] = "/ssd/nichefate/m3_v2/reports"
    config["paths"]["logs_dir"] = "/ssd/nichefate/m3_v2/logs"
    config["paths"]["figures_dir"] = "/ssd/nichefate/m3_v2/reports/figures"

    with pytest.raises(ValueError):
        m3_v2_full.validate_output_path_separation(config)


def test_m3_v2_is_allowed_as_distinct_from_m3() -> None:
    assert not m3_v2_full.paths_overlap(
        Path("/home/zhutao/scratch/nichefate/m3_v2"),
        Path("/home/zhutao/scratch/nichefate/m3"),
    )
