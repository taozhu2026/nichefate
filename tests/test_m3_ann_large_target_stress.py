import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m3_13_stress_test_ann_large_target_pair.py"
SPEC = importlib.util.spec_from_file_location("m3_ann_large_target_stress", SCRIPT_PATH)
m3_stress = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m3_stress
SPEC.loader.exec_module(m3_stress)


def toy_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "use_ssd": False,
            "m3_output_dir": str(tmp_path / "m3"),
        },
        "full_m3": {
            "output_root": str(tmp_path / "m3" / "by_pair"),
            "max_memory_gb_warning": 80,
        },
    }


def toy_args(tmp_path: Path, sample_size: int = 3000) -> Namespace:
    return Namespace(
        source_time="D3",
        target_time="D9",
        source_slice_id="092421_D3_m3_1_slice_3",
        sample_source_anchors=sample_size,
        candidate_k=30,
        exact_backend="sklearn_exact",
        ann_backend="pynndescent",
        output_dir=tmp_path / "ann_stress_D3_to_D9",
        random_seed=1,
        allow_larger_sample=False,
        allow_non_default_pair=False,
        allow_non_default_shard=False,
        allow_non_default_output_dir=True,
        fallback_source_anchors=None,
    )


def stress_shard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_time": "D3",
                "target_time": "D9",
                "source_slice_id": "092421_D3_m3_1_slice_3",
                "source_slice_file": "092421_D3_m3_1_slice_3.m0.h5ad",
                "source_rows": 21962,
                "target_time_rows": 660977,
                "candidate_k": 30,
                "expected_edge_rows": 658860,
                "time_delta": 6.0,
            }
        ]
    )


def test_fixed_output_names_are_used(tmp_path: Path) -> None:
    paths = m3_stress.output_paths(tmp_path)

    assert paths["report"].name == "ann_stress_report_D3_to_D9.md"
    assert paths["metrics"].name == "ann_stress_metrics_D3_to_D9.csv"
    assert paths["summary"].name == "ann_stress_summary_D3_to_D9.json"
    assert paths["overlap"].name == "ann_stress_candidate_overlap_D3_to_D9.csv"
    assert paths["figures_dir"].name == "figures"


def test_requested_scope_accepts_default_d3_d9(tmp_path: Path) -> None:
    m3_stress.validate_requested_scope(toy_args(tmp_path), toy_config(tmp_path))


def test_requested_scope_rejects_non_d3_d9_pair(tmp_path: Path) -> None:
    args = toy_args(tmp_path)
    args.source_time = "D9"
    args.target_time = "D14"

    with pytest.raises(ValueError, match="D3->D9"):
        m3_stress.validate_requested_scope(args, toy_config(tmp_path))


def test_requested_scope_allows_explicit_pair_override(tmp_path: Path) -> None:
    args = toy_args(tmp_path)
    args.source_time = "D9"
    args.target_time = "D14"
    args.allow_non_default_pair = True

    m3_stress.validate_requested_scope(args, toy_config(tmp_path))


def test_requested_scope_rejects_larger_sample_without_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="5000"):
        m3_stress.validate_requested_scope(toy_args(tmp_path, sample_size=5001), toy_config(tmp_path))


def test_requested_scope_rejects_wrong_backends(tmp_path: Path) -> None:
    args = toy_args(tmp_path)
    args.exact_backend = "numpy_chunked"
    with pytest.raises(ValueError, match="sklearn_exact"):
        m3_stress.validate_requested_scope(args, toy_config(tmp_path))

    args = toy_args(tmp_path)
    args.ann_backend = "faiss"
    with pytest.raises(ValueError, match="pynndescent"):
        m3_stress.validate_requested_scope(args, toy_config(tmp_path))


def test_output_directory_guards_reject_production_ssd_and_downstream_names(tmp_path: Path) -> None:
    config = toy_config(tmp_path)

    with pytest.raises(ValueError, match="production M3"):
        m3_stress.ensure_validation_output_dir(tmp_path / "m3" / "by_pair" / "x", config)
    with pytest.raises(ValueError, match="/ssd"):
        m3_stress.ensure_validation_output_dir(Path("/ssd/zhutao/tmp/ann_stress_D3_to_D9"), config)
    with pytest.raises(ValueError, match="gpcca"):
        m3_stress.ensure_validation_output_dir(tmp_path / "m3" / "gpcca_ann_stress", config)


def test_project_name_containing_fate_is_allowed(tmp_path: Path) -> None:
    config = toy_config(tmp_path)

    m3_stress.ensure_validation_output_dir(tmp_path / "nichefate" / "m3" / "ann_stress_D3_to_D9", config)


def test_stress_shard_metadata_is_checked() -> None:
    shard = m3_stress.select_stress_shard(
        stress_shard(),
        "D3",
        "D9",
        "092421_D3_m3_1_slice_3",
    )

    assert shard["source_slice_id"] == "092421_D3_m3_1_slice_3"


def test_stress_shard_metadata_mismatch_is_rejected() -> None:
    shards = stress_shard()
    shards.loc[0, "target_time_rows"] = 12

    with pytest.raises(ValueError, match="target_time_rows"):
        m3_stress.select_stress_shard(shards, "D3", "D9", "092421_D3_m3_1_slice_3")


def test_exact_reference_safety_guard_allows_default_workload() -> None:
    safety = m3_stress.exact_reference_safety_guard(3000, 660977, 80.0)

    assert safety.should_run
    assert safety.actual_source_sample_size == 3000
    assert not safety.fallback_sample_used
    assert safety.pairwise_distance_evaluations == 3000 * 660977
    assert safety.estimated_dense_distance_gib < 80.0


def test_exact_reference_safety_guard_skips_when_too_large() -> None:
    safety = m3_stress.exact_reference_safety_guard(3000, 660977, 1.0)

    assert not safety.should_run
    assert safety.actual_source_sample_size == 0
    assert not safety.fallback_sample_used


def test_exact_reference_safety_guard_uses_explicit_fallback() -> None:
    safety = m3_stress.exact_reference_safety_guard(3000, 660977, 1.0, fallback_source_anchors=100)

    assert safety.should_run
    assert safety.actual_source_sample_size == 100
    assert safety.fallback_sample_used
    assert safety.estimated_dense_distance_gib < 1.0


def test_stop_outputs_write_report_summary_metrics_and_overlap(tmp_path: Path) -> None:
    paths = m3_stress.output_paths(tmp_path)
    safety = m3_stress.exact_reference_safety_guard(3000, 660977, 1.0)
    args = toy_args(tmp_path)
    shard = m3_stress.select_stress_shard(
        stress_shard(),
        "D3",
        "D9",
        "092421_D3_m3_1_slice_3",
    )
    context = m3_stress.base_context(args, shard, safety)

    m3_stress.write_stop_outputs(paths, "SKIPPED", safety.reason, context)

    assert paths["report"].exists()
    assert paths["metrics"].exists()
    assert paths["overlap"].exists()
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["status"] == "SKIPPED"
    assert summary["no_production_m3_edges"]
    assert summary["no_global_markov_p"]


def test_no_global_or_downstream_output_paths_are_produced(tmp_path: Path) -> None:
    paths = m3_stress.output_paths(tmp_path)
    text = "\n".join(path.name.lower() for path in paths.values())

    for token in ["markov", "gpcca", "fate", "branched", "nicheflow", "m5", "regulator"]:
        assert token not in text
