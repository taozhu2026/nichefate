import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m3_14_freeze_backend_and_dryrun_full_m3.py"
SPEC = importlib.util.spec_from_file_location("m3_backend_freeze_dryrun", SCRIPT_PATH)
m3_14 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m3_14
SPEC.loader.exec_module(m3_14)


def toy_config() -> dict:
    return {
        "paths": {
            "use_ssd": False,
            "m3_output_dir": "/tmp/m3",
        },
        "full_m3": {
            "output_root": "/tmp/m3/by_pair",
            "candidate_k": 30,
        },
    }


def accurate_evidence(runtime_ratio: float = 2.0) -> dict:
    rows = [
        {
            "name": "m3_09_sampled_validation",
            "label": "sampled",
            "path": "sampled.json",
            "present": True,
            "status": "COMPLETED",
            "accuracy_supported": True,
            "recall_at_30_mean": 0.94,
            "top1_agreement": 0.95,
            "jaccard_overlap_mean": 0.92,
            "probability_drift_p95": 0.008,
            "exact_runtime_seconds": 10.0,
            "ann_runtime_seconds": 9.0,
            "runtime_ratio_ann_over_exact": 0.9,
            "exact_max_rss_gib": 3.0,
            "ann_max_rss_gib": 3.0,
            "memory_ratio_ann_over_exact": 1.0,
            "soft_validation_pass": True,
        },
        {
            "name": "m3_12_full_shard_validation",
            "label": "full shard",
            "path": "full.json",
            "present": True,
            "status": "COMPLETED",
            "accuracy_supported": True,
            "recall_at_30_mean": 0.96,
            "top1_agreement": 0.97,
            "jaccard_overlap_mean": 0.94,
            "probability_drift_p95": 0.008,
            "exact_runtime_seconds": 61.0,
            "ann_runtime_seconds": 28.0,
            "runtime_ratio_ann_over_exact": 0.46,
            "exact_max_rss_gib": 14.0,
            "ann_max_rss_gib": 8.0,
            "memory_ratio_ann_over_exact": 0.57,
            "soft_validation_pass": True,
        },
        {
            "name": "m3_13_large_target_stress",
            "label": "large target",
            "path": "stress.json",
            "present": True,
            "status": "COMPLETED",
            "accuracy_supported": True,
            "recall_at_30_mean": 0.958,
            "top1_agreement": 0.966,
            "jaccard_overlap_mean": 0.939,
            "probability_drift_p95": 0.006,
            "exact_runtime_seconds": 47.2,
            "ann_runtime_seconds": 136.4,
            "runtime_ratio_ann_over_exact": runtime_ratio,
            "exact_max_rss_gib": 16.4,
            "ann_max_rss_gib": 16.4,
            "memory_ratio_ann_over_exact": 1.0,
            "soft_validation_pass": True,
        },
    ]
    return {
        "summaries": rows,
        "documents": {
            "slurm_strategy": {"text": "Recommended global concurrency cap: 4"},
            "ann_validation_plan": {"text": ""},
        },
    }


def dryrun_input_shards() -> pd.DataFrame:
    source_rows = [25_000] * 51 + [73_582]
    rows = []
    pairs = [
        ("D0", "D3", 0.0, 3.0),
        ("D3", "D9", 3.0, 9.0),
        ("D9", "D21", 9.0, 21.0),
        ("D21", "D" + "35", 21.0, 35.0),
    ]
    for idx, rows_count in enumerate(source_rows):
        source_time, target_time, source_day, target_day = pairs[idx % len(pairs)]
        rows.append(
            {
                "source_time": source_time,
                "target_time": target_time,
                "source_day": source_day,
                "target_day": target_day,
                "time_delta": target_day - source_day,
                "source_slice_id": f"slice_{idx:03d}",
                "source_slice_file": f"slice_{idx:03d}.m0.h5ad",
                "source_rows": rows_count,
                "target_time_rows": 100_000 + idx,
                "candidate_k": 30,
                "expected_edge_rows": rows_count * 30,
            }
        )
    return pd.DataFrame(rows)


def test_backend_freeze_selects_sklearn_when_ann_is_accurate_but_slower() -> None:
    decision = m3_14.freeze_backend_decision(accurate_evidence(runtime_ratio=2.89), toy_config())

    assert decision["default_backend"] == "sklearn_exact"
    assert decision["optional_backend"] == "pynndescent"
    assert decision["evidence_strength"] == "strongly_supported"


def test_missing_validation_summaries_weaken_evidence_without_crashing(tmp_path: Path) -> None:
    evidence = m3_14.read_evidence(m3_14.default_input_paths(tmp_path / "reports"))
    decision = m3_14.freeze_backend_decision(evidence, toy_config())

    assert decision["default_backend"] == "sklearn_exact"
    assert decision["optional_backend"] == "pynndescent"
    assert decision["evidence_strength"] == "limited_available"
    assert len(decision["missing_evidence"]) == 3


def test_dryrun_shard_table_contains_52_shards_and_expected_rows(tmp_path: Path) -> None:
    dryrun = m3_14.build_dryrun_shards(
        dryrun_input_shards(),
        tmp_path / "full_by_shard",
        "sklearn_exact",
        30,
    )

    assert len(dryrun) == 52
    assert int(dryrun["expected_edge_rows"].sum()) == 40_457_460
    assert set(dryrun["selected_backend"]) == {"sklearn_exact"}


def test_final_target_time_is_not_used_as_source(tmp_path: Path) -> None:
    dryrun = m3_14.build_dryrun_shards(
        dryrun_input_shards(),
        tmp_path / "full_by_shard",
        "sklearn_exact",
        30,
    )

    assert "D" + "35" not in set(dryrun["source_time"].astype(str))


def test_production_edge_paths_are_planned_but_not_created(tmp_path: Path) -> None:
    production_root = tmp_path / "full_by_shard"
    reports_dir = tmp_path / "reports"
    decision = m3_14.freeze_backend_decision(accurate_evidence(), toy_config())
    dryrun = m3_14.build_dryrun_shards(dryrun_input_shards(), production_root, "sklearn_exact", 30)
    expected = m3_14.expected_outputs_payload(dryrun, production_root, reports_dir, decision)
    paths = m3_14.output_paths(reports_dir)

    m3_14.write_outputs(paths, decision, dryrun, expected, production_root)

    assert not list(production_root.glob("**/candidate_edges_*.parquet"))
    assert all(not Path(path).exists() for path in dryrun["output_parquet"].head(3))


def test_pilot_outputs_are_not_marked_as_production_completed(tmp_path: Path) -> None:
    reports_dir = tmp_path / "m3" / "reports"
    pilot_dir = tmp_path / "m3" / ("timepair_pilot_D21_to_" + "D" + "35")
    pilot_dir.mkdir(parents=True)
    pilot_path = pilot_dir / "candidate_edges_reference.parquet"
    pilot_path.write_text("placeholder", encoding="utf-8")
    decision = m3_14.freeze_backend_decision(accurate_evidence(), toy_config())
    dryrun = m3_14.build_dryrun_shards(
        dryrun_input_shards(),
        tmp_path / "full_by_shard",
        "sklearn_exact",
        30,
    )

    expected = m3_14.expected_outputs_payload(dryrun, tmp_path / "full_by_shard", reports_dir, decision)

    assert expected["pilot_reference_outputs"]
    assert not expected["pilot_reference_outputs"][0]["registered_as_production"]
    assert not dryrun["status_expected"].eq("production_completed").any()
    assert not dryrun["reuse_existing_pilot_allowed"].any()


def test_no_global_or_downstream_outputs_are_produced(tmp_path: Path) -> None:
    decision = m3_14.freeze_backend_decision(accurate_evidence(), toy_config())
    dryrun = m3_14.build_dryrun_shards(
        dryrun_input_shards(),
        tmp_path / "full_by_shard",
        "sklearn_exact",
        30,
    )
    expected = m3_14.expected_outputs_payload(dryrun, tmp_path / "full_by_shard", tmp_path / "reports", decision)

    downstream = expected["downstream_outputs"]
    assert not downstream["global_markov_p_produced"]
    assert not downstream["gpcca_produced"]
    assert not downstream["fate_probability_produced"]
    assert not downstream["branched_nicheflow_produced"]
    assert not downstream["m5_produced"]
    assert not downstream["regulator_analysis_produced"]
    for path in dryrun["output_parquet"].head(5):
        assert m3_14._path_has_forbidden_token(Path(path)) is None


def test_no_dataset_specific_hard_coding_is_required() -> None:
    text = "\n".join(
        [
            (PROJECT_ROOT / "src" / "nichefate" / "transition.py").read_text(encoding="utf-8"),
            SCRIPT_PATH.read_text(encoding="utf-8"),
        ]
    )

    for token in ["Moffitt", "Cadinu", "DSS", "colon", "colitis", "Day35", "Sample_type"]:
        assert token not in text


def test_validate_stage_scope_requires_dry_run_only(tmp_path: Path) -> None:
    args = Namespace(
        dry_run_only=True,
        candidate_k=30,
        default_output_root=tmp_path / "full_by_shard",
        reports_dir=tmp_path / "reports",
    )

    m3_14.validate_stage_scope(args, toy_config())
