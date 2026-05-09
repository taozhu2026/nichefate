import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "k_gpcca_03_biological_benchmark.py"
SPEC = importlib.util.spec_from_file_location("k_gpcca03", SCRIPT_PATH)
k_gpcca03 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = k_gpcca03
SPEC.loader.exec_module(k_gpcca03)


def test_output_root_safety_rejects_protected_and_ssd(tmp_path: Path) -> None:
    safe = k_gpcca03.output_paths(tmp_path / "k_gpcca_pilot_benchmark")
    assert safe["root"] == (tmp_path / "k_gpcca_pilot_benchmark").resolve()

    with pytest.raises(ValueError, match="protected root"):
        k_gpcca03.output_paths(Path("/home/zhutao/scratch/nichefate/m4c_v2/reports"))

    with pytest.raises(ValueError, match="Refusing /ssd path"):
        k_gpcca03.output_paths(Path("/ssd/nichefate/k_gpcca_pilot_benchmark"))


def test_bool_parsing_is_robust_for_checkpoint_csv_values() -> None:
    assert k_gpcca03.as_bool(True)
    assert k_gpcca03.as_bool("True")
    assert not k_gpcca03.as_bool(False)
    assert not k_gpcca03.as_bool("False")
    assert not k_gpcca03.as_bool("")


def test_checkpoint_upsert_replaces_same_candidate_and_k(tmp_path: Path) -> None:
    paths = k_gpcca03.output_paths(tmp_path / "benchmark")
    k_gpcca03.ensure_dirs(paths)
    base = {
        "candidate_id": "pilot_v1_balanced",
        "k": 4,
        "success": False,
        "error": "first",
        "source": "standard pyGPCCA",
        "runtime_seconds": 1.0,
        "macrostate_count": 0,
        "macrostate_size_min": np.nan,
        "macrostate_size_max": np.nan,
        "largest_macrostate_fraction": np.nan,
        "smallest_macrostate_fraction": np.nan,
        "membership_entropy_mean": np.nan,
        "membership_entropy_median": np.nan,
        "max_membership_mean": np.nan,
        "max_membership_median": np.nan,
        "macro_path": "",
        "memberships_path": "",
        "coarse_path": "",
    }
    k_gpcca03.upsert_sensitivity(paths, base)
    k_gpcca03.upsert_sensitivity(paths, {**base, "success": True, "error": ""})

    saved = k_gpcca03.load_sensitivity(paths)
    assert len(saved) == 1
    assert bool(saved.iloc[0]["success"]) is True
    assert pd.isna(saved.iloc[0]["error"]) or saved.iloc[0]["error"] == ""


def test_artifact_flags_apply_required_macrostate_imbalance_thresholds() -> None:
    sensitivity = pd.DataFrame(
        [
            {
                "candidate_id": "pilot_v1_balanced",
                "k": 8,
                "success": True,
                "error": "",
                "largest_macrostate_fraction": 0.58479,
                "smallest_macrostate_fraction": 0.00146,
            }
        ]
    )
    annotation_summary = pd.DataFrame(
        [
            {
                "candidate_id": "pilot_v1_balanced",
                "k": 8,
                "annotation_group": "time",
                "dominant_fraction": 0.7,
            }
        ]
    )

    flags = k_gpcca03.artifact_flags(sensitivity, annotation_summary)
    flagged = dict(zip(flags["artifact"], flags["status"], strict=False))
    assert flagged["major_macrostate_imbalance"] == "WARN"
    assert flagged["tiny_macrostate"] == "WARN"


def test_select_preferred_k_uses_primary_candidate_and_warnings() -> None:
    sensitivity = pd.DataFrame(
        [
            {
                "candidate_id": "pilot_v1_balanced",
                "k": 4,
                "success": True,
                "largest_macrostate_fraction": 0.6,
                "smallest_macrostate_fraction": 0.004,
                "max_membership_mean": 0.8,
            },
            {
                "candidate_id": "pilot_v1_balanced",
                "k": 6,
                "success": True,
                "largest_macrostate_fraction": 0.4,
                "smallest_macrostate_fraction": 0.01,
                "max_membership_mean": 0.75,
            },
        ]
    )
    flags = pd.DataFrame(
        [
            {"candidate_id": "pilot_v1_balanced", "k": 4, "status": "WARN"},
            {"candidate_id": "pilot_v2_balanced", "k": 6, "status": "WARN"},
        ]
    )

    selected = k_gpcca03.select_preferred_k(sensitivity, flags)
    assert selected["selected_k"] == 6
    assert selected["decision_category"] == "select_k_for_terminal_review"


def test_maybe_run_v2_skips_when_primary_has_no_success(tmp_path: Path, monkeypatch) -> None:
    paths = k_gpcca03.output_paths(tmp_path / "benchmark")
    k_gpcca03.ensure_dirs(paths)
    k_gpcca03.upsert_sensitivity(
        paths,
        {
            "candidate_id": "pilot_v1_balanced",
            "k": 4,
            "success": False,
            "error": "failed",
            "source": "standard pyGPCCA",
            "runtime_seconds": 1.0,
            "macrostate_count": 0,
            "macrostate_size_min": np.nan,
            "macrostate_size_max": np.nan,
            "largest_macrostate_fraction": np.nan,
            "smallest_macrostate_fraction": np.nan,
            "membership_entropy_mean": np.nan,
            "membership_entropy_median": np.nan,
            "max_membership_mean": np.nan,
            "max_membership_median": np.nan,
            "macro_path": "",
            "memberships_path": "",
            "coarse_path": "",
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("v2 should not run without a successful v1 sensitivity result")

    monkeypatch.setattr(k_gpcca03, "run_standard_pygpcca_for_k", fail_if_called)
    comparison = k_gpcca03.maybe_run_v2(paths, skip_v2=False, timeout_seconds=1)
    assert comparison.empty


def test_p_fate_comparison_reports_consistency_not_replacement() -> None:
    table = pd.DataFrame(
        {
            "candidate_id": ["pilot_v1_balanced", "pilot_v1_balanced"],
            "k": [8, 8],
            "macrostate": [0, 0],
            "label": ["endpoint_a", "endpoint_b"],
            "node_count": [90, 10],
            "fraction_within_macrostate": [0.9, 0.1],
        }
    )
    result = k_gpcca03.p_fate_comparison({"p_fate_v1": table}, selected_k=8)
    row = result[result["comparison"] == "p_fate_v1"].iloc[0]
    assert row["status"] == "PASS"
    assert "P_fate remains valid frozen baseline" in row["interpretation"]


def test_validate_inputs_schema_with_toy_files(tmp_path: Path, monkeypatch) -> None:
    files = {}
    for name in ["kernel", "node_table", "memberships", "macrostates", "coarse", "summary", "annotation"]:
        path = tmp_path / f"{name}.txt"
        path.write_text("x\n", encoding="utf-8")
        files[name] = path

    monkeypatch.setattr(
        k_gpcca03,
        "input_paths",
        lambda: {
            "v1_kernel": files["kernel"],
            "v1_node_table": files["node_table"],
            "v2_kernel": files["kernel"],
            "v2_node_table": files["node_table"],
            "v1_k8_memberships": files["memberships"],
            "v1_k8_macrostates": files["macrostates"],
            "v1_k8_coarse": files["coarse"],
            "k02_summary": files["summary"],
            "k02_annotation": files["annotation"],
        },
    )
    monkeypatch.setattr(k_gpcca03, "K_PILOT_ROOT", tmp_path)
    result = k_gpcca03.validate_inputs({})
    assert {"input_name", "path", "exists", "bytes", "status"} <= set(result.columns)
    assert set(result["status"]) == {"PASS"}
