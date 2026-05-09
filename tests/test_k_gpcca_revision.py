import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "k_gpcca_04_kernel_revision_pilot.py"
SPEC = importlib.util.spec_from_file_location("k_gpcca04", SCRIPT_PATH)
k_gpcca04 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = k_gpcca04
SPEC.loader.exec_module(k_gpcca04)


def minimal_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "output_root": str(tmp_path / "k_gpcca_revision"),
            "reports_dir": str(tmp_path / "k_gpcca_revision" / "reports"),
            "m2_root": str(tmp_path / "m2"),
            "k03_benchmark_root": str(tmp_path / "k03"),
        },
        "feature_processing": {
            "mode": "group_balanced_zscore_svd50",
            "svd_components": 2,
            "metadata_denylist": ["time", "slice", "mouse", "anchor_index", "x", "y"],
        },
        "revision_candidates": [
            {
                "grid_id": "rev_v1_cross45",
                "route": "full_resolution_subset",
                "cross_time_source": "M3-v1",
                "alpha": 0.5,
                "beta": 0.45,
                "gamma": 0.05,
                "delta": 0.0,
                "within_time_k": 30,
                "similarity_metric": "cosine",
                "priority": "bounded_revision",
            }
        ],
        "acceptance": {
            "k": 10,
            "largest_macrostate_fraction_lt": 0.40,
            "smallest_macrostate_fraction_gte": 0.005,
            "mean_max_membership_gt": 0.60,
        },
        "protected_roots": ["/home/zhutao/scratch/nichefate/m4c"],
        "forbidden_downstream_roots": ["/home/zhutao/scratch/nichefate/k_gpcca"],
    }


def test_config_and_candidate_loading() -> None:
    config = k_gpcca04.load_config(PROJECT_ROOT / "configs" / "k_gpcca_revision.yaml")
    candidates = k_gpcca04.load_candidates(config)

    assert config["project"]["stage"] == "K_gpcca-04"
    assert [candidate.grid_id for candidate in candidates] == [
        "rev_v1_cross45",
        "rev_v1_cross50",
        "rev_v1_k50_cross40",
        "rev_v2_cross45",
    ]
    assert candidates[0].as_k02().cross_time_source == "M3-v1"


def test_output_dir_safety_rejects_protected_and_ssd(tmp_path: Path) -> None:
    safe = k_gpcca04.output_dirs(minimal_config(tmp_path))
    assert safe["root"] == (tmp_path / "k_gpcca_revision").resolve()

    protected = minimal_config(tmp_path)
    protected["paths"]["output_root"] = "/home/zhutao/scratch/nichefate/m4c/reports"
    protected["paths"]["reports_dir"] = "/home/zhutao/scratch/nichefate/m4c/reports/k04"
    with pytest.raises(ValueError, match="protected root"):
        k_gpcca04.output_dirs(protected)

    ssd = minimal_config(tmp_path)
    ssd["paths"]["output_root"] = "/ssd/nichefate/k_gpcca_revision"
    with pytest.raises(ValueError, match="Refusing /ssd path"):
        k_gpcca04.output_dirs(ssd)


def test_feature_classification_uses_groups_and_metadata_denylist() -> None:
    denylist = ["time", "slice", "mouse", "anchor_index", "x", "y"]

    assert k_gpcca04.classify_feature("radius_x2__ct_l3__b_cell", denylist)[0] == "composition"
    assert k_gpcca04.classify_feature("radius_x2__emb_mean_pc001", denylist)[0] == "molecular_state"
    assert k_gpcca04.classify_feature("radius_x4__emb_var_pc050", denylist)[0] == "molecular_state"
    assert k_gpcca04.classify_feature("radius_x8__pseudo_local_density", denylist)[0] == "spatial_topology"
    group, excluded, _ = k_gpcca04.classify_feature("time_day", denylist)
    assert group == "excluded_metadata_like"
    assert excluded


def test_group_balanced_embedding_is_finite_and_bounded() -> None:
    features = np.array(
        [
            [1.0, 2.0, 10.0, 100.0],
            [2.0, 3.0, 11.0, 110.0],
            [3.0, 4.0, 12.0, 120.0],
            [4.0, 5.0, 13.0, 130.0],
        ],
        dtype=np.float32,
    )
    groups = {
        "composition": np.array([0, 1]),
        "molecular_state": np.array([2]),
        "spatial_topology": np.array([3]),
    }

    embedding, report = k_gpcca04.group_balanced_embedding(features, groups, n_components=2)

    assert embedding.shape == (4, 2)
    assert np.isfinite(embedding).all()
    assert set(report["feature_group"]) == {"composition", "molecular_state", "spatial_topology"}
    assert report["finite_values"].all()


def test_kernel_qc_blocks_gpcca_when_failed() -> None:
    assert k_gpcca04.kernel_qc_allows_gpcca({"kernel_qc_pass": False}) is False
    assert k_gpcca04.kernel_qc_allows_gpcca({"kernel_qc_pass": "False"}) is False
    assert k_gpcca04.kernel_qc_allows_gpcca({"kernel_qc_pass": True}) is True


def test_comparison_selection_requires_tiny_macrostate_fix() -> None:
    baseline = {
        "largest_macrostate_fraction": 0.36981,
        "smallest_macrostate_fraction": 0.00148,
        "max_membership_mean": 0.60825,
    }
    candidate = k_gpcca04.RevisionCandidate(
        grid_id="rev",
        route="full_resolution_subset",
        cross_time_source="M3-v1",
        alpha=0.5,
        beta=0.45,
        gamma=0.05,
        delta=0.0,
        within_time_k=30,
        similarity_metric="cosine",
        priority="bounded_revision",
    )
    kernel_qc = pd.DataFrame([{"candidate_id": "rev", "kernel_qc_pass": True}])
    gpcca = pd.DataFrame(
        [
            {
                "candidate_id": "rev",
                "success": True,
                "largest_macrostate_fraction": 0.30,
                "smallest_macrostate_fraction": 0.001,
                "max_membership_mean": 0.70,
            }
        ]
    )
    flags = pd.DataFrame([{"candidate_id": "rev", "status": "WARN"}])
    comparison = k_gpcca04.build_comparison(
        baseline,
        baseline_warn_count=6,
        candidates=[candidate],
        kernel_qc=kernel_qc,
        gpcca=gpcca,
        flags=flags,
        acceptance={
            "largest_macrostate_fraction_lt": 0.40,
            "smallest_macrostate_fraction_gte": 0.005,
            "mean_max_membership_gt": 0.60,
        },
    )

    assert not bool(comparison.iloc[0]["selection_eligible"])
    assert k_gpcca04.select_revision(comparison)["decision_category"] == "need_feature_processing_redesign"


def test_no_custom_terminal_or_fate_fallback_is_exposed() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8").lower()

    assert "terminal_states_computed\": false" in source or "terminal_states_computed" in source
    assert "fate_probabilities_computed\": false" in source or "fate_probabilities_computed" in source
    assert "custom gpcca-like fallback" in source
    assert "custom_absorbing" not in source
    assert "manual_fate" not in source


def test_p_fate_overlap_schema() -> None:
    candidate = k_gpcca04.RevisionCandidate(
        grid_id="rev",
        route="full_resolution_subset",
        cross_time_source="M3-v1",
        alpha=0.5,
        beta=0.45,
        gamma=0.05,
        delta=0.0,
        within_time_k=30,
        similarity_metric="cosine",
        priority="bounded_revision",
    )
    table = pd.DataFrame(
        {
            "candidate_id": ["rev", "rev"],
            "k": [10, 10],
            "macrostate": [0, 0],
            "node_count": [8, 2],
            "fraction_within_macrostate": [0.8, 0.2],
        }
    )

    result = k_gpcca04.p_fate_overlap({"p_fate_v1": table}, [candidate], 10)

    assert {"candidate_id", "comparison", "mean_dominant_fraction", "max_dominant_fraction"} <= set(result.columns)
    assert result[result["comparison"] == "p_fate_v1"].iloc[0]["mean_dominant_fraction"] == 0.8
