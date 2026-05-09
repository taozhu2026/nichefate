import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "k_gpcca_02_construct_and_run_pilot.py"
SPEC = importlib.util.spec_from_file_location("k_gpcca02", SCRIPT_PATH)
k_gpcca02 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = k_gpcca02
SPEC.loader.exec_module(k_gpcca02)


def minimal_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "output_root": str(tmp_path / "k_gpcca_pilot"),
            "reports_dir": str(tmp_path / "k_gpcca_pilot" / "reports"),
            "design_root": str(tmp_path / "design"),
        },
        "pilot": {
            "preferred_time_points": ["D9", "D21", "D35"],
            "preferred_time_pairs": ["D9_to_D21", "D21_to_D35"],
            "target_max_nodes": 12,
        },
        "protected_roots": ["/home/zhutao/scratch/nichefate/m3"],
        "forbidden_downstream_roots": ["/home/zhutao/scratch/nichefate/k_gpcca"],
    }


def toy_selected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "local_index": np.arange(6),
            "global_node_index": np.arange(100, 106),
            "anchor_id": [f"s{i // 2}::{i}" for i in range(6)],
            "slice_id": [f"s{i // 2}" for i in range(6)],
            "anchor_index": np.arange(6),
            "time": ["D9", "D9", "D21", "D21", "D35", "D35"],
            "time_day": [9, 9, 21, 21, 35, 35],
            "mouse_id": ["m0", "m1", "m0", "m1", "m0", "m1"],
            "cell_type_l3": ["a", "b", "a", "b", "a", "b"],
        }
    )


def toy_candidate(source: str = "M3-v1") -> k_gpcca02.Candidate:
    return k_gpcca02.Candidate(
        grid_id="toy",
        route="full_resolution_subset",
        cross_time_source=source,
        alpha=0.6,
        beta=0.35,
        gamma=0.05,
        delta=0.0,
        within_time_k=1,
        similarity_metric="cosine",
        priority="default",
    )


def test_config_parsing_and_output_dirs_are_safe(tmp_path: Path) -> None:
    config = k_gpcca02.k01.load_config(PROJECT_ROOT / "configs" / "k_gpcca_pilot.yaml")
    assert config["project"]["stage"] == "K_gpcca-01"

    safe_paths = k_gpcca02.output_dirs(minimal_config(tmp_path))
    assert safe_paths["kernels"].parent == (tmp_path / "k_gpcca_pilot").resolve()

    protected = minimal_config(tmp_path)
    protected["paths"]["output_root"] = "/home/zhutao/scratch/nichefate/m3/reports"
    with pytest.raises(ValueError, match="protected production root"):
        k_gpcca02.output_dirs(protected)

    ssd = minimal_config(tmp_path)
    ssd["paths"]["output_root"] = "/ssd/nichefate/k_gpcca_pilot"
    with pytest.raises(ValueError, match="Refusing /ssd path"):
        k_gpcca02.output_dirs(ssd)


def test_candidate_grid_row_selection_uses_existing_ids() -> None:
    config = k_gpcca02.k01.load_config(PROJECT_ROOT / "configs" / "k_gpcca_pilot.yaml")

    candidate = k_gpcca02.select_candidate(config, "pilot_v1_balanced")
    assert candidate.grid_id == "pilot_v1_balanced"
    assert candidate.cross_time_source == "M3-v1"

    with pytest.raises(ValueError):
        k_gpcca02.select_candidate(config, "future_barcode_placeholder")
    with pytest.raises(ValueError):
        k_gpcca02.select_candidate(config, "pilot_mixed_cross_time_review")


def test_toy_within_time_graph_construction_has_same_time_edges() -> None:
    selected = toy_selected()
    features = np.array(
        [
            [1, 0],
            [0.9, 0.1],
            [0, 1],
            [0.1, 0.9],
            [1, 1],
            [0.8, 1],
        ],
        dtype=np.float32,
    )
    graph = k_gpcca02.build_within_time_graph(selected, features, k=1, metric="cosine")

    assert graph.shape == (6, 6)
    assert graph.nnz == 6
    rows, cols = graph.nonzero()
    for row, col in zip(rows, cols, strict=True):
        assert selected.loc[row, "time"] == selected.loc[col, "time"]


def test_toy_cross_time_edge_mapping_and_row_normalization(tmp_path: Path, monkeypatch) -> None:
    selected = toy_selected()
    edge_root = tmp_path / "edges"
    shard = edge_root / "D9_to_D21" / "slice"
    shard.mkdir(parents=True)
    edges = pd.DataFrame(
        {
            "source_anchor_id": ["s0::0", "s0::1", "outside"],
            "target_anchor_id": ["s1::2", "s1::3", "s1::2"],
            "source_time": ["D9", "D9", "D9"],
            "target_time": ["D21", "D21", "D21"],
            "row_normalized_transition_prob": [0.2, 0.8, 1.0],
        }
    )
    edges.to_parquet(shard / "edges.parquet", index=False)
    config = {
        "cross_time": {
            "m3_v1_edge_root": str(edge_root),
            "m3_v2_edge_root": str(edge_root),
            "m3_v1_probability_column": "row_normalized_transition_prob",
            "m3_v2_probability_column": "row_normalized_transition_prob",
            "batch_rows": 100,
        }
    }

    graph = k_gpcca02.build_cross_time_graph(config, selected, toy_candidate(), ["D9_to_D21"])

    assert graph.shape == (6, 6)
    assert graph.nnz == 2
    assert np.allclose(np.asarray(graph.sum(axis=1)).ravel()[:2], [1.0, 1.0])


def test_self_loop_addition_and_row_normalization_qc() -> None:
    selected = toy_selected()
    within = sparse.csr_matrix(
        (
            np.ones(6, dtype=np.float32),
            ([0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]),
        ),
        shape=(6, 6),
    )
    cross = sparse.csr_matrix(
        (
            np.ones(2, dtype=np.float32),
            ([0, 1], [2, 3]),
        ),
        shape=(6, 6),
    )
    within = k_gpcca02.row_normalize(within)
    cross = k_gpcca02.row_normalize(cross)
    kernel, masses = k_gpcca02.combine_components(within, cross, toy_candidate())
    qc = k_gpcca02.kernel_qc(kernel, within, cross, selected, toy_candidate(), masses)

    assert qc["row_sum_max_error"] <= 1e-6
    assert qc["invalid_entries"] == 0
    assert qc["negative_entries"] == 0
    assert qc["zero_outgoing_rows"] == 0
    assert qc["self_loop_count"] == 6
    assert qc["within_time_mass_fraction"] > qc["cross_time_mass_fraction"]


def test_sparse_matrix_invalid_checks_detect_bad_values() -> None:
    selected = toy_selected()
    bad = sparse.csr_matrix(([np.nan], ([0], [0])), shape=(6, 6))
    good = sparse.identity(6, format="csr", dtype=np.float32)
    qc = k_gpcca02.kernel_qc(bad, good, good, selected, toy_candidate(), {})

    assert qc["invalid_entries"] == 1
    assert not qc["kernel_qc_pass"]


def test_pygpcca_command_is_blocked_when_kernel_qc_fails(monkeypatch) -> None:
    invoked = {"called": False}

    def fake_run(*args, **kwargs):
        invoked["called"] = True

    monkeypatch.setattr(k_gpcca02, "run_pygpcca_subprocess", fake_run)

    assert k_gpcca02.kernel_qc_allows_gpcca({"kernel_qc_pass": False}) is False
    assert invoked["called"] is False


def test_no_custom_gpcca_fallback_is_exposed() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "custom GPCCA-like" in source
    assert "fallback" in source
    assert "scipy_pcca_like_diagnostic_fallback" not in source
    assert "CellRank" in source


def test_output_schema_helpers(tmp_path: Path) -> None:
    paths = {
        "root": tmp_path,
        "kernels": tmp_path / "kernels",
        "gpcca": tmp_path / "gpcca",
        "reports": tmp_path / "reports",
        "figures": tmp_path / "reports" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(paths["kernels"] / "K_gpcca_smoke_toy.npz", sparse.identity(2, format="csr"))
    inventory = k_gpcca02.scan_kernel_inventory(paths)

    assert {"artifact", "path", "bytes", "node_table_exists", "node_table_path"} <= set(inventory.columns)
    assert len(inventory) == 1
