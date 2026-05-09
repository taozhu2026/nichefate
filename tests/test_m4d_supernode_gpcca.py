import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from nichefate import m4d_standard_gpcca as m4d


def test_supernode_allocation_sums_exactly_to_target() -> None:
    counts = pd.DataFrame(
        {
            "time_label": ["D0", "D3", "D9"],
            "time_day": [0.0, 3.0, 9.0],
            "node_count": [10, 20, 70],
        }
    )

    allocation = m4d.allocate_supernodes_largest_remainder(counts, 10, (5, 20))

    assert sum(allocation.values()) == 10
    assert allocation == {"D0": 1, "D3": 2, "D9": 7}


def test_supernode_assignment_completeness_and_no_empty_supernodes() -> None:
    assignments = pd.DataFrame(
        {
            "global_node_index": [0, 1, 2, 3],
            "time_label": ["D0", "D0", "D3", "D3"],
            "time_day": [0.0, 0.0, 3.0, 3.0],
            "slice_id": ["s0", "s0", "s1", "s1"],
            "anchor_index": [0, 1, 0, 1],
            "supernode_id": [0, 1, 2, 3],
        }
    )
    counts = pd.DataFrame(
        {
            "time_label": ["D0", "D3"],
            "time_day": [0.0, 3.0],
            "node_count": [2, 2],
        }
    )

    sizes = m4d.supernode_size_table(assignments, counts, {"D0": 2, "D3": 2})

    assert sizes["supernode_id"].tolist() == [0, 1, 2, 3]
    assert sizes["supernode_size"].tolist() == [1, 1, 1, 1]
    assert not sizes["empty_supernode_warning"].any()


def test_p_super_aggregation_normalization_and_zero_row_closure(tmp_path) -> None:
    matrix = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    path = tmp_path / "P_forward.npz"
    sp.save_npz(path, matrix)
    assignments = pd.DataFrame({"global_node_index": [0, 1, 2, 3], "supernode_id": [0, 1, 2, 3]})
    sizes = pd.DataFrame(
        {
            "supernode_id": [0, 1, 2, 3],
            "time_label": ["D0", "D0", "D3", "D3"],
            "time_day": [0.0, 0.0, 3.0, 3.0],
            "supernode_size": [1, 1, 1, 1],
            "is_final_time": [False, False, True, True],
        }
    )

    p_super, edges, qc = m4d.aggregate_p_super(path, assignments, sizes, 1e-6)

    assert p_super.shape == (4, 4)
    assert np.allclose(np.asarray(p_super.sum(axis=1)).ravel(), 1.0)
    assert qc["final_time_zero_outgoing_supernode_count"] == 2
    assert qc["nonfinal_zero_outgoing_supernode_count"] == 1
    assert "not terminal-state inference" in qc["structural_closure_note"]
    assert len(edges) == p_super.nnz


def test_component_review_reports_fragmentation() -> None:
    p_super = sp.csr_matrix(
        np.array(
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    sizes = pd.DataFrame({"supernode_id": [0, 1, 2], "supernode_size": [5, 5, 1]})

    largest, summary, excluded, qc = m4d.component_review(p_super, sizes)

    assert qc["n_components"] == 2
    assert qc["run_scope"] == "largest_component"
    assert largest.tolist() == [0, 1]
    assert len(excluded) == 1
    assert len(summary) == 2


def test_pygpcca_toy_runner_uses_lazy_import(monkeypatch) -> None:
    class FakeGPCCA:
        def __init__(self, matrix, z="LM", method="krylov"):
            self.matrix = matrix
            self.method = method

        def optimize(self, k):
            self.memberships = np.array(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.1, 0.9],
                ]
            )
            self.macrostate_assignment = np.array([0, 0, 1, 1])
            self.coarse_grained_transition_matrix = np.eye(2)
            self.eigenvalues = np.array([1.0, 0.8])

    monkeypatch.setitem(sys.modules, "pygpcca", SimpleNamespace(GPCCA=FakeGPCCA))

    result = m4d.run_pygpcca_candidate(sp.eye(4, format="csr"), 2, "krylov")

    assert result["success"]
    assert result["observed_k"] == 2
    assert result["min_macrostate_size"] == 2
    assert result["metastability"] == 1.0


def test_best_k_selection_rejects_degenerate_runs() -> None:
    results = [
        {"k": 2, "success": True, "observed_k": 2, "min_macrostate_size": 1, "max_macrostate_size": 999, "min_macrostate_fraction": 0.001, "metastability": 0.99},
        {"k": 3, "success": True, "observed_k": 3, "min_macrostate_size": 20, "max_macrostate_size": 960, "min_macrostate_fraction": 0.02, "metastability": 0.90},
    ]

    selected = m4d.select_best_candidate(results, 1000)

    assert selected["selected"]["k"] == 3
    table = selected["table"].set_index("k")
    assert not bool(table.loc[2, "nondegenerate"])
    assert bool(table.loc[3, "nondegenerate"])


def test_node_projection_preserves_identity_and_wide_probability_columns(tmp_path) -> None:
    paths = SimpleNamespace(node_projection=tmp_path / "node_gpcca_macrostate_membership.parquet")
    assignments = pd.DataFrame({"global_node_index": [0, 1, 2], "supernode_id": [0, 1, 1]})
    memberships = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    macro = pd.DataFrame(
        {
            "supernode_id": [0, 1],
            "gpcca_macrostate_id": [0, 1],
            "gpcca_macrostate_probability": [0.7, 0.8],
        }
    )

    projected = m4d.project_gpcca_to_nodes(assignments, memberships, macro, paths, overwrite=True)

    assert projected["global_node_index"].tolist() == [0, 1, 2]
    assert projected["gpcca_macrostate_id"].tolist() == [0, 1, 1]
    assert {"gpcca_prob_00", "gpcca_prob_01"}.issubset(projected.columns)
    assert np.allclose(projected[["gpcca_prob_00", "gpcca_prob_01"]].sum(axis=1), 1.0)


def test_no_absorption_probability_output_terms_in_m4d_report() -> None:
    text = m4d.gpcca_report_markdown(
        {
            "generated_at_utc": "now",
            "backend": "pygpcca",
            "input_shape": [4, 4],
            "run_scope": "full_p_super",
            "selected_k": 2,
            "selected_metastability": 1.0,
            "selected_min_macrostate_size": 2,
            "no_full_node_gpcca": True,
            "no_absorption_probability": True,
            "no_fate_probability": True,
            "candidate_table": [
                {
                    "k": 2,
                    "success": True,
                    "nondegenerate": True,
                    "metastability": 1.0,
                    "min_macrostate_size": 2,
                    "error": "",
                }
            ],
        }
    )

    assert "not fate probabilities" in text
    assert "not absorption probabilities" in text
    assert "Branched" not in text
    assert "regulator" not in text.lower()
