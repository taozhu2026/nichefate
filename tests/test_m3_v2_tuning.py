import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "m3_v2_02_kernel_diagnostic_tuning.py"
SPEC = importlib.util.spec_from_file_location("m3_v2_tuning", RUNNER_PATH)
m3_v2_tuning = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m3_v2_tuning
SPEC.loader.exec_module(m3_v2_tuning)


def toy_edges() -> pd.DataFrame:
    rows = []
    for source_idx in range(2):
        for target_idx, distance in enumerate([0.0, 1.0, 2.0]):
            rows.append(
                {
                    "source_anchor_id": f"s{source_idx}",
                    "target_anchor_id": f"t{source_idx}_{target_idx}",
                    "target_slice_id": f"slice{target_idx % 2}",
                    "target_mouse_id": f"mouse{target_idx % 2}",
                    "row_normalized_transition_prob": [0.7, 0.2, 0.1][target_idx],
                    "source_leiden_neigh": "A",
                    "target_leiden_neigh": "A" if target_idx == 0 else "B",
                    "source_cell_type_l3": "ct",
                    "target_cell_type_l3": "ct" if target_idx < 2 else "other",
                    "source_refined_endpoint_id": "endpoint",
                    "target_refined_endpoint_id": "endpoint" if target_idx == 0 else "other",
                    "v2_d_state": distance,
                    "v2_tau_state": 1.0,
                    "v2_g_composition": [1.0, 0.5, 0.2][target_idx],
                    "v2_g_spatial_topology": [1.0, 0.5, 0.2][target_idx],
                    "v2_g_slice_mouse": 1.0,
                    "v2_unnormalized_weight": np.exp(-distance),
                    "v2_row_normalized_transition_prob": [0.7, 0.2, 0.1][target_idx],
                }
            )
    return pd.DataFrame(rows)


def source_codes(edges: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(edges["source_anchor_id"], sort=False)
    return codes.astype(np.int32)


def test_v1_prior_reweighting_preserves_row_normalization() -> None:
    edges = toy_edges()
    probs, qc = m3_v2_tuning.compute_variant_probabilities(
        edges,
        source_codes(edges),
        m3_v2_tuning.VariantSpec("toy", v1_lambda=1.0),
    )

    assert qc["row_sum_pass"]
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)


def test_tau_scaling_sharpens_weights() -> None:
    edges = toy_edges()
    codes = source_codes(edges)
    tau1, _ = m3_v2_tuning.compute_variant_probabilities(
        edges,
        codes,
        m3_v2_tuning.VariantSpec("tau1", tau_scale=1.0),
    )
    tau_half, _ = m3_v2_tuning.compute_variant_probabilities(
        edges,
        codes,
        m3_v2_tuning.VariantSpec("tau_half", tau_scale=0.5),
    )

    assert tau_half[:3].max() > tau1[:3].max()


def test_gate_powers_sharpen_weaker_gates() -> None:
    edges = toy_edges()
    codes = source_codes(edges)
    gate1, _ = m3_v2_tuning.compute_variant_probabilities(
        edges,
        codes,
        m3_v2_tuning.VariantSpec("gate1", comp_power=1.0, topo_power=1.0),
    )
    gate2, _ = m3_v2_tuning.compute_variant_probabilities(
        edges,
        codes,
        m3_v2_tuning.VariantSpec("gate2", comp_power=2.0, topo_power=2.0),
    )

    assert gate2[2] < gate1[2]


def test_top_k_truncation_preserves_row_sums() -> None:
    edges = toy_edges()
    codes = source_codes(edges)
    probs, qc = m3_v2_tuning.compute_variant_probabilities(
        edges,
        codes,
        m3_v2_tuning.VariantSpec("top1", top_k=1),
    )

    assert qc["row_sum_pass"]
    assert np.count_nonzero(probs[:3]) == 1
    assert np.count_nonzero(probs[3:]) == 1


def test_variant_metric_schema_is_stable() -> None:
    edges = toy_edges()
    codes = source_codes(edges)
    probs, qc = m3_v2_tuning.compute_variant_probabilities(
        edges,
        codes,
        m3_v2_tuning.VariantSpec("toy"),
    )
    row = m3_v2_tuning.variant_metrics(edges, probs, codes, m3_v2_tuning.VariantSpec("toy"), qc)

    expected = {
        "variant",
        "row_sum_pass",
        "leiden_consistency",
        "fine_cell_cluster_consistency",
        "refined_endpoint_plausibility",
        "transition_entropy_mean",
        "top1_probability_mean",
        "target_neighborhood_diversity",
        "slice_mouse_collapse",
        "mean_js_divergence_from_v1",
    }
    assert expected <= set(row)


def test_output_path_separation_rejects_protected_roots(tmp_path: Path) -> None:
    m3_v2_tuning.validate_output_root(tmp_path / "m3_v2_pilot_tuning")

    with pytest.raises(ValueError):
        m3_v2_tuning.validate_output_root(Path("/home/zhutao/scratch/nichefate/m3_v2_pilot"))
    with pytest.raises(ValueError):
        m3_v2_tuning.validate_output_root(Path("/home/zhutao/scratch/nichefate/m3/full_by_shard"))
