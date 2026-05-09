import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nichefate.m3_v2_kernel import (
    exponential_gate,
    jensen_shannon_by_source,
    pairwise_l2_for_edges,
    robust_scale_fit,
    robust_scale_transform,
    row_normalize_weights,
    slice_mouse_gate,
    source_adaptive_tau,
    source_entropy_and_top1,
    validate_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "m3_v2_01_run_small_pilot.py"
SPEC = importlib.util.spec_from_file_location("m3_v2_runner", RUNNER_PATH)
m3_v2_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(m3_v2_runner)


def test_primary_state_cost_computation_on_toy_data() -> None:
    source = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    target = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    stats = robust_scale_fit([source, target])
    source_scaled = robust_scale_transform(source, stats)
    target_scaled = robust_scale_transform(target, stats)

    distances = pairwise_l2_for_edges(
        source_scaled,
        target_scaled,
        np.array([0, 1]),
        np.array([0, 1]),
        chunk_size=1,
    )

    assert distances.shape == (2,)
    assert float(distances[0]) == pytest.approx(0.0)
    assert float(distances[1]) > 0.0


def test_source_adaptive_tau_is_positive_and_source_specific() -> None:
    distances = np.array([1.0, 3.0, 10.0, 20.0], dtype=np.float32)
    source_codes = np.array([0, 0, 1, 1])

    tau = source_adaptive_tau(distances, source_codes, quantile=0.5)

    assert np.all(tau > 0)
    assert tau[0] == pytest.approx(tau[1])
    assert tau[2] == pytest.approx(tau[3])
    assert tau[0] != pytest.approx(tau[2])


def test_soft_gate_multiplication_and_neutral_barcode() -> None:
    distances = np.array([0.0, 1.0], dtype=np.float32)
    tau = np.array([1.0, 1.0], dtype=np.float32)
    state_gate = exponential_gate(distances, tau)
    barcode_gate = np.ones_like(state_gate)

    weights = state_gate * barcode_gate

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(np.exp(-1.0))


def test_slice_mouse_gate_is_soft_and_bounded() -> None:
    gate = slice_mouse_gate(
        pd.Series(["s0", "s0", "s0", "s1"]),
        pd.Series(["m0", "m0", "m0", "m1"]),
        strength=0.5,
        min_gate=0.2,
    )

    assert np.all(gate >= 0.2)
    assert np.all(gate <= 1.0)
    assert gate[0] < gate[-1]


def test_row_normalization_and_probability_validation() -> None:
    weights = np.array([1.0, 3.0, 0.0, 0.0], dtype=np.float32)
    source_codes = np.array([0, 0, 1, 1])

    probabilities = row_normalize_weights(weights, source_codes)
    qc = validate_probabilities(probabilities, source_codes)

    assert probabilities[:2].sum() == pytest.approx(1.0)
    assert probabilities[2:].sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert qc["row_sum_pass"]


def test_source_level_comparison_schema() -> None:
    v1 = np.array([0.25, 0.75, 0.5, 0.5], dtype=np.float32)
    v2 = np.array([0.75, 0.25, 0.1, 0.9], dtype=np.float32)
    source_codes = np.array([0, 0, 1, 1])

    entropy = source_entropy_and_top1(v2, source_codes)
    js = jensen_shannon_by_source(v1, v2, source_codes)

    assert {"source_code", "transition_entropy", "top1_probability"} <= set(entropy.columns)
    assert {"source_code", "v1_v2_js_divergence"} <= set(js.columns)
    assert len(entropy) == 2
    assert len(js) == 2


def test_sampling_reproducibility() -> None:
    rows = []
    for idx in range(40):
        rows.append(
            {
                "anchor_id": f"s::{idx}",
                "slice_id": f"slice_{idx % 2}",
                "mouse_id": f"mouse_{idx % 3}",
                "leiden_neigh": f"n{idx % 4}",
                "cell_type_l1": f"ct{idx % 5}",
                "cell_type_l3": f"fine{idx % 7}",
                "x": float(idx),
                "y": float(idx),
            }
        )
    meta = pd.DataFrame(rows)

    first = m3_v2_runner.stratified_source_sample(meta, cap=20, seed=1729)
    second = m3_v2_runner.stratified_source_sample(meta, cap=20, seed=1729)

    assert first["anchor_id"].tolist() == second["anchor_id"].tolist()
    assert len(first) == 20


def test_sampling_handles_non_contiguous_source_index() -> None:
    rows = []
    for idx in range(40):
        rows.append(
            {
                "anchor_id": f"s::{idx}",
                "slice_id": f"slice_{idx % 2}",
                "mouse_id": f"mouse_{idx % 3}",
                "leiden_neigh": f"n{idx % 4}",
                "cell_type_l1": f"ct{idx % 5}",
                "cell_type_l3": f"fine{idx % 7}",
                "x": float(idx),
                "y": float(idx),
            }
        )
    meta = pd.DataFrame(rows)
    meta.index = np.arange(1000, 1040)

    sample = m3_v2_runner.stratified_source_sample(meta, cap=20, seed=1729)

    assert len(sample) == 20
    assert sample["anchor_id"].is_unique


def test_output_path_separation_from_m3_v1_production(tmp_path: Path) -> None:
    m3_v2_runner.validate_output_root(tmp_path / "m3_v2_pilot")

    with pytest.raises(ValueError):
        m3_v2_runner.validate_output_root(Path("/home/zhutao/scratch/nichefate/m3/full_by_shard"))


def test_no_forbidden_downstream_outputs_are_defined() -> None:
    text = RUNNER_PATH.read_text()

    forbidden_actions = [
        "assemble_global_transition_object",
        "compute_markov_fate",
        "run_supernode_gpcca",
        "run_branched_nicheflow",
        "run_regulator",
    ]
    assert not any(f"{action}(" in text for action in forbidden_actions)
