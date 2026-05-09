import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "m3_v2_03_confirmatory_constrained_pilot.py"
SPEC = importlib.util.spec_from_file_location("m3_v2_confirmatory", RUNNER_PATH)
m3_v2_confirmatory = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m3_v2_confirmatory
SPEC.loader.exec_module(m3_v2_confirmatory)


def toy_edges() -> pd.DataFrame:
    rows = []
    for source_idx in range(2):
        for target_idx, distance in enumerate([0.0, 1.0, 2.0]):
            rows.append(
                {
                    "source_anchor_id": f"s{source_idx}",
                    "target_anchor_id": f"t{source_idx}_{target_idx}",
                    "source_slice_id": "src",
                    "target_slice_id": f"slice{target_idx}",
                    "source_mouse_id": "m0",
                    "target_mouse_id": f"m{target_idx}",
                    "row_normalized_transition_prob": [0.65, 0.25, 0.10][target_idx],
                    "v2_d_state": distance,
                    "v2_tau_state": 1.0,
                    "v2_g_composition": [1.0, 0.6, 0.3][target_idx],
                    "v2_g_spatial_topology": [1.0, 0.6, 0.3][target_idx],
                    "v2_g_slice_mouse": 1.0,
                }
            )
    return pd.DataFrame(rows)


def source_codes(edges: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(edges["source_anchor_id"], sort=False)
    return codes.astype(np.int32)


def test_confirmatory_uses_only_selected_best_variant() -> None:
    specs = m3_v2_confirmatory.pilot_specs(skip_optional_c=True)

    assert m3_v2_confirmatory.VARIANT_NAME == "v1prior_1.0_tau_0.5_top10"
    assert [spec.pilot_id for spec in specs] == ["A_D9_D21_repeat", "B_D3_D9"]


def test_repeat_seed_changes_source_subset() -> None:
    rows = []
    for idx in range(120):
        rows.append(
            {
                "anchor_id": f"s::{idx}",
                "slice_id": f"slice_{idx % 3}",
                "mouse_id": f"mouse_{idx % 2}",
                "time_label": "D9",
                "leiden_neigh": f"n{idx % 4}",
                "cell_type_l1": f"ct{idx % 5}",
                "cell_type_l3": f"fine{idx % 7}",
                "x": float(idx),
                "y": float(idx),
            }
        )
    meta = pd.DataFrame(rows)

    first = m3_v2_confirmatory.stratified_source_sample(meta, cap=40, seed=1729)
    second = m3_v2_confirmatory.stratified_source_sample(meta, cap=40, seed=271829)

    assert first["anchor_id"].tolist() != second["anchor_id"].tolist()
    assert len(first) == len(second) == 40


def test_time_pair_shard_pattern_supports_required_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for pair in ["D9_to_D21", "D3_to_D9"]:
        shard_dir = tmp_path / pair / "slice_a"
        shard_dir.mkdir(parents=True)
        source_time, target_time = pair.split("_to_")
        (shard_dir / f"candidate_edges_{source_time}_to_{target_time}__slice_a.parquet").touch()
    monkeypatch.setattr(m3_v2_confirmatory, "M3_FULL_BY_SHARD", tmp_path)

    assert "slice_a" in m3_v2_confirmatory.discover_edge_shards("D9", "D21")
    assert "slice_a" in m3_v2_confirmatory.discover_edge_shards("D3", "D9")


def test_best_variant_top10_row_normalization() -> None:
    edges = toy_edges()
    codes = source_codes(edges)
    probs, qc = m3_v2_confirmatory.apply_best_variant(edges, codes)

    assert qc["row_sum_pass"]
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0)
    assert np.count_nonzero(probs[:3]) == 3


def test_output_path_separation_from_previous_pilots_and_production(tmp_path: Path) -> None:
    m3_v2_confirmatory.validate_output_root(tmp_path / "m3_v2_pilot_confirmatory")

    with pytest.raises(ValueError):
        m3_v2_confirmatory.validate_output_root(Path("/home/zhutao/scratch/nichefate/m3_v2_pilot"))
    with pytest.raises(ValueError):
        m3_v2_confirmatory.validate_output_root(Path("/home/zhutao/scratch/nichefate/m3_v2_pilot_tuning"))
    with pytest.raises(ValueError):
        m3_v2_confirmatory.validate_output_root(Path("/home/zhutao/scratch/nichefate/m4c/fate_probabilities"))
