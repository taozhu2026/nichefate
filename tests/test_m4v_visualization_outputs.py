import numpy as np
import pandas as pd
import pytest

from nichefate import m4v_markov_visualization as m4v


def test_visualization_table_joins_by_global_node_index(tmp_path) -> None:
    paths = m4v.M4VPaths(
        m4c_summary=tmp_path / "m4c.parquet",
        m4d_projection=tmp_path / "m4d.parquet",
        coordinate_cache=tmp_path / "coords.parquet",
        node_table=tmp_path / "nodes.parquet",
        m2_root=tmp_path / "m2",
        visualization_table=tmp_path / "viz.parquet",
        reports_dir=tmp_path / "reports",
        figures_root=tmp_path / "figures",
        m4c_figures=tmp_path / "figures" / "m4c",
        m4d_figures=tmp_path / "figures" / "m4d",
        comparison_figures=tmp_path / "figures" / "comparison",
        state_space_figures=tmp_path / "figures" / "state",
        report_md=tmp_path / "report.md",
        summary_json=tmp_path / "summary.json",
        figure_inventory=tmp_path / "inventory.csv",
    )
    pd.DataFrame(
        {
            "global_node_index": [0, 1],
            "time": ["D0", "D3"],
            "time_day": [0.0, 3.0],
            "slice_id": ["s0", "s1"],
            "mouse_id": ["m0", "m1"],
            "cell_type_l1": ["a", "b"],
            "cell_type_l2": ["a", "b"],
            "cell_type_l3": ["a", "b"],
        }
    ).to_parquet(paths.node_table)
    pd.DataFrame(
        {
            "global_node_index": [0, 1],
            "x_raw": [1.0, 2.0],
            "y_raw": [3.0, 4.0],
            "x_scaled_by_slice": [0.1, 0.2],
            "y_scaled_by_slice": [0.3, 0.4],
        }
    ).to_parquet(paths.coordinate_cache)
    pd.DataFrame(
        {
            "global_node_index": [0, 1],
            "dominant_fate": [1, 2],
            "dominant_fate_label": ["f1", "f2"],
            "dominant_fate_probability": [0.6, 0.7],
            "plasticity_entropy": [0.2, 0.3],
            "normalized_plasticity_entropy": [0.4, 0.5],
            "fate_margin_top1_minus_top2": [0.1, 0.2],
        }
    ).to_parquet(paths.m4c_summary)
    pd.DataFrame(
        {
            "global_node_index": [0, 1],
            "gpcca_macrostate_id": [0, 1],
            "gpcca_macrostate_probability": [0.8, 0.9],
            "gpcca_membership_entropy": [0.1, 0.2],
            "gpcca_prob_00": [0.8, 0.1],
            "gpcca_prob_01": [0.2, 0.9],
        }
    ).to_parquet(paths.m4d_projection)

    table = m4v.build_visualization_table(paths, overwrite=True)

    assert table["global_node_index"].tolist() == [0, 1]
    assert {"dominant_fate", "gpcca_macrostate_id", "x_scaled_by_slice"}.issubset(table.columns)
    assert paths.visualization_table.is_file()


def test_representative_slice_selection_is_deterministic() -> None:
    table = pd.DataFrame(
        {
            "time_day": [0, 0, 0, 0, 3, 3],
            "time_label": ["D0", "D0", "D0", "D0", "D3", "D3"],
            "slice_id": ["a", "a", "b", "c", "x", "y"],
            "global_node_index": list(range(6)),
        }
    )

    selected = m4v.representative_slices(table)

    assert selected["time_label"].tolist() == ["D0", "D3"]
    assert selected["slice_id"].tolist() == ["b", "x"]


def test_cross_time_physical_arrows_are_rejected(tmp_path, monkeypatch) -> None:
    config = {
        "m4_visualization": {
            "enabled": True,
            "cross_time_physical_arrows_allowed": True,
            "m4c_baseline_node_summary": str(tmp_path / "m4c.parquet"),
            "m4d_coordinate_cache": str(tmp_path / "coords.parquet"),
        },
        "paths": {
            "reports_dir": str(tmp_path / "reports"),
            "m4a_node_table": str(tmp_path / "nodes.parquet"),
            "m2_by_slice_root": str(tmp_path / "m2"),
            "visualization_dir": str(tmp_path / "viz"),
        },
        "standard_gpcca": {"output_gpcca_root": str(tmp_path / "gpcca")},
    }
    monkeypatch.setattr(m4v, "require_parquet_engine", lambda: {"pyarrow": {"available": True}})

    with pytest.raises(ValueError, match="Cross-time physical arrows"):
        m4v.run_m4v_01(config)


def test_visualization_report_uses_strict_terminology() -> None:
    text = m4v.visualization_report(
        {
            "generated_at_utc": "now",
            "status": "completed",
            "m4d_gate_passed": True,
            "m4d_gate_message": "ok",
            "figures_generated": 3,
            "warnings": [],
        }
    )

    assert "M4C outputs are Markov baseline fate probabilities" in text
    assert "M4D outputs are GPCCA macrostate memberships" in text
    assert "no absorption probability was computed" in text


def test_state_space_arrows_are_transition_tendency_only() -> None:
    summary = {
        "generated_at_utc": "now",
        "status": "completed",
        "m4d_gate_passed": True,
        "m4d_gate_message": "ok",
        "figures_generated": 1,
        "warnings": ["State-space transition vector field skipped in this first pass; no tissue-space arrows were drawn."],
    }

    text = m4v.visualization_report(summary)

    assert "cross-time physical tissue arrows drawn: `False`" in text
    assert "tissue-space arrows" in text
