import importlib.util
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m4c_02_review_and_freeze_markov_fate_results.py"
SPEC = importlib.util.spec_from_file_location("m4c_final_review", SCRIPT_PATH)
m4c_review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4c_review
SPEC.loader.exec_module(m4c_review)


def toy_terminal_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "terminal_macrostate_id": [0, 1, 2, 3],
            "terminal_macrostate_label": ["tm0", "tm1", "tm2", "tm3"],
            "n_nodes": [100, 8, 90, 70],
            "fraction_final_nodes": [0.35, 0.02, 0.30, 0.23],
            "incoming_mass_sum_structural": [100.0, 2.0, 90.0, 75.0],
            "incoming_degree_sum_structural": [1000, 10, 850, 700],
            "dominant_cell_type_l1": ["epi", "rare", "mixed", "stromal"],
            "dominant_cell_type_l1_fraction": [0.80, 0.90, 0.35, 0.72],
            "cell_type_l1_entropy": [0.20, 0.10, 1.80, 0.30],
        }
    )


def toy_fate_mass() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "terminal_macrostate": [0, 1, 2, 3],
            "terminal_macrostate_label": ["tm0", "tm1", "tm2", "tm3"],
            "total_fate_mass": [400.0, 20.0, 300.0, 250.0],
            "total_fate_mass_fraction": [0.40, 0.02, 0.30, 0.25],
            "dominant_fate_node_count": [420, 10, 260, 240],
            "dominant_fate_node_fraction": [0.42, 0.01, 0.26, 0.24],
            "mean_dominant_fate_fraction": [0.50, 0.03, 0.28, 0.25],
            "max_dominant_fate_fraction": [0.60, 0.10, 0.42, 0.72],
        }
    )


def no_association_warnings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "terminal_macrostate": [0, 1, 2, 3],
            "max_association_group": ["s0", "s1", "s2", "s3"],
            "association_warning": [False, False, False, False],
        }
    )


def test_confidence_tier_assignment_on_toy_terminal_summary() -> None:
    confidence = m4c_review.assign_confidence_tiers(
        toy_terminal_summary(),
        toy_fate_mass(),
        no_association_warnings(),
        no_association_warnings(),
    )

    tiers = dict(zip(confidence["terminal_macrostate"], confidence["confidence_tier"], strict=True))
    assert tiers[0] == "high_confidence_terminal_like"
    assert tiers[1] == "low_size_or_low_mass_endpoint"
    assert tiers[2] == "mixed_or_intermediate_final_time_state"
    assert len(confidence) == len(toy_terminal_summary())


def test_detection_of_dominant_fate_collapse() -> None:
    by_time = pd.DataFrame(
        {
            "time_day": [0, 0, 1, 1],
            "time": ["early", "early", "late", "late"],
            "terminal_macrostate": [0, 1, 0, 1],
            "terminal_macrostate_label": ["tm0", "tm1", "tm0", "tm1"],
            "dominant_fate_fraction": [0.90, 0.10, 0.80, 0.20],
            "mean_probability": [0.85, 0.15, 0.75, 0.25],
        }
    )

    collapse = m4c_review.detect_dominant_fate_collapse(by_time, dominance_threshold=0.50)

    assert collapse.loc[0, "terminal_macrostate"] == 0
    assert collapse.loc[0, "dominates_all_time_points"]
    assert collapse.loc[0, "warning"]


def test_detection_of_slice_and_mouse_association_warning() -> None:
    summary = pd.DataFrame(
        {
            "slice_id": ["s0", "s1", "s2"],
            "mouse_id": ["m0", "m1", "m2"],
            "terminal_macrostate": [0, 0, 0],
            "terminal_macrostate_label": ["tm0", "tm0", "tm0"],
            "dominant_fate_fraction": [0.95, 0.05, 0.05],
            "normalized_mass_fraction": [0.90, 0.05, 0.05],
        }
    )

    slice_warnings = m4c_review.association_warnings(m4c_review.group_variability_summary(summary, "slice_id"))
    mouse_warnings = m4c_review.association_warnings(m4c_review.group_variability_summary(summary, "mouse_id"))

    assert slice_warnings["association_warning"].tolist() == [True]
    assert mouse_warnings["association_warning"].tolist() == [True]


def test_generation_of_final_inventory_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    figure_path = tmp_path / "figure.png"
    input_path.write_text("x\n", encoding="utf-8")
    output_path.write_text("y\n", encoding="utf-8")
    figure_path.write_text("z\n", encoding="utf-8")
    inputs = {
        "fate_matrix": input_path,
        "node_summary": input_path,
        "by_time": input_path,
        "by_slice": input_path,
        "by_mouse": input_path,
        "qc": input_path,
        "m4c_report": input_path,
        "m4c_schema": input_path,
        "terminal_assignments": input_path,
        "terminal_summary": input_path,
        "terminal_feature_summary": input_path,
        "m4b_design_report": input_path,
    }
    outputs = {
        "final_review_report": output_path,
        "freeze_summary": output_path,
        "confidence_tiers": output_path,
        "interpretation_cautions": output_path,
        "result_inventory": output_path,
        "final_figures_dir": tmp_path,
    }

    inventory = m4c_review.inventory_rows(inputs, outputs, [figure_path])

    assert {"m4c_input", "m4b_input", "m4c_final_review_output", "m4c_final_review_figure"} <= set(
        inventory["artifact_category"]
    )
    assert inventory["exists"].all()


def test_no_terminal_macrostate_merging() -> None:
    confidence = m4c_review.assign_confidence_tiers(
        toy_terminal_summary(),
        toy_fate_mass(),
        no_association_warnings(),
        no_association_warnings(),
    )

    assert confidence["terminal_macrostate"].tolist() == [0, 1, 2, 3]


def test_no_fate_recomputation_or_fate_matrix_load_contract() -> None:
    assert m4c_review.FATE_PROBABILITIES_RECOMPUTED is False
    assert m4c_review.FATE_MATRIX_LOADED is False


def test_no_forbidden_downstream_review_outputs(tmp_path: Path) -> None:
    outputs = m4c_review.review_output_paths({"reports_dir": tmp_path / "reports", "figures_dir": tmp_path / "figures"})
    forbidden = ["gpcca", "branched_nicheflow", "branchsbm", "m5", "regulator"]

    assert not any(token in str(path).lower() for key, path in outputs.items() if key != "final_figures_dir" for token in forbidden)
    assert m4c_review.NO_DOWNSTREAM_FLAGS["no_gpcca"] is True
    assert m4c_review.NO_DOWNSTREAM_FLAGS["no_branched_nicheflow_training"] is True
    assert m4c_review.NO_DOWNSTREAM_FLAGS["no_m5"] is True
    assert m4c_review.NO_DOWNSTREAM_FLAGS["no_regulator_analysis"] is True


def test_barcode_compatible_interpretation_note_exists() -> None:
    text = m4c_review.interpretation_cautions_text(
        pd.DataFrame({"confidence_tier": ["high_confidence_terminal_like"]}),
        pd.DataFrame({"association_warning": [False]}),
        pd.DataFrame({"association_warning": [False]}),
    )

    assert "barcode-aware M3" in text
    assert "preserving the M4C fate interface" in text
    assert "pseudo-lineage/time-coupled Markov fate probabilities" in text
