import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from nichefate.transition import full_transition_schema_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m3_16_freeze_full_m3_qc_report.py"
SPEC = importlib.util.spec_from_file_location("m3_final_qc_freeze", SCRIPT_PATH)
m3_16 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m3_16
SPEC.loader.exec_module(m3_16)


def expectations() -> m3_16.FreezeExpectations:
    return m3_16.FreezeExpectations(
        shard_count=2,
        total_edge_rows=8,
        candidate_k=2,
        expected_time_pairs=(("D0", "D3"),),
        row_sum_atol=1e-8,
    )


def synthetic_plan(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for idx in range(2):
        source_slice = f"slice_{idx}"
        output_dir = tmp_path / "full_by_shard" / "D0_to_D3" / source_slice
        stem = f"D0_to_D3__{source_slice}"
        rows.append(
            {
                "shard_id": f"m3_full_{idx + 1:04d}",
                "source_time": "D0",
                "target_time": "D3",
                "source_day": 0.0,
                "target_day": 3.0,
                "time_delta": 3.0,
                "source_slice_id": source_slice,
                "source_slice_file": f"{source_slice}.m0.h5ad",
                "source_rows": 2,
                "target_rows": 3,
                "candidate_k": 2,
                "expected_edge_rows": 4,
                "selected_backend": "sklearn_exact",
                "output_dir": str(output_dir),
                "output_parquet": str(output_dir / f"candidate_edges_{stem}.parquet"),
                "shard_report": str(output_dir / f"shard_report_{stem}.md"),
                "reuse_existing_pilot_allowed": False,
                "requires_explicit_approval": True,
            }
        )
    return pd.DataFrame(rows)


def edge_frame(source_slice_id: str) -> pd.DataFrame:
    rows = []
    for source_idx in range(2):
        for rank in range(2):
            rows.append(
                {
                    "source_anchor_id": f"{source_slice_id}::s::{source_idx}",
                    "target_anchor_id": f"target::{rank}",
                    "source_anchor_index": source_idx,
                    "target_anchor_index": rank,
                    "source_time": "D0",
                    "target_time": "D3",
                    "source_day": 0.0,
                    "target_day": 3.0,
                    "time_delta": 3.0,
                    "source_slice_id": source_slice_id,
                    "target_slice_id": f"target_slice_{rank}",
                    "source_slice_file": f"{source_slice_id}.m0.h5ad",
                    "target_slice_file": f"target_{rank}.m0.h5ad",
                    "source_mouse_id": "source_mouse",
                    "target_mouse_id": f"target_mouse_{rank}",
                    "evidence_mode": "pseudo_lineage",
                    "raw_molecular_distance": 1.0,
                    "raw_composition_distance": 1.0,
                    "raw_entropy_distance": 1.0,
                    "raw_spatial_summary_distance": 1.0,
                    "raw_topology_distance": 1.0,
                    "raw_pseudotime_score": 0.0,
                    "raw_barcode_score": 0.0,
                    "scaled_molecular_distance": 0.0,
                    "scaled_composition_distance": 0.0,
                    "scaled_entropy_distance": 0.0,
                    "scaled_spatial_summary_distance": 0.0,
                    "scaled_topology_distance": 0.0,
                    "scaled_pseudotime_score": 0.0,
                    "scaled_barcode_score": 0.0,
                    "scaling_method_molecular": "toy",
                    "scaling_method_composition": "toy",
                    "scaling_method_entropy": "toy",
                    "scaling_method_spatial_summary": "toy",
                    "scaling_method_topology": "toy",
                    "zero_variance_molecular": False,
                    "zero_variance_composition": False,
                    "zero_variance_entropy": False,
                    "zero_variance_spatial_summary": False,
                    "zero_variance_topology": False,
                    "source_mass": 1.0,
                    "target_mass": 1.0,
                    "growth_prior": 1.0,
                    "unbalanced_weight": 1.0,
                    "mass_adjusted_weight": 1.0,
                    "combined_cost": 0.0,
                    "tau_pair": 1.0,
                    "raw_edge_weight": 1.0,
                    "row_normalized_transition_prob": 0.5,
                }
            )
    return pd.DataFrame(rows)[full_transition_schema_columns()]


def write_shard_outputs(plan: pd.DataFrame, invalid_first: bool = False) -> None:
    for idx, shard in plan.iterrows():
        frame = edge_frame(str(shard["source_slice_id"]))
        if invalid_first and idx == 0:
            frame = frame.iloc[:-1]
        path = Path(shard["output_parquet"])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        Path(shard["shard_report"]).write_text("synthetic shard report\n", encoding="utf-8")


def write_control_outputs(tmp_path: Path, plan: pd.DataFrame, failed_text: str = "") -> None:
    output_root = tmp_path / "full_by_shard"
    reports_dir = tmp_path / "reports"
    output_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for _, shard in plan.iterrows():
        records.append(
            {
                "shard_id": shard["shard_id"],
                "status": "COMPLETED",
                "observed_edge_rows": int(shard["expected_edge_rows"]),
                "runtime_seconds": 1.25,
                "max_rss_gib": 2.5,
                "backend": "sklearn_exact",
                "tau_pair": 1.0,
            }
        )
    pd.DataFrame(records).to_csv(output_root / "completed_shards.csv", index=False)
    (output_root / "failed_shards.txt").write_text(failed_text, encoding="utf-8")
    pd.DataFrame({"shard_id": plan["shard_id"]}).to_csv(output_root / "full_m3_manifest.csv", index=False)
    (output_root / "full_m3_manifest.json").write_text('{"records": []}\n', encoding="utf-8")
    pd.DataFrame({"source_time": ["D0"], "target_time": ["D3"], "observed_edge_rows": [8]}).to_csv(
        reports_dir / "m3_full_m3_run_summary.csv",
        index=False,
    )
    (reports_dir / "m3_full_m3_run_summary.md").write_text("summary\n", encoding="utf-8")


def test_final_freeze_rejects_missing_shard_outputs(tmp_path: Path) -> None:
    plan = synthetic_plan(tmp_path)
    write_control_outputs(tmp_path, plan)

    with pytest.raises(RuntimeError, match="Invalid or incomplete"):
        m3_16.validate_existing_shards_for_freeze(
            plan,
            pd.read_csv(tmp_path / "full_by_shard" / "completed_shards.csv"),
        )


def test_final_freeze_rejects_nonempty_failed_shards(tmp_path: Path) -> None:
    plan = synthetic_plan(tmp_path)
    write_shard_outputs(plan)
    write_control_outputs(tmp_path, plan, failed_text="m3_full_0001\tboom\n")

    with pytest.raises(RuntimeError, match="failed_shards.txt is not empty"):
        m3_16.load_control_outputs(tmp_path / "full_by_shard", tmp_path / "reports", expectations())


def test_final_freeze_rejects_wrong_total_rows(tmp_path: Path) -> None:
    plan = synthetic_plan(tmp_path)
    write_shard_outputs(plan)
    write_control_outputs(tmp_path, plan)
    control = m3_16.load_control_outputs(tmp_path / "full_by_shard", tmp_path / "reports", expectations())
    records = m3_16.validate_existing_shards_for_freeze(plan, control["completed"])
    wrong = m3_16.FreezeExpectations(
        shard_count=2,
        total_edge_rows=9,
        candidate_k=2,
        expected_time_pairs=(("D0", "D3"),),
        row_sum_atol=1e-8,
    )

    with pytest.raises(RuntimeError, match="total_observed_edge_rows"):
        m3_16.validate_final_criteria(plan, records, control, wrong)


def test_final_freeze_accepts_valid_synthetic_completed_shards(tmp_path: Path) -> None:
    plan = synthetic_plan(tmp_path)
    write_shard_outputs(plan)
    write_control_outputs(tmp_path, plan)

    outputs = m3_16.build_freeze_outputs(
        plan,
        tmp_path / "full_by_shard",
        tmp_path / "reports",
        tmp_path / "reports" / "figures" / "full_m3",
        tmp_path / "reports" / "figures" / "full_m3_final",
        expectations=expectations(),
        generate_figures=False,
    )

    assert len(outputs["manifest"]) == 2
    assert int(outputs["summary"]["observed_edge_rows"].sum()) == 8
    assert outputs["criteria"]["failed_shards_eq_zero"]


def test_handoff_report_contains_no_downstream_confirmations(tmp_path: Path) -> None:
    plan = synthetic_plan(tmp_path)
    write_shard_outputs(plan)
    write_control_outputs(tmp_path, plan)
    outputs = m3_16.build_freeze_outputs(
        plan,
        tmp_path / "full_by_shard",
        tmp_path / "reports",
        tmp_path / "reports" / "figures" / "full_m3",
        tmp_path / "reports" / "figures" / "full_m3_final",
        expectations=expectations(),
        generate_figures=False,
    )

    report = m3_16.handoff_report(
        outputs["summary"],
        outputs["manifest"],
        outputs["figure_inventory"],
        outputs["criteria"],
        outputs["disk_usage_bytes"],
        outputs["figure_warnings"],
        tmp_path / "reports",
        tmp_path / "reports" / "figures" / "full_m3_final",
    )

    for text in [
        "no global Markov P was assembled",
        "no GPCCA was run",
        "no fate probability was computed",
        "no Branched NicheFlow was run",
        "no M5 was run",
        "no regulator analysis was run",
    ]:
        assert text in report


def test_figure_inventory_marks_existing_and_missing_figures(tmp_path: Path) -> None:
    existing_dir = tmp_path / "figures" / "full_m3"
    final_dir = tmp_path / "figures" / "full_m3_final"
    existing_dir.mkdir(parents=True)
    (existing_dir / m3_16.EXPECTED_EXISTING_FIGURES[0]).write_text("png", encoding="utf-8")

    inventory = m3_16.inventory_figures(existing_dir, final_dir)

    first = inventory[inventory["figure_name"] == m3_16.EXPECTED_EXISTING_FIGURES[0]].iloc[0]
    final = inventory[inventory["figure_name"] == m3_16.FINAL_FIGURE_NAMES[0]].iloc[0]
    assert bool(first["exists"])
    assert not bool(final["exists"])
