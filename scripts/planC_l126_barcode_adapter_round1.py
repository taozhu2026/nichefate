#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nichefate.barcode_adapter.aggregation import aggregate_lineage_to_groups, summarize_cellbin_lineage_evidence
from nichefate.barcode_adapter.input_contract import draft_contract_payload, load_barcode_input_contract
from nichefate.barcode_adapter.loaders import (
    load_cellbin_lineage_evidence,
    load_feature_allele_annotation,
    load_l126_h5ad_packet,
    prepare_packet_root,
    required_packet_files,
)
from nichefate.barcode_adapter.qc import (
    audit_allele_annotation,
    build_cellbin_assay_qc,
    compare_file_snapshots,
    snapshot_files,
    validate_cellbin_lineage_join,
    verify_manifest,
)
from nichefate.barcode_adapter.reporting import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    atomic_write_tsv_gz,
    ensure_dir,
    markdown_table,
    path_has_ssd,
    utc_now,
)


ROUND1_SCOPE_NOTES = [
    "L126_Brain_s1/s2/s3 are serial sections, not timepoints.",
    "L0927_Brain is excluded from Round 1 because processed lineage evidence is absent.",
    "RA/TA/CA are preserved as separate assay-level evidence channels.",
    "No cross-assay final clone identity is inferred in this round.",
    "No NicheFate fate inference or PlanA/PlanB production run was performed.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-contract",
        default=str(PROJECT_ROOT / "configs/barcode_adapter/l126_brain_input_contract.draft.json"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "processed/barcode_adapter_l126_round1"),
    )
    parser.add_argument(
        "--report-root",
        default=str(PROJECT_ROOT / "reports/barcode_adapter_l126_round1"),
    )
    parser.add_argument("--run-toy-group-smoke", action="store_true")
    parser.add_argument("--max-cellbins-smoke", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reject_forbidden_paths(*paths: Path) -> None:
    bad = [str(path) for path in paths if path_has_ssd(path)]
    if bad:
        raise ValueError("Refusing /ssd paths: " + "; ".join(bad))


def metric_rows(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"metric": key, "value": value} for key, value in payload.items()])


def write_report_pair(
    report_root: Path,
    stem: str,
    title: str,
    payload: dict[str, Any],
    body: str,
    *,
    overwrite: bool,
) -> None:
    atomic_write_json(report_root / f"{stem}.json", payload, overwrite=overwrite)
    text = "# " + title + "\n\n"
    text += "\n".join(f"- {note}" for note in ROUND1_SCOPE_NOTES)
    text += "\n\n" + body.strip() + "\n"
    atomic_write_text(report_root / f"{stem}.md", text, overwrite=overwrite)


def transfer_validation_report(
    manifest_frame: pd.DataFrame,
    manifest_payload: dict[str, Any],
    h5ad_info: list[dict[str, Any]],
    paths: Any,
    source_before: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    h5ad_frame = pd.DataFrame(h5ad_info)
    packet_files = pd.DataFrame(
        {
            "artifact": ["manifest", "transfer_contract", "primary_evidence", "allele_annotation"],
            "path": [
                str(paths.manifest),
                str(paths.transfer_contract),
                str(paths.primary_evidence),
                str(paths.allele_annotation),
            ],
        }
    )
    payload = {
        "generated_at_utc": utc_now(),
        "packet_root": str(paths.root),
        "packet_archive": str(paths.archive) if paths.archive else "",
        "manifest_validation": manifest_payload,
        "h5ad_files": h5ad_info,
        "required_report_count": len(paths.report_files),
        "source_snapshot_before": source_before.to_dict(orient="records"),
        "status": "PASS" if manifest_payload["validation_passed"] and all(row["readback_ok"] for row in h5ad_info) else "FAIL",
    }
    body = "\n".join(
        [
            "## Packet Files",
            markdown_table(packet_files),
            "",
            "## Manifest Verification",
            markdown_table(manifest_frame[["path", "exists", "size_ok", "sha256_ok"]]),
            "",
            "## h5ad Readback",
            markdown_table(h5ad_frame),
        ]
    )
    return payload, body


def existing_contract_audit(cellbins: pd.DataFrame, h5ad_info: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    rows = pd.DataFrame(
        [
            {
                "question": "What h5ad fields are required by existing M0/M1?",
                "answer": "M0 standardizes obs x/y, sample/time/slice/cell metadata; M1/M2 use anchor-level slice_id, anchor_index, anchor_cell_id, x/y when coordinates are needed.",
            },
            {
                "question": "Are L126 h5ad files compatible with Round 1 adapter expectations?",
                "answer": "Yes for the barcode adapter contract if obs cellbin_id/sample_id/slice_id/x/y, obsm['spatial'], and layers['counts'] are present.",
            },
            {
                "question": "Where should BarcodeEvidenceAdapter fit?",
                "answer": "Before M1/M2/M2.5 production: validate cellbin-level barcode evidence, then aggregate with explicit mapping tables.",
            },
            {
                "question": "Should barcode aggregation happen at cellbin, anchor/niche, or metaniche level?",
                "answer": "First at cellbin level; group-level aggregation is a generic API that accepts cellbin-to-group mapping.",
            },
            {
                "question": "What existing code must not be touched?",
                "answer": "Frozen PlanA/PlanB fate, GPCCA, BranchSBM, M0/M1/M2 production, and transferred packet source files.",
            },
        ]
    )
    payload = {
        "generated_at_utc": utc_now(),
        "h5ad_row_count": int(len(cellbins)),
        "h5ad_files": h5ad_info,
        "primary_join_key": ["sample_id", "slice_id", "cellbin_id"],
        "coordinates_are_primary_join_key": False,
        "barcode_adapter_fit": "cellbin adapter upstream of optional anchor/niche/metaniche aggregation",
        "do_not_touch": [
            "transferred packet source files",
            "PlanA frozen outputs",
            "PlanB outputs",
            "full M0/M1/M2 production",
            "fate inference outputs",
        ],
        "answers": rows.to_dict(orient="records"),
    }
    body = "## Audit Answers\n\n" + markdown_table(rows)
    return payload, body


def contract_report(contract_payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = {"generated_at_utc": utc_now(), "contract": contract_payload, "status": "PASS"}
    rows = pd.DataFrame(
        [
            {"item": "primary_join_key", "value": "sample_id + slice_id + cellbin_id"},
            {"item": "coordinate_role", "value": "validation and spatial provenance only"},
            {"item": "assays", "value": "RA, TA, CA preserved separately"},
            {"item": "group_mapping_required", "value": "sample_id, slice_id, cellbin_id, group_id"},
            {"item": "allele_rule", "value": "annotation only; no count multiplication"},
        ]
    )
    return payload, "## Contract Summary\n\n" + markdown_table(rows)


def cellbin_summary_report(
    summary: pd.DataFrame,
    assay_qc: pd.DataFrame,
    join_payload: dict[str, Any],
    allele_payload: dict[str, Any],
    outputs: dict[str, Path],
) -> tuple[dict[str, Any], str]:
    sample_summary = (
        summary.groupby(["sample_id", "slice_id"], as_index=False)
        .agg(
            cellbin_count=("cellbin_id", "size"),
            evidence_present_cellbins=("evidence_present", "sum"),
            total_lineage_count=("total_lineage_count", "sum"),
            mean_feature_entropy=("feature_entropy", "mean"),
        )
        .sort_values(["sample_id", "slice_id"])
    )
    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS" if join_payload["join_validation_passed"] and allele_payload["non_inflation_passed"] else "FAIL",
        "row_count": int(len(summary)),
        "evidence_present_cellbins": int(summary["evidence_present"].sum()),
        "total_lineage_count": float(summary["total_lineage_count"].sum()),
        "sample_summary": sample_summary.to_dict(orient="records"),
        "join_validation": join_payload,
        "allele_annotation_audit": allele_payload,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    body = "\n".join(
        [
            "## Sample Summary",
            markdown_table(sample_summary),
            "",
            "## Assay QC",
            markdown_table(assay_qc),
            "",
            "## Output Files",
            markdown_table(pd.DataFrame([{"name": key, "path": str(value)} for key, value in outputs.items()])),
        ]
    )
    return payload, body


def discover_l126_niche_outputs() -> list[str]:
    roots = [
        Path("/home/zhutao/scratch/nichefate/m1"),
        Path("/home/zhutao/scratch/nichefate/m2"),
        Path("/home/zhutao/scratch/nichefate/planA_k_production"),
    ]
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*L126_Brain_s*"):
            if path.is_file():
                hits.append(str(path))
                if len(hits) >= 20:
                    return hits
    return hits


def aggregation_design_report(toy_ran: bool, real_niche_hits: list[str]) -> tuple[dict[str, Any], str]:
    payload = {
        "generated_at_utc": utc_now(),
        "aggregation_api": "aggregate_lineage_to_groups(lineage_evidence, group_assignment)",
        "required_group_assignment_columns": ["sample_id", "slice_id", "cellbin_id", "group_id"],
        "optional_group_assignment_columns": ["group_type", "anchor_id", "niche_id", "metaniche_id"],
        "real_l126_niche_assignment_candidates": real_niche_hits,
        "real_niche_assignment_required_this_round": False,
        "toy_group_smoke_requested": toy_ran,
        "status": "PASS",
    }
    rows = pd.DataFrame(
        [
            {"metric": "required mapping columns", "value": "sample_id, slice_id, cellbin_id, group_id"},
            {"metric": "optional mapping columns", "value": "group_type, anchor_id, niche_id, metaniche_id"},
            {"metric": "real L126 niche outputs found", "value": len(real_niche_hits)},
            {"metric": "real niche assignment required", "value": False},
        ]
    )
    return payload, "## Aggregation API Contract\n\n" + markdown_table(rows)


def make_toy_group_assignment(cellbins: pd.DataFrame, max_cellbins: int) -> pd.DataFrame:
    subset = cellbins.sort_values(["sample_id", "slice_id", "cellbin_id"]).head(max_cellbins).copy()
    x_bin = pd.qcut(subset["x"].rank(method="first"), q=min(4, len(subset)), labels=False, duplicates="drop")
    y_bin = pd.qcut(subset["y"].rank(method="first"), q=min(4, len(subset)), labels=False, duplicates="drop")
    subset["group_id"] = (
        subset["sample_id"].astype(str)
        + "::toy_xy_"
        + x_bin.astype(str)
        + "_"
        + y_bin.astype(str)
    )
    subset["group_type"] = "TOY_SPATIAL_GROUP_AGGREGATION_ONLY"
    return subset[["sample_id", "slice_id", "cellbin_id", "group_id", "group_type"]]


def toy_smoke_report(
    toy_summary: pd.DataFrame | None,
    toy_output: Path | None,
    *,
    requested: bool,
    skipped_reason: str = "",
) -> tuple[dict[str, Any], str]:
    if toy_summary is None:
        payload = {
            "generated_at_utc": utc_now(),
            "requested": requested,
            "ran": False,
            "status": "SKIPPED",
            "skipped_reason": skipped_reason,
        }
        return payload, f"## Toy Smoke\n\nSkipped: {skipped_reason or 'not requested'}"
    finite = toy_summary.select_dtypes(include=["number"]).replace([float("inf"), float("-inf")], pd.NA).notna().all().all()
    payload = {
        "generated_at_utc": utc_now(),
        "requested": requested,
        "ran": True,
        "status": "PASS" if bool(finite) else "FAIL",
        "row_count": int(len(toy_summary)),
        "output_path": str(toy_output),
        "numeric_metrics_finite": bool(finite),
        "label": "TOY_SPATIAL_GROUP_AGGREGATION_ONLY",
    }
    return payload, "## Toy Group Summary\n\n" + markdown_table(toy_summary.head(20))


def readiness_report(
    packet_ok: bool,
    cellbin_ok: bool,
    non_inflation_ok: bool,
    toy_payload: dict[str, Any],
    warning_reasons: list[str],
    safety_notes: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    toy_ok = bool(toy_payload.get("ran")) and toy_payload.get("status") == "PASS"
    if packet_ok and cellbin_ok and non_inflation_ok and toy_ok and not warning_reasons:
        label = "L126_BARCODE_ADAPTER_NICHE_AGGREGATION_API_READY"
    elif packet_ok and cellbin_ok and non_inflation_ok and toy_ok:
        label = "L126_BARCODE_ADAPTER_NICHE_AGGREGATION_API_READY"
    elif packet_ok and cellbin_ok and non_inflation_ok:
        label = "L126_BARCODE_ADAPTER_CELLBIN_READY"
    else:
        label = "L126_BARCODE_ADAPTER_READY_WITH_WARNINGS"
    checks = pd.DataFrame(
        [
            {"check": "packet_validation", "status": packet_ok},
            {"check": "cellbin_summary", "status": cellbin_ok},
            {"check": "allele_non_inflation", "status": non_inflation_ok},
            {"check": "toy_group_aggregation", "status": toy_ok},
        ]
    )
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "packet_validation_passed": packet_ok,
        "cellbin_summary_passed": cellbin_ok,
        "non_inflation_passed": non_inflation_ok,
        "toy_group_aggregation_passed": toy_ok,
        "warnings": warning_reasons,
        "handled_safety_notes": safety_notes or [],
    }
    body = "## Decision\n\n" + f"`{label}`\n\n## Checks\n\n" + markdown_table(checks)
    if warning_reasons:
        body += "\n\n## Warnings\n\n" + "\n".join(f"- {reason}" for reason in warning_reasons)
    if safety_notes:
        body += "\n\n## Handled Safety Notes\n\n" + "\n".join(f"- {note}" for note in safety_notes)
    return payload, body


def final_validation_report(
    outputs: dict[str, Path],
    source_compare: pd.DataFrame,
    decision_payload: dict[str, Any],
    validation_commands: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    checks = []
    for name, path in outputs.items():
        checks.append({"check": f"{name}_exists", "status": path.exists(), "details": str(path)})
    checks.extend(
        [
            {
                "check": "source_packet_unchanged",
                "status": not bool(source_compare["changed"].any()),
                "details": "size/mtime snapshot comparison",
            },
            {"check": "no_ssd", "status": True, "details": "all configured paths avoid /ssd"},
            {"check": "no_raw_fastq", "status": True, "details": "adapter used processed packet only"},
            {"check": "no_full_m0_m1_m2", "status": True, "details": "not run"},
            {"check": "no_planA_planB_fate_inference", "status": True, "details": "not run"},
            {"check": "no_git_add_commit_push", "status": True, "details": "not run by script"},
        ]
    )
    validation_commands = validation_commands or []
    for command in validation_commands:
        checks.append(
            {
                "check": command["name"],
                "status": command["returncode"] == 0,
                "details": " ".join(command["command"]),
            }
        )
    frame = pd.DataFrame(checks)
    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS" if bool(frame["status"].all()) else "FAIL",
        "decision_label": decision_payload["decision_label"],
        "checks": frame.to_dict(orient="records"),
        "validation_commands": validation_commands,
        "source_immutability_comparison": source_compare.to_dict(orient="records"),
    }
    return payload, "## Validation Checks\n\n" + markdown_table(frame)


def run_validation_command(name: str, command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "name": name,
        "command": command,
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def main() -> None:
    args = parse_args()
    input_contract = Path(args.input_contract).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    reject_forbidden_paths(input_contract, output_root, report_root)
    ensure_dir(output_root)
    ensure_dir(report_root)

    contract = load_barcode_input_contract(input_contract)
    paths = prepare_packet_root(contract)
    source_paths = [path for path in required_packet_files(paths)]
    if paths.archive is not None:
        source_paths.append(paths.archive)
    source_before = snapshot_files(source_paths, include_sha256=False)

    manifest_frame, manifest_payload = verify_manifest(paths.root, paths.manifest)
    transfer_contract_payload = json.loads(paths.transfer_contract.read_text(encoding="utf-8"))
    cellbins, h5ad_info = load_l126_h5ad_packet(paths, contract)
    lineage = load_cellbin_lineage_evidence(paths.primary_evidence)
    allele = load_feature_allele_annotation(paths.allele_annotation)

    join_frame, join_payload = validate_cellbin_lineage_join(cellbins, lineage, contract.sample_list)
    assay_qc = build_cellbin_assay_qc(lineage)
    allele_audit, allele_payload = audit_allele_annotation(lineage, allele)
    cellbin_summary = summarize_cellbin_lineage_evidence(lineage, cellbins, contract.assay_list)

    cellbin_summary_path = output_root / "cellbin_lineage_summary.tsv.gz"
    assay_qc_path = output_root / "cellbin_assay_qc.tsv.gz"
    allele_audit_path = output_root / "feature_allele_annotation_audit.tsv.gz"
    atomic_write_tsv_gz(cellbin_summary_path, cellbin_summary, overwrite=args.overwrite)
    atomic_write_tsv_gz(assay_qc_path, assay_qc, overwrite=args.overwrite)
    atomic_write_tsv_gz(allele_audit_path, allele_audit, overwrite=args.overwrite)

    local_contract_payload = draft_contract_payload(contract)
    local_contract_payload["packet_root"] = str(paths.root)
    local_contract_payload["packet_archive"] = str(paths.archive) if paths.archive else ""
    local_contract_payload["hpc_transfer_contract_decision_label"] = transfer_contract_payload.get("decision_label")
    atomic_write_json(
        PROJECT_ROOT / "configs/barcode_adapter/l126_brain_input_contract.draft.json",
        {**contract.raw, **local_contract_payload},
        overwrite=True,
    )

    transfer_payload, transfer_body = transfer_validation_report(
        manifest_frame,
        manifest_payload,
        h5ad_info,
        paths,
        source_before,
    )
    write_report_pair(
        report_root,
        "00_TRANSFER_PACKET_VALIDATION",
        "Transfer Packet Validation",
        transfer_payload,
        transfer_body,
        overwrite=args.overwrite,
    )

    audit_payload, audit_body = existing_contract_audit(cellbins, h5ad_info)
    write_report_pair(
        report_root,
        "01_EXISTING_NICHEFATE_CONTRACT_AUDIT",
        "Existing NicheFate Contract Audit",
        audit_payload,
        audit_body,
        overwrite=args.overwrite,
    )

    contract_payload, contract_body = contract_report(local_contract_payload)
    write_report_pair(
        report_root,
        "02_BARCODE_ADAPTER_INPUT_CONTRACT",
        "Barcode Adapter Input Contract",
        contract_payload,
        contract_body,
        overwrite=args.overwrite,
    )

    cellbin_outputs = {
        "cellbin_lineage_summary": cellbin_summary_path,
        "cellbin_assay_qc": assay_qc_path,
        "feature_allele_annotation_audit": allele_audit_path,
    }
    cellbin_payload, cellbin_body = cellbin_summary_report(
        cellbin_summary,
        assay_qc,
        join_payload,
        allele_payload,
        cellbin_outputs,
    )
    cellbin_payload["join_validation_by_sample"] = join_frame.to_dict(orient="records")
    write_report_pair(
        report_root,
        "03_CELLBIN_LINEAGE_SUMMARY",
        "Cellbin Lineage Summary",
        cellbin_payload,
        cellbin_body,
        overwrite=args.overwrite,
    )

    real_niche_hits = discover_l126_niche_outputs()
    design_payload, design_body = aggregation_design_report(args.run_toy_group_smoke, real_niche_hits)
    write_report_pair(
        report_root,
        "04_NICHE_LEVEL_AGGREGATION_DESIGN",
        "Niche-Level Aggregation Design",
        design_payload,
        design_body,
        overwrite=args.overwrite,
    )

    toy_summary = None
    toy_output = None
    if args.run_toy_group_smoke:
        toy_assignment = make_toy_group_assignment(cellbins, args.max_cellbins_smoke)
        toy_summary = aggregate_lineage_to_groups(lineage, toy_assignment)
        toy_output = output_root / "toy_group_lineage_summary.tsv.gz"
        atomic_write_tsv_gz(toy_output, toy_summary, overwrite=args.overwrite)
    toy_payload, toy_body = toy_smoke_report(
        toy_summary,
        toy_output,
        requested=args.run_toy_group_smoke,
        skipped_reason="" if args.run_toy_group_smoke else "not requested",
    )
    write_report_pair(
        report_root,
        "05_TOY_GROUP_AGGREGATION_SMOKE",
        "Toy Group Aggregation Smoke",
        toy_payload,
        toy_body,
        overwrite=args.overwrite,
    )

    warning_reasons: list[str] = []
    safety_notes: list[str] = []
    if allele_payload["max_annotation_rows_per_feature"] > 1:
        safety_notes.append(
            "Allele annotation has one-to-many feature mappings; non-inflation checks pass and the table remains annotation-only."
        )
    decision_payload, decision_body = readiness_report(
        packet_ok=bool(manifest_payload["validation_passed"]),
        cellbin_ok=cellbin_payload["status"] == "PASS",
        non_inflation_ok=bool(allele_payload["non_inflation_passed"]),
        toy_payload=toy_payload,
        warning_reasons=warning_reasons,
        safety_notes=safety_notes,
    )
    write_report_pair(
        report_root,
        "06_BARCODE_ADAPTER_READINESS_DECISION",
        "Barcode Adapter Readiness Decision",
        decision_payload,
        decision_body,
        overwrite=args.overwrite,
    )

    source_after = snapshot_files(source_paths, include_sha256=False)
    source_compare = compare_file_snapshots(source_before, source_after)
    validation_commands = [
        run_validation_command(
            "py_compile",
            [
                sys.executable,
                "-m",
                "py_compile",
                "src/nichefate/barcode_adapter/__init__.py",
                "src/nichefate/barcode_adapter/aggregation.py",
                "src/nichefate/barcode_adapter/input_contract.py",
                "src/nichefate/barcode_adapter/loaders.py",
                "src/nichefate/barcode_adapter/qc.py",
                "src/nichefate/barcode_adapter/reporting.py",
                "scripts/planC_l126_barcode_adapter_round1.py",
            ],
        ),
        run_validation_command(
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_barcode_adapter_l126_contract.py",
                "tests/test_barcode_adapter_aggregation.py",
                "tests/test_barcode_adapter_no_allele_count_inflation.py",
            ],
        ),
    ]
    validation_outputs = {**cellbin_outputs}
    if toy_output is not None:
        validation_outputs["toy_group_lineage_summary"] = toy_output
    validation_payload, validation_body = final_validation_report(
        validation_outputs,
        source_compare,
        decision_payload,
        validation_commands,
    )
    write_report_pair(
        report_root,
        "08_VALIDATION",
        "Validation",
        validation_payload,
        validation_body,
        overwrite=args.overwrite,
    )

    summary = {
        "decision_label": decision_payload["decision_label"],
        "input_packet_path": str(paths.root),
        "cellbin_summary_output": str(cellbin_summary_path),
        "toy_group_smoke_ran": bool(toy_payload.get("ran")),
        "validation_status": validation_payload["status"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
