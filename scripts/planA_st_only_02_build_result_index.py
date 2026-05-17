#!/usr/bin/env python
"""Build the PlanA-ST-only v1 result index.

The index points to final reports and figures but does not create, modify, or
stage figure binaries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nichefate.planA_st_only.module_registry import CLAIM_GUARDRAILS, FINAL_RESULT_PACKAGE, production_rows

INDEX_ROOT = ROOT / "reports" / "planA_st_only_v1_index"
FINAL_ROOT = ROOT / FINAL_RESULT_PACKAGE


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    return "\n".join(lines)


def result_manifest_rows() -> list[dict[str, object]]:
    rows = []
    for path in sorted(FINAL_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        suffix = path.suffix.lower()
        if "/figures/" in rel:
            artifact_type = "figure_reference"
            include_in_commit = "no"
        elif suffix in {".tsv", ".md", ".json"} and path.stat().st_size <= 1_000_000:
            artifact_type = "curated_report"
            include_in_commit = "yes"
        else:
            artifact_type = "large_generated_table"
            include_in_commit = "no"
        rows.append(
            {
                "path": rel,
                "artifact_type": artifact_type,
                "file_size": path.stat().st_size,
                "include_in_commit": include_in_commit,
                "reason": "index-only figure or large generated artifact" if include_in_commit == "no" else "small final result report",
            }
        )
    return rows


def infer_shows(value_type: str) -> str:
    mapping = {
        "workflow": "workflow",
        "macrostate_label": "macrostate label",
        "absorption": "absorption/fate probability",
        "diagnostic": "diagnostic",
        "membership_vs_absorption": "GPCCA membership and absorption/fate probability",
    }
    return mapping.get(value_type, value_type or "diagnostic")


def figure_rows() -> list[dict[str, object]]:
    provenance = FINAL_ROOT / "03_FINAL_FIGURE_SOURCE_PROVENANCE.tsv"
    rows = []
    for row in read_tsv(provenance):
        role = row.get("main_or_support", "")
        rows.append(
            {
                "figure_id": row.get("figure_id", ""),
                "title": row.get("title", ""),
                "main_or_supplementary": "main" if role == "main" else "supplementary",
                "png": row.get("png", ""),
                "pdf": row.get("pdf", ""),
                "shows": infer_shows(row.get("value_type", "")),
                "data_source": row.get("source_table", ""),
                "source_report": row.get("source_report", ""),
                "advisor_facing": "yes" if role == "main" else "no",
                "validation_note": row.get("validation_note", ""),
            }
        )
    return rows


def write_index() -> None:
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    modules = [
        {
            **row,
            "primary_legacy_modules": "; ".join(row["primary_legacy_modules"]),
            "primary_legacy_scripts": "; ".join(row["primary_legacy_scripts"]),
        }
        for row in production_rows()
    ]
    module_fields = [
        "order",
        "module_name",
        "legacy_name",
        "facade",
        "status",
        "role",
        "primary_legacy_modules",
        "primary_legacy_scripts",
    ]
    write_tsv(INDEX_ROOT / "01_MODULE_REGISTRY.tsv", modules, module_fields)

    manifest = result_manifest_rows()
    manifest_fields = ["path", "artifact_type", "file_size", "include_in_commit", "reason"]
    write_tsv(INDEX_ROOT / "02_FINAL_RESULT_MANIFEST.tsv", manifest, manifest_fields)

    figures = figure_rows()
    figure_fields = [
        "figure_id",
        "title",
        "main_or_supplementary",
        "png",
        "pdf",
        "shows",
        "data_source",
        "source_report",
        "advisor_facing",
        "validation_note",
    ]
    write_tsv(INDEX_ROOT / "03_FINAL_FIGURE_INDEX.tsv", figures, figure_fields)
    (INDEX_ROOT / "03_FINAL_FIGURE_INDEX.md").write_text(
        "# Final Figure Index\n\n" + markdown_table(figures, figure_fields) + "\n",
        encoding="utf-8",
    )

    claim_payload = {
        "required_wording": CLAIM_GUARDRAILS,
        "excluded_claims": [
            "warning-level terminal-like candidate",
            "maybe terminal",
            "validated biological endpoint",
            "DARLIN-supported fate",
            "barcode-backed transition",
            "final clone-supported fate",
        ],
        "boundary": "PlanA-ST-only v1 is barcode-free. DARLIN/barcode validation is future work.",
    }
    write_json(INDEX_ROOT / "04_CLAIM_BOUNDARY.json", claim_payload)
    claim_lines = ["# Claim Boundary", ""]
    claim_lines.extend(f"- {value}" for value in CLAIM_GUARDRAILS.values())
    claim_lines.extend(
        [
            "",
            "## Excluded Main-Result Claims",
            "",
            "- Do not present the result as supported by DARLIN lineage evidence.",
            "- Do not present transitions as backed by barcode evidence.",
            "- Do not present fate as supported by clone evidence.",
            "- Do not present endpoint status as biologically validated.",
            "",
            "Validation caveats belong in limitations sections only.",
        ]
    )
    (INDEX_ROOT / "04_CLAIM_BOUNDARY.md").write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    index_payload = {
        "decision": "PLAN_A_ST_ONLY_V1_INDEX_READY",
        "final_result_package": FINAL_RESULT_PACKAGE.as_posix(),
        "module_count": len(modules),
        "manifest_count": len(manifest),
        "figure_count": len(figures),
        "claim_boundary": "ST-only / barcode-free; DARLIN/barcode validation is future work.",
    }
    write_json(INDEX_ROOT / "00_PLAN_A_ST_ONLY_V1_INDEX.json", index_payload)
    index_lines = [
        "# PlanA-ST-only v1 Index",
        "",
        "PlanA-ST-only v1 has completed through corrected feature-only Kmix_A, corrected full GPCCA k=6, macrostate annotation, source/terminal diagnostics, CellRank-aligned terminal audit, Kmix_A absorption/fate probability to M5, sensitivity checks, final result package, and visualization QA.",
        "",
        "- Final result package: `reports/planA_k_final_result_package/`",
        "- Main figures: `reports/planA_k_final_result_package/figures/main_figures/`",
        "- Supplementary figures: `reports/planA_k_final_result_package/figures/supplementary_figures/`",
        "- Claim boundary: `reports/planA_st_only_v1_index/04_CLAIM_BOUNDARY.md`",
        "",
        "This is ST-only / barcode-free. DARLIN/barcode validation is a future development stage.",
    ]
    (INDEX_ROOT / "00_PLAN_A_ST_ONLY_V1_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def main() -> None:
    write_index()
    print(f"Wrote PlanA-ST-only v1 index under {INDEX_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
