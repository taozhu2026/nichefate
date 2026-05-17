#!/usr/bin/env python
"""Write PlanA-ST-only v1 Git preflight, file inventory, and module map.

This script is metadata-only. It does not run M2.5, Kmix_A, GPCCA, DARLIN, or
any production computation.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nichefate.planA_st_only.module_registry import legacy_mapping_rows, production_rows

REPORT_ROOT = ROOT / "reports" / "git_update_planA_st_only_modules"
FROZEN_BACKBONE_COMMIT = "6f921694fb81613d73b1a4ad3dfe2622b869fbba"
PROHIBITED_SSD_PATH = "/" + "ssd"


def run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode:
        raise RuntimeError(result.stdout.strip())
    return result.stdout.strip()


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


def markdown_table(rows: list[dict[str, object]], fields: list[str], limit: int | None = None) -> str:
    visible = rows[:limit] if limit else rows
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in visible:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    if limit and len(rows) > limit:
        lines.append(f"\nShowing {limit} of {len(rows)} rows. See TSV/JSON for the full table.")
    return "\n".join(lines)


def git_status_paths() -> tuple[list[str], list[str]]:
    output = run_git("status", "--porcelain=v1", "-uall")
    tracked_or_modified: list[str] = []
    untracked: list[str] = []
    for line in output.splitlines():
        if not line:
            continue
        status, path = parse_status_line(line)
        if status == "??":
            untracked.append(path)
        else:
            tracked_or_modified.append(path)
    return tracked_or_modified, untracked


def parse_status_line(line: str) -> tuple[str, str]:
    if line.startswith("?? "):
        return "??", line[3:]
    if len(line) >= 4 and line[2] == " ":
        return line[:2].strip() or "modified", line[3:].split(" -> ")[-1]
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1].split(" -> ")[-1]
    return line.strip(), ""


def file_size(path: str) -> int:
    target = ROOT / path
    return target.stat().st_size if target.is_file() else 0


def file_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if path.startswith("src/"):
        return "source"
    if path.startswith("scripts/"):
        return "script"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("docs/") or path == "README.md":
        return "doc"
    if path.startswith("configs/"):
        return "config"
    if path.startswith("reports/"):
        if suffix in {".png", ".pdf", ".svg"}:
            return "generated"
        if suffix in {".parquet", ".npz", ".npy", ".h5ad", ".fastq", ".gz"}:
            return "data"
        return "report"
    return "generated"


def legacy_milestone(path: str) -> str:
    lower = path.lower()
    if "darlin" in lower or "barcode" in lower:
        return "DARLIN/barcode"
    if "branchsbm" in lower or "planb" in lower:
        return "PlanB/BranchSBM"
    for token in ["m0", "m1", "m2_5", "m2", "m3_v2", "m3", "m4a", "m4c", "m4d", "m4e"]:
        if token in lower:
            return token.replace("_", ".").upper()
    if "kmix" in lower or "sparse_kernel" in lower:
        return "M4A / Kmix_A"
    if "gpcca" in lower:
        return "K_gpcca"
    if "absorption" in lower or "fate" in lower:
        return "M4C absorption"
    if "macrostate" in lower:
        return "M4E"
    if "visual" in lower or "figure" in lower:
        return "Visualization"
    if "plana" in lower:
        return "PlanA"
    return ""


def production_category(path: str) -> str:
    lower = path.lower()
    if "darlin" in lower or "barcode" in lower:
        return "DARLIN/barcode integration, to exclude from this ST-only freeze"
    if any(x in lower for x in ["branchsbm", "planb"]):
        return "Legacy or experimental branches"
    if any(x in lower for x in ["h5ad", "fastq", ".parquet", ".npz", ".npy", "scratch"]):
        return "Raw/data/scratch outputs, to exclude"
    if "full_kmix_a" in lower or "kmix" in lower or "sparse_kernel" in lower:
        return "Corrected feature-only Kmix_A / kernel assembly"
    if "gpcca" in lower:
        return "GPCCA macrostate inference"
    if "macrostate_annotation" in lower or "biological_annotation" in lower:
        return "Macrostate annotation"
    if "source_terminal" in lower:
        return "Source/terminal role inference"
    if "cellrank_aligned" in lower:
        return "CellRank-aligned terminal audit"
    if "absorption" in lower or "fate_probability" in lower:
        return "Absorption / fate probability"
    if "visual" in lower or "figure" in lower or "result_package" in lower:
        return "Visualization / final result package"
    if any(x in lower for x in ["m0", "m1", "m2", "metaniche", "niche_encoder", "niche_builder"]):
        return "Stable M0-M2.5 backbone"
    return "Legacy or experimental branches"


def safe_for_github(path: str, size: int) -> tuple[str, str]:
    lower = path.lower()
    blocked = [".h5ad", ".fastq", ".fq", ".bam", ".parquet", ".npz", ".npy", ".pkl"]
    if any(lower.endswith(ext) for ext in blocked):
        return "no", "blocked data/matrix format"
    if "/figures/" in lower and lower.endswith((".png", ".pdf")):
        return "no", "figure binary indexed only for this update"
    if "darlin" in lower or "barcode" in lower:
        return "no", "DARLIN/barcode evidence excluded from ST-only freeze"
    if "scratch" in lower or lower.startswith(("data/", "raw/", "external/")):
        return "no", "raw/data/scratch path excluded"
    if size > 1_000_000 and path.startswith("reports/"):
        return "no", "large generated report/table indexed only"
    return "yes", "small source, test, doc, config, or curated report"


def include_decision(path: str, kind: str, safe: str) -> tuple[str, str]:
    lower = path.lower()
    if safe != "yes":
        return "no", "not safe for GitHub in this ST-only commit"
    if approved_staging_path(path):
        return "yes", "approved PlanA-ST-only module reorg file"
    if lower.startswith(("src/nichefate/plana_k/", "scripts/plana_k_", "tests/test_plana_k")):
        return "no", "legacy PlanA-K provenance indexed only in this reorg commit"
    if lower.startswith(("docs/", "reports/plana")):
        return "no", "PlanA or DARLIN provenance indexed only unless explicitly approved"
    return "no", "outside approved staging scope"


def approved_staging_path(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    allowed_report_suffixes = {".md", ".json", ".tsv"}
    exact = {
        "README.md",
        "docs/pipeline_module_index.md",
        "docs/planA_st_only_v1_production_modules.md",
        "scripts/planA_st_only_00_module_inventory.py",
        "scripts/planA_st_only_01_validate_frozen_outputs.py",
        "scripts/planA_st_only_02_build_result_index.py",
        "tests/test_planA_st_only_facades.py",
    }
    if path in exact:
        return True
    if path.startswith("src/nichefate/planA_st_only/") and suffix == ".py":
        return True
    if path.startswith("reports/git_update_planA_st_only_modules/") and suffix in allowed_report_suffixes:
        return True
    if path.startswith("reports/planA_st_only_v1_index/") and suffix in allowed_report_suffixes:
        return True
    return False


def collect_inventory() -> list[dict[str, object]]:
    tracked = set(run_git("ls-files").splitlines())
    _, untracked = git_status_paths()
    paths = tracked | set(untracked)
    selected: list[str] = []
    prefixes = (
        "src/nichefate/planA_k/",
        "src/nichefate/planA_st_only/",
        "scripts/planA",
        "scripts/m0_",
        "scripts/m1_",
        "scripts/m2_",
        "scripts/m3",
        "scripts/m4",
        "scripts/k_gpcca",
        "docs/planA",
        "docs/pipeline_module_index.md",
        "README.md",
        "docs/darlin",
        "reports/planA",
        "reports/git_update_planA_st_only_modules/",
        "reports/darlin",
        "tests/test_planA",
    )
    for path in paths:
        if path.startswith(prefixes):
            selected.append(path)
    rows: list[dict[str, object]] = []
    for path in sorted(selected):
        size = file_size(path)
        kind = file_type(path)
        safe, safe_reason = safe_for_github(path, size)
        include, reason = include_decision(path, kind, safe)
        rows.append(
            {
                "path": path,
                "file_type": kind,
                "legacy_milestone": legacy_milestone(path),
                "production_module_category": production_category(path),
                "include_in_commit": include,
                "reason": reason if include == "yes" else safe_reason,
                "file_size": size,
                "safe_for_github": safe,
            }
        )
    return rows


def write_preflight() -> None:
    tracked_or_modified, untracked = git_status_paths()
    large_untracked = [
        {"path": path, "file_size": file_size(path)}
        for path in untracked
        if file_size(path) > 1_000_000
    ]
    gitignore = ROOT / ".gitignore"
    main_contains = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_BACKBONE_COMMIT, "origin/main"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    payload = {
        "hostname": platform.node(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "pwd": str(ROOT),
        "current_branch": run_git("branch", "--show-current"),
        "git_status_short": run_git("status", "--short", "--branch"),
        "git_remote_v": run_git("remote", "-v"),
        "latest_commit_hash": run_git("rev-parse", "HEAD"),
        "origin_main_includes_frozen_backbone_commit": main_contains,
        "frozen_backbone_commit": FROZEN_BACKBONE_COMMIT,
        "untracked_file_count": len(untracked),
        "changed_tracked_file_count": len(tracked_or_modified),
        "untracked_file_summary": dict(Counter(Path(path).parts[0] for path in untracked)),
        "large_untracked_files": sorted(large_untracked, key=lambda row: row["file_size"], reverse=True),
        "gitignore_status": {
            "exists": gitignore.exists(),
            "size": gitignore.stat().st_size if gitignore.exists() else 0,
            "has_h5ad_rule": "*.h5ad" in gitignore.read_text(encoding="utf-8"),
            "has_fastq_rule": "*.fastq" in gitignore.read_text(encoding="utf-8"),
            "has_npz_rule": "*.npz" in gitignore.read_text(encoding="utf-8"),
            "has_parquet_rule": "*.parquet" in gitignore.read_text(encoding="utf-8"),
        },
    }
    write_json(REPORT_ROOT / "00_GIT_PREFLIGHT.json", payload)
    md = [
        "# Git Preflight",
        "",
        f"- Hostname: `{payload['hostname']}`",
        f"- Date UTC: `{payload['date_utc']}`",
        f"- PWD: `{payload['pwd']}`",
        f"- Branch: `{payload['current_branch']}`",
        f"- HEAD: `{payload['latest_commit_hash']}`",
        f"- origin/main contains frozen backbone commit: `{main_contains}`",
        f"- Untracked files: `{len(untracked)}`",
        f"- Changed tracked files: `{len(tracked_or_modified)}`",
        "",
        "## Git Status",
        "",
        "```text",
        str(payload["git_status_short"]),
        "```",
        "",
        "## Remotes",
        "",
        "```text",
        str(payload["git_remote_v"]),
        "```",
        "",
        "## Large Untracked Files",
        "",
        markdown_table(payload["large_untracked_files"], ["path", "file_size"], limit=80)
        if large_untracked
        else "No untracked files larger than 1 MB were detected.",
    ]
    (REPORT_ROOT / "00_GIT_PREFLIGHT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_inventory_and_map() -> None:
    rows = collect_inventory()
    fields = [
        "path",
        "file_type",
        "legacy_milestone",
        "production_module_category",
        "include_in_commit",
        "reason",
        "file_size",
        "safe_for_github",
    ]
    write_tsv(REPORT_ROOT / "01_PLAN_A_FILE_INVENTORY.tsv", rows, fields)
    write_json(REPORT_ROOT / "01_PLAN_A_FILE_INVENTORY.json", rows)
    counts = Counter(row["production_module_category"] for row in rows)
    md = ["# PlanA File Inventory", "", "## Category Counts", ""]
    md.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    md.extend(["", "## Inventory Preview", "", markdown_table(rows, fields, limit=120)])
    (REPORT_ROOT / "01_PLAN_A_FILE_INVENTORY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    map_rows = legacy_mapping_rows()
    map_fields = ["legacy_milestone", "production_module", "status_or_note"]
    write_tsv(REPORT_ROOT / "02_LEGACY_TO_PRODUCTION_MODULE_MAP.tsv", map_rows, map_fields)
    write_json(
        REPORT_ROOT / "02_LEGACY_TO_PRODUCTION_MODULE_MAP.json",
        {"legacy_to_production": map_rows, "production_pipeline": production_rows()},
    )
    md = ["# Legacy To Production Module Map", "", markdown_table(map_rows, map_fields)]
    (REPORT_ROOT / "02_LEGACY_TO_PRODUCTION_MODULE_MAP.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def status_path_rows() -> list[dict[str, str]]:
    output = run_git("status", "--porcelain=v1", "-uall")
    rows = []
    for line in output.splitlines():
        if not line:
            continue
        status, path = parse_status_line(line)
        rows.append({"git_status": status, "path": path})
    return rows


def write_proposed_staging_list() -> None:
    rows = []
    for row in status_path_rows():
        path = row["path"]
        size = file_size(path)
        safe, reason = safe_for_github(path, size)
        approved = approved_staging_path(path)
        if approved and safe == "yes":
            decision = "stage"
            detail = "approved reorg source, script, doc, test, or small report"
        elif approved:
            decision = "do_not_stage"
            detail = reason
        else:
            decision = "do_not_stage"
            detail = "outside approved PlanA-ST-only module reorg staging scope"
        rows.append(
            {
                "path": path,
                "git_status": row["git_status"],
                "proposed_action": decision,
                "reason": detail,
                "file_size": size,
            }
        )
    fields = ["path", "git_status", "proposed_action", "reason", "file_size"]
    stage_rows = [row for row in rows if row["proposed_action"] == "stage"]
    write_tsv(REPORT_ROOT / "04_PROPOSED_STAGING_LIST.tsv", stage_rows, fields)
    write_json(REPORT_ROOT / "04_PROPOSED_STAGING_LIST.json", rows)
    md = [
        "# Proposed Staging List",
        "",
        "Only approved PlanA-ST-only reorg source, scripts, docs, tests, and small reports are proposed.",
        "",
        markdown_table(stage_rows, fields),
    ]
    (REPORT_ROOT / "04_PROPOSED_STAGING_LIST.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def staged_files() -> list[str]:
    return [line for line in run_git("diff", "--cached", "--name-only").splitlines() if line]


def text_file_contains(path: str, needle: str) -> bool:
    target = ROOT / path
    if not target.is_file():
        return False
    if target.suffix.lower() not in {".py", ".md", ".json", ".tsv", ".txt", ".yaml", ".yml"}:
        return False
    try:
        return needle in target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def write_staging_audit() -> bool:
    files = staged_files()
    blocked_suffixes = (".h5ad", ".fastq", ".fq", ".parquet", ".npz", ".npy", ".bam", ".cram")
    checks = [
        {
            "check": "only_approved_paths_staged",
            "status": all(approved_staging_path(path) for path in files),
            "detail": "all staged paths are within the approved reorg scope",
        },
        {
            "check": "no_raw_data_staged",
            "status": not any(path.startswith(("data/", "raw/", "external/")) for path in files),
            "detail": "raw/external data roots are absent from the index",
        },
        {
            "check": "no_h5ad_fastq_staged",
            "status": not any(path.lower().endswith((".h5ad", ".fastq", ".fastq.gz", ".fq", ".fq.gz")) for path in files),
            "detail": "h5ad and FASTQ files are absent from the index",
        },
        {
            "check": "no_production_matrix_tables_staged",
            "status": not any(path.lower().endswith(blocked_suffixes) for path in files),
            "detail": "parquet, npz, npy, and large matrix formats are absent from the index",
        },
        {
            "check": "no_scratch_outputs_staged",
            "status": not any("scratch" in path.lower() for path in files),
            "detail": "scratch outputs are absent from staged paths",
        },
        {
            "check": "no_darlin_evidence_staged",
            "status": not any(("darlin" in path.lower() or "barcode" in path.lower()) and path.startswith("reports/") for path in files),
            "detail": "DARLIN/barcode evidence reports are absent from the index",
        },
        {
            "check": "no_figure_binaries_staged",
            "status": not any("/figures/" in path and path.lower().endswith((".png", ".pdf")) for path in files),
            "detail": "figure binaries are absent from the index",
        },
        {
            "check": "no_ssd_paths_in_staged_text",
            "status": not any(
                PROHIBITED_SSD_PATH in path
                or text_file_contains(path, PROHIBITED_SSD_PATH)
                for path in files
            ),
            "detail": "staged text files and staged paths contain no SSD-root references",
        },
    ]
    decision = all(row["status"] for row in checks)
    payload = {
        "decision": "PASS" if decision else "FAIL",
        "staged_file_count": len(files),
        "staged_files": files,
        "checks": [
            {**row, "status": "pass" if row["status"] else "fail"}
            for row in checks
        ],
    }
    write_json(REPORT_ROOT / "05_STAGING_AUDIT.json", payload)
    lines = ["# Staging Audit", "", f"Decision: `{payload['decision']}`", "", "## Checks", ""]
    lines.extend(f"- `{row['status']}` {row['check']}: {row['detail']}" for row in payload["checks"])
    lines.extend(["", "## Staged Files", ""])
    lines.extend(f"- `{path}`" for path in files)
    (REPORT_ROOT / "05_STAGING_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def read_report_decision(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "INVALID_JSON"
    return str(payload.get("decision", "UNKNOWN"))


def write_summary(staging_audit_passed: bool) -> None:
    validation_decision = read_report_decision(REPORT_ROOT / "03_VALIDATION.json")
    preflight = json.loads((REPORT_ROOT / "00_GIT_PREFLIGHT.json").read_text(encoding="utf-8"))
    backbone_ok = bool(preflight.get("origin_main_includes_frozen_backbone_commit"))
    final_ready = backbone_ok and staging_audit_passed and validation_decision == "PASS"
    payload = {
        "decision": "PLAN_A_ST_ONLY_V1_MODULE_REORG_READY_FOR_REVIEW" if final_ready else "PLAN_A_ST_ONLY_V1_MODULE_REORG_PENDING_VALIDATION",
        "origin_main_includes_frozen_backbone_commit": backbone_ok,
        "frozen_backbone_commit": FROZEN_BACKBONE_COMMIT,
        "validation_decision": validation_decision,
        "staging_audit_decision": "PASS" if staging_audit_passed else "FAIL",
        "claim_boundary": "ST-only / barcode-free; DARLIN/barcode validation is future work.",
    }
    write_json(REPORT_ROOT / "00_PLAN_A_ST_ONLY_MODULE_REORG_SUMMARY.json", payload)
    md = [
        "# PlanA-ST-only Module Reorg Summary",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        f"- origin/main contains frozen backbone commit: `{backbone_ok}`",
        f"- Frozen backbone commit: `{FROZEN_BACKBONE_COMMIT}`",
        f"- Validation decision: `{validation_decision}`",
        f"- Staging audit decision: `{payload['staging_audit_decision']}`",
        "- Claim boundary: ST-only / barcode-free; DARLIN/barcode validation is future work.",
    ]
    (REPORT_ROOT / "00_PLAN_A_ST_ONLY_MODULE_REORG_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_preflight()
    write_inventory_and_map()
    write_proposed_staging_list()
    staging_audit_passed = write_staging_audit()
    write_summary(staging_audit_passed)
    print(f"Wrote Git preflight and PlanA inventory under {REPORT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
