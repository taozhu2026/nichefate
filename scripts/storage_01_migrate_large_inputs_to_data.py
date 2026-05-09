#!/usr/bin/env python
"""Dry-run or execute conservative migration of approved large input paths."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.storage_ops import (  # noqa: E402
    DRY_RUN_SUMMARY_FIELDS,
    ELIGIBLE_CATEGORIES,
    LOGS_DIR,
    REPORTS_DIR,
    bool_text,
    copy_path,
    dry_run_summary_row,
    human_bytes,
    is_row_eligible,
    parse_bool,
    path_kind,
    read_csv_rows,
    sha256_file,
    tree_size_bytes,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument(
        "--category",
        default="raw,m0_input,m0_intermediate,archive",
        help="Comma-separated categories allowed for this run.",
    )
    parser.add_argument("--allow-overwrite-symlink", default="false")
    parser.add_argument("--sha256-threshold-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR)
    return parser.parse_args()


def setup_logger(logs_dir: Path) -> tuple[logging.Logger, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"storage_migration_{timestamp}.log"
    logger = logging.getLogger("nichefate.storage_migration")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    return logger, log_path


def selected_rows(
    rows: list[dict[str, str]],
    allowed_categories: set[str],
    max_paths: int | None,
) -> list[dict[str, str]]:
    eligible = [row for row in rows if is_row_eligible(row, allowed_categories)]
    if max_paths is not None:
        return eligible[:max_paths]
    return eligible


def verify_copy(source: Path, target: Path, sha256_threshold: int) -> list[str]:
    errors: list[str] = []
    if not os.path.lexists(source):
        errors.append(f"source missing: {source}")
    if not os.path.lexists(target):
        errors.append(f"target missing: {target}")
    if errors:
        return errors
    source_size = tree_size_bytes(source)
    target_size = tree_size_bytes(target)
    if source_size != target_size:
        errors.append(f"size mismatch: source={source_size} target={target_size}")
    if source.is_file() and target.is_file() and source_size <= sha256_threshold:
        if sha256_file(source) != sha256_file(target):
            errors.append("sha256 mismatch")
    return errors


def target_needs_copy(target: Path) -> bool:
    if not os.path.lexists(target):
        return True
    if target.is_dir() and not target.is_symlink():
        try:
            next(target.iterdir())
        except StopIteration:
            return True
    return False


def replace_with_symlink(
    source: Path,
    target: Path,
    allow_overwrite_symlink: bool,
    logger: logging.Logger,
) -> None:
    if source.is_symlink():
        current = source.resolve(strict=False)
        if current == target.resolve(strict=False):
            logger.info("Source already symlinks to target: %s", source)
            return
        if not allow_overwrite_symlink:
            raise RuntimeError(f"Refusing to overwrite existing symlink: {source}")
        source.unlink()
        source.symlink_to(target, target_is_directory=target.is_dir())
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = source.with_name(f"{source.name}.pre_symlink_backup_{timestamp}")
    source.rename(backup)
    try:
        source.symlink_to(target, target_is_directory=target.is_dir())
        if not source.exists():
            raise RuntimeError(f"Created symlink does not resolve: {source}")
    except Exception:
        if os.path.lexists(source):
            source.unlink()
        backup.rename(source)
        raise
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink()


def execute_row(
    row: dict[str, str],
    allow_overwrite_symlink: bool,
    sha256_threshold: int,
    logger: logging.Logger,
) -> dict[str, object]:
    source = Path(row["old_path"])
    target = Path(row["new_path"])
    result = dry_run_summary_row(row)
    result["operation_status"] = "started"
    if not os.path.lexists(source):
        result["operation_status"] = "source_missing"
        return result
    if source.is_symlink() and not allow_overwrite_symlink:
        resolved = source.resolve(strict=False)
        if resolved != target.resolve(strict=False):
            result["operation_status"] = "skipped_existing_symlink"
            return result
    if target_needs_copy(target):
        logger.info("Copying %s -> %s", source, target)
        copy_path(source, target)
    errors = verify_copy(source, target, sha256_threshold)
    if errors:
        result["operation_status"] = "verification_failed"
        result["notes"] = "; ".join(errors)
        return result
    replace_with_symlink(source, target, allow_overwrite_symlink, logger)
    result["operation_status"] = "moved_and_symlinked"
    result["source_path_type_after"] = path_kind(source)
    return result


def write_markdown_summary(
    path: Path,
    rows: list[dict[str, object]],
    *,
    dry_run: bool,
    log_path: Path,
) -> None:
    total = sum(int(row.get("size", 0) or 0) for row in rows)
    mode = "dry-run" if dry_run else "execute"
    lines = [
        "# Storage Migration Summary",
        "",
        f"- Mode: {mode}",
        f"- Planned rows: {len(rows)}",
        f"- Planned size: {human_bytes(total)}",
        f"- Log: `{log_path}`",
        "",
        "## Planned Paths",
    ]
    for row in rows:
        lines.append(
            "- "
            f"`{row['planned_old_path']}` -> `{row['planned_new_path']}` "
            f"size={row['size']} category={row['category']} "
            f"risk={row['downstream_risk']} symlink={row['symlink_will_be_created']} "
            f"source_on_data={row['source_already_exists_on_data']} "
            f"target_exists={row['target_already_exists']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.dry_run and args.execute:
        raise SystemExit("--dry-run and --execute are mutually exclusive")
    execute = bool(args.execute)
    dry_run = not execute
    allowed_categories = {
        item.strip() for item in args.category.split(",") if item.strip()
    }
    disallowed = allowed_categories - ELIGIBLE_CATEGORIES
    if disallowed:
        raise SystemExit(f"Unsupported migration categories: {sorted(disallowed)}")
    allow_overwrite = parse_bool(args.allow_overwrite_symlink)
    logger, log_path = setup_logger(args.logs_dir)
    rows = read_csv_rows(args.manifest)
    planned = selected_rows(rows, allowed_categories, args.max_paths)
    logger.info("Mode: %s", "execute" if execute else "dry-run")
    logger.info("Selected planned rows: %s", len(planned))
    summaries: list[dict[str, object]] = []
    if dry_run:
        summaries = [dry_run_summary_row(row) for row in planned]
    else:
        for row in planned:
            summaries.append(
                execute_row(
                    row,
                    allow_overwrite,
                    args.sha256_threshold_bytes,
                    logger,
                )
            )
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.reports_dir / "storage_migration_summary.csv"
    summary_md = args.reports_dir / "storage_migration_summary.md"
    fields = list(DRY_RUN_SUMMARY_FIELDS)
    extra_fields = sorted({key for row in summaries for key in row} - set(fields))
    write_csv_rows(summary_csv, summaries, fields + extra_fields)
    write_markdown_summary(summary_md, summaries, dry_run=dry_run, log_path=log_path)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {log_path}")
    if dry_run:
        print("Dry-run only. No files were copied, removed, or symlinked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
