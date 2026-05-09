#!/usr/bin/env python
"""Audit nichefate storage paths and propose a conservative migration manifest."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.storage_ops import (  # noqa: E402
    DATA_ROOT,
    LEGACY_DATA_ROOTS,
    MANIFEST_FIELDS,
    PROJECT_ROOT as DEFAULT_PROJECT_ROOT,
    SCRATCH_ROOT,
    STORAGE_ROOT,
    bool_text,
    ensure_data_root,
    ensure_report_dirs,
    human_bytes,
    path_kind,
    tree_size_bytes,
    write_csv_rows,
)


AUDIT_FIELDS = ["section", "path", "size_bytes", "size_human", "notes"]
INVENTORY_FIELDS = [
    "root",
    "path",
    "relative_path",
    "path_type",
    "exists",
    "is_symlink",
    "symlink_target",
    "symlink_resolved",
    "broken_symlink",
    "size_bytes",
    "stage",
    "category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, default=SCRATCH_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--storage-root", type=Path, default=STORAGE_ROOT)
    parser.add_argument(
        "--legacy-data-root",
        action="append",
        type=Path,
        default=list(LEGACY_DATA_ROOTS),
    )
    parser.add_argument("--max-inventory-paths", type=int, default=200_000)
    parser.add_argument("--largest-count", type=int, default=30)
    parser.add_argument("--no-create-data-root", action="store_true")
    return parser.parse_args()


def stage_and_category(path: Path, scratch_root: Path, data_root: Path) -> tuple[str, str]:
    text = path.as_posix()
    if "merfish_colitis_raw" in text or "nichefate_data" in text:
        return "raw", "raw"
    try:
        rel = path.relative_to(scratch_root)
    except ValueError:
        if text.startswith(data_root.as_posix()):
            parts = path.relative_to(data_root).parts
            stage = parts[0] if parts else "data"
            return stage, "data"
        return "external", "external"
    parts = rel.parts
    if not parts:
        return "root", "scratch"
    stage = parts[0]
    if stage == "m0":
        if len(parts) > 1 and parts[1] in {"by_slice", "by_time", "processed", "graphs"}:
            return stage, "m0_intermediate"
        if len(parts) > 1 and "archive" in parts[1]:
            return stage, "archive"
        return stage, "archive"
    if stage in {"m1", "m2"}:
        return stage, "review_required"
    if stage in {"m3", "m4a", "m4b", "m4c", "m4d"}:
        return stage, "do_not_move"
    return stage, "other"


def inventory_paths(
    roots: list[Path],
    scratch_root: Path,
    data_root: Path,
    max_paths: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        if not os.path.lexists(root):
            rows.append(
                {
                    "root": str(root),
                    "path": str(root),
                    "relative_path": "",
                    "path_type": "missing",
                    "exists": "false",
                    "is_symlink": "false",
                    "symlink_target": "",
                    "symlink_resolved": "",
                    "broken_symlink": "false",
                    "size_bytes": 0,
                    "stage": "",
                    "category": "",
                }
            )
            continue
        for current_root, dirs, files in os.walk(root, followlinks=False):
            if len(rows) >= max_paths:
                return rows
            current = Path(current_root)
            entries = [current] + [current / name for name in dirs + files]
            for path in entries:
                if len(rows) >= max_paths:
                    return rows
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    rel = ""
                is_link = path.is_symlink()
                target = os.readlink(path) if is_link else ""
                resolved = path.resolve(strict=False).as_posix() if is_link else ""
                broken = is_link and not path.exists()
                stage, category = stage_and_category(path, scratch_root, data_root)
                rows.append(
                    {
                        "root": str(root),
                        "path": str(path),
                        "relative_path": rel,
                        "path_type": path_kind(path),
                        "exists": bool_text(path.exists()),
                        "is_symlink": bool_text(is_link),
                        "symlink_target": target,
                        "symlink_resolved": resolved,
                        "broken_symlink": bool_text(broken),
                        "size_bytes": tree_size_bytes(path),
                        "stage": stage,
                        "category": category,
                    }
                )
    return rows


def top_level_usage(root: Path) -> list[dict[str, object]]:
    if not os.path.lexists(root):
        return [
            {
                "section": "top_level_usage",
                "path": str(root),
                "size_bytes": 0,
                "size_human": "0 B",
                "notes": "missing",
            }
        ]
    children = sorted(root.iterdir(), key=lambda path: path.name)
    rows = []
    for path in [root] + children:
        size = tree_size_bytes(path)
        rows.append(
            {
                "section": "top_level_usage",
                "path": str(path),
                "size_bytes": size,
                "size_human": human_bytes(size),
                "notes": path_kind(path),
            }
        )
    return rows


def largest_entries(inventory: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    sortable = [row for row in inventory if row["path_type"] in {"file", "dir"}]
    sortable.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    rows = []
    for row in sortable[:count]:
        rows.append(
            {
                "section": "largest_entries",
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "size_human": human_bytes(int(row["size_bytes"])),
                "notes": row["path_type"],
            }
        )
    return rows


def duplicate_looking_dirs(inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in inventory:
        if row["path_type"] == "dir":
            by_name[Path(str(row["path"])).name].append(row)
    rows = []
    for name, matches in sorted(by_name.items()):
        roots = {Path(str(row["root"])).as_posix() for row in matches}
        if len(matches) > 1 and len(roots) > 1:
            paths = "; ".join(str(row["path"]) for row in matches[:5])
            rows.append(
                {
                    "section": "duplicate_looking_dirs",
                    "path": name,
                    "size_bytes": "",
                    "size_human": "",
                    "notes": paths,
                }
            )
    return rows


def manifest_row(
    old_path: Path,
    new_path: Path,
    category: str,
    move_status: str,
    reason: str,
    downstream_risk: str,
    symlink_required: bool,
    notes: str = "",
) -> dict[str, object]:
    size_path = old_path
    if old_path.is_symlink() and old_path.exists():
        size_path = old_path.resolve(strict=False)
    return {
        "old_path": str(old_path),
        "new_path": str(new_path),
        "path_type": path_kind(old_path),
        "category": category,
        "size_bytes": tree_size_bytes(size_path),
        "symlink_required": bool_text(symlink_required),
        "move_status": move_status,
        "reason": reason,
        "downstream_risk": downstream_risk,
        "notes": notes,
    }


def build_manifest(scratch_root: Path, data_root: Path, legacy_roots: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_link = scratch_root / "merfish_colitis_raw"
    if os.path.lexists(raw_link):
        target = raw_link.resolve(strict=False) if raw_link.is_symlink() else data_root / "raw" / "merfish_colitis_raw"
        status = "verified" if raw_link.is_symlink() and target.exists() else "planned"
        rows.append(
            manifest_row(
                raw_link,
                target,
                "raw",
                status,
                "Raw MERFISH core files exposed through compatibility path.",
                "low",
                True,
                "Already symlinked to /data." if status == "verified" else "",
            )
        )
    for legacy_root in legacy_roots:
        if os.path.lexists(legacy_root):
            target_parent = "raw" if "merfish_colitis_raw" in legacy_root.name else "external"
            rows.append(
                manifest_row(
                    legacy_root,
                    data_root / target_parent / legacy_root.name,
                    "raw",
                    "skipped",
                    "Existing /data root recorded; no relocation planned in this stage.",
                    "low",
                    False,
                    "Reference or link later to reduce root sprawl.",
                )
            )
    m0_targets = {
        "by_slice": data_root / "m0" / "by_slice",
        "by_time": data_root / "m0" / "intermediate" / "by_time",
        "processed": data_root / "m0" / "intermediate" / "processed",
        "graphs": data_root / "m0" / "intermediate" / "graphs",
    }
    for name, target in m0_targets.items():
        path = scratch_root / "m0" / name
        if os.path.lexists(path):
            size = tree_size_bytes(path)
            move_status = "planned" if size > 0 else "skipped"
            rows.append(
                manifest_row(
                    path,
                    target,
                    "m0_intermediate",
                    move_status,
                    (
                        "Large M0 intermediate; preserve old path through symlink."
                        if move_status == "planned"
                        else "Empty M0 intermediate directory; no migration needed."
                    ),
                    "low",
                    move_status == "planned",
                )
            )
    for path in sorted((scratch_root / "m0").glob("*archive*")):
        rows.append(
            manifest_row(
                path,
                data_root / "m0" / "intermediate" / path.name,
                "archive",
                "planned",
                "Archived M0 artifact; safe low-risk relocation candidate.",
                "low",
                True,
            )
        )
    for name in ("reports", "logs"):
        path = scratch_root / "m0" / name
        if os.path.lexists(path):
            rows.append(
                manifest_row(
                    path,
                    data_root / "m0" / "reports" / name,
                    "archive",
                    "skipped",
                    "Lightweight M0 reports/logs can remain on /home.",
                    "low",
                    False,
                )
            )
    for stage in ("m1", "m2"):
        path = scratch_root / stage
        if os.path.lexists(path):
            rows.append(
                manifest_row(
                    path,
                    data_root / stage / "archived_or_heavy",
                    "archive",
                    "review_required",
                    f"{stage.upper()} outputs may be consumed by downstream stages.",
                    "medium",
                    True,
                )
            )
    for stage in ("m3", "m4a", "m4b", "m4c", "m4d"):
        path = scratch_root / stage
        if os.path.lexists(path):
            rows.append(
                manifest_row(
                    path,
                    path,
                    "do_not_move",
                    "skipped",
                    f"{stage.upper()} production output is active and must stay in place.",
                    "high",
                    False,
                )
            )
    return rows


def disk_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    seen: set[str] = set()
    for path in paths:
        anchor = path if path.exists() else path.parent
        try:
            usage = shutil.disk_usage(anchor)
        except OSError as exc:
            rows.append(
                {
                    "section": "disk_free",
                    "path": str(path),
                    "size_bytes": "",
                    "size_human": "",
                    "notes": f"unavailable: {exc}",
                }
            )
            continue
        key = str(anchor)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "section": "disk_free",
                "path": str(anchor),
                "size_bytes": usage.free,
                "size_human": human_bytes(usage.free),
                "notes": f"total={human_bytes(usage.total)} used={human_bytes(usage.used)}",
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]], manifest: list[dict[str, object]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in manifest:
        counts[f"{row['category']}:{row['move_status']}"] += 1
    lines = [
        "# Nichefate Storage Audit",
        "",
        "## Summary",
        "- Audit is non-migrating: no data were moved.",
        f"- Manifest rows: {len(manifest)}",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Audit Rows"])
    for row in rows:
        lines.append(
            f"- {row['section']}: `{row['path']}` "
            f"{row['size_human']} {row['notes']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dirs = ensure_report_dirs(args.storage_root)
    if not args.no_create_data_root:
        ensure_data_root(args.data_root)
    roots = [args.scratch_root, args.data_root] + args.legacy_data_root
    inventory = inventory_paths(
        roots,
        args.scratch_root,
        args.data_root,
        max_paths=args.max_inventory_paths,
    )
    rows: list[dict[str, object]] = []
    rows.extend(disk_rows([Path("/home"), Path("/data")]))
    for root in roots:
        rows.extend(top_level_usage(root))
    rows.extend(largest_entries(inventory, args.largest_count))
    rows.extend(duplicate_looking_dirs(inventory))
    manifest = build_manifest(args.scratch_root, args.data_root, args.legacy_data_root)

    write_csv_rows(dirs["reports"] / "storage_audit.csv", rows, AUDIT_FIELDS)
    write_csv_rows(dirs["reports"] / "storage_path_inventory.csv", inventory, INVENTORY_FIELDS)
    write_csv_rows(
        dirs["manifests"] / "storage_migration_manifest.csv",
        manifest,
        MANIFEST_FIELDS,
    )
    write_markdown(dirs["reports"] / "storage_audit.md", rows, manifest)
    print(f"Wrote {dirs['reports'] / 'storage_audit.md'}")
    print(f"Wrote {dirs['reports'] / 'storage_audit.csv'}")
    print(f"Wrote {dirs['reports'] / 'storage_path_inventory.csv'}")
    print(f"Wrote {dirs['manifests'] / 'storage_migration_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
