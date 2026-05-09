import importlib.util
from pathlib import Path

import pytest

from nichefate.storage_ops import (
    dry_run_summary_row,
    is_row_eligible,
    parse_bool,
    tree_size_bytes,
)


def load_script(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def planned_row(tmp_path: Path, category: str = "m0_intermediate") -> dict[str, str]:
    old_path = tmp_path / "old"
    old_path.mkdir()
    (old_path / "data.txt").write_text("abc", encoding="utf-8")
    return {
        "old_path": str(old_path),
        "new_path": str(tmp_path / "data" / "old"),
        "path_type": "dir",
        "category": category,
        "size_bytes": str(tree_size_bytes(old_path)),
        "symlink_required": "true",
        "move_status": "planned",
        "reason": "test",
        "downstream_risk": "low",
        "notes": "",
    }


def test_parse_bool_accepts_explicit_values() -> None:
    assert parse_bool("true") is True
    assert parse_bool("false") is False
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_is_row_eligible_requires_planned_low_risk_allowed_category(tmp_path: Path) -> None:
    row = planned_row(tmp_path)

    assert is_row_eligible(row, {"m0_intermediate"}) is True

    row["move_status"] = "review_required"
    assert is_row_eligible(row, {"m0_intermediate"}) is False

    row["move_status"] = "planned"
    row["downstream_risk"] = "high"
    assert is_row_eligible(row, {"m0_intermediate"}) is False

    row["downstream_risk"] = "low"
    row["category"] = "do_not_move"
    assert is_row_eligible(row, {"do_not_move"}) is False


def test_dry_run_summary_reports_required_fields(tmp_path: Path) -> None:
    row = planned_row(tmp_path, category="archive")
    summary = dry_run_summary_row(row)

    assert summary == {
        "planned_old_path": row["old_path"],
        "planned_new_path": row["new_path"],
        "size": row["size_bytes"],
        "category": "archive",
        "downstream_risk": "low",
        "symlink_will_be_created": "true",
        "source_already_exists_on_data": "false",
        "target_already_exists": "false",
    }


def test_selected_rows_excludes_active_and_review_required(tmp_path: Path) -> None:
    module = load_script("storage_01_migrate_large_inputs_to_data")
    row = planned_row(tmp_path)
    active = {**row, "category": "do_not_move"}
    review = {**row, "move_status": "review_required"}

    selected = module.selected_rows(
        [row, active, review],
        {"raw", "m0_input", "m0_intermediate", "archive"},
        max_paths=None,
    )

    assert selected == [row]


def test_lightweight_read_check_reads_text_and_json(tmp_path: Path) -> None:
    module = load_script("storage_02_validate_path_compatibility")
    csv_path = tmp_path / "table.csv"
    json_path = tmp_path / "payload.json"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    json_path.write_text('{"ok": true}', encoding="utf-8")

    assert module.lightweight_read_check(csv_path).startswith("text_table_head_ok")
    assert module.lightweight_read_check(json_path) == "json_ok:dict"


def test_hard_coded_path_classification_detects_symlink(tmp_path: Path) -> None:
    module = load_script("storage_02_validate_path_compatibility")
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    assert module.classify_hard_coded_path(link, []) == "OK_via_symlink"


def test_audit_manifest_marks_m4d_do_not_move(tmp_path: Path) -> None:
    module = load_script("storage_00_audit_nichefate_paths")
    scratch = tmp_path / "scratch" / "nichefate"
    data = tmp_path / "data" / "nichefate"
    (scratch / "m4d").mkdir(parents=True)

    rows = module.build_manifest(scratch, data, [])

    assert rows[0]["old_path"].endswith("/m4d")
    assert rows[0]["category"] == "do_not_move"
    assert rows[0]["move_status"] == "skipped"
