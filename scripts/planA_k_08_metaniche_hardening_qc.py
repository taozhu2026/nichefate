#!/usr/bin/env python
"""Write rare-state audit, M2.5 state contract v2, and final hardening summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.planA_k_gpcca import (
    DOC_ROOT,
    METANICHE_HARDENING_ROOT,
    PILOT_OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    ensure_dir,
    git_status_short,
    hardening_final_summary_markdown,
    hardening_final_summary_payload,
    m2_5_state_contract_v2,
    rare_state_audit_markdown,
    rare_state_preservation_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=METANICHE_HARDENING_ROOT)
    parser.add_argument("--doc-root", type=Path, default=DOC_ROOT)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    doc_root = ensure_dir(args.doc_root)

    rare_frame, rare_payload = rare_state_preservation_audit(pilot_root=args.pilot_root)
    atomic_write_text(
        output_dir / "05_rare_state_preservation_audit.md",
        rare_state_audit_markdown(rare_frame, rare_payload),
        overwrite=args.overwrite,
    )
    atomic_write_tsv(output_dir / "05_rare_state_preservation_audit.tsv", rare_frame, overwrite=args.overwrite)
    atomic_write_json(output_dir / "05_rare_state_preservation_audit.json", rare_payload, overwrite=args.overwrite)

    contract_text, contract_frame = m2_5_state_contract_v2()
    atomic_write_text(doc_root / "06_m2_5_state_contract_v2.md", contract_text, overwrite=args.overwrite)
    atomic_write_text(output_dir / "08_m2_5_state_contract_v2.md", contract_text, overwrite=args.overwrite)
    atomic_write_tsv(output_dir / "08_m2_5_state_contract_v2.tsv", contract_frame, overwrite=args.overwrite)

    coord_payload = read_json_if_present(output_dir / "03_coordinate_join_preview.json")
    spatial_payload = read_json_if_present(output_dir / "04_spatial_compactness_qc.json")
    strat_payload = read_json_if_present(output_dir / "07_stratified_pilot_summary.json")
    summary_payload = hardening_final_summary_payload(
        output_dir=output_dir,
        coord_payload=coord_payload,
        spatial_payload=spatial_payload,
        rare_payload=rare_payload,
        strat_payload=strat_payload,
        git_status_after=git_status_short(),
    )
    atomic_write_text(
        output_dir / "00_METANICHE_HARDENING_SUMMARY.md",
        hardening_final_summary_markdown(summary_payload),
        overwrite=args.overwrite,
    )
    atomic_write_json(
        output_dir / "00_METANICHE_HARDENING_SUMMARY.json",
        summary_payload,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
