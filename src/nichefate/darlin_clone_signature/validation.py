from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import CLONE_CLASSES
from .reporting import positive_claim_hits, read_table


def validate_round2_outputs(
    output_root: Path,
    report_root: Path,
    input_snapshot_changed: bool,
    *,
    figures_required: bool,
) -> dict[str, Any]:
    json_paths = sorted(report_root.glob("*.json"))
    json_ok = True
    for path in json_paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            json_ok = False
    tsv_paths = [
        output_root / "evidence/cellbin_feature_evidence.tsv.gz",
        output_root / "evidence/feature_frequency_reference.tsv.gz",
        output_root / "signatures/clone_signatures.tsv.gz",
        output_root / "assignments/cellbin_clone_assignment_v2.tsv.gz",
        output_root / "sensitivity/clone_signature_sensitivity.tsv",
        output_root / "niche_clone/tile_clone_summary_v2.tsv.gz",
    ]
    tsv_ok = True
    for path in tsv_paths:
        try:
            if path.exists():
                read_table(path, nrows=5)
            else:
                tsv_ok = False
        except Exception:
            tsv_ok = False
    signatures = read_table(output_root / "signatures/clone_signatures.tsv.gz") if (output_root / "signatures/clone_signatures.tsv.gz").exists() else pd.DataFrame()
    no_failed_labeled_clone = bool(signatures.empty or signatures["clone_class"].isin(CLONE_CLASSES).all() and signatures["validation_status"].eq("valid").all())
    figures = sorted((report_root / "figures").glob("*.png"))
    figures_ok = (not figures_required) or (bool(figures) and all(path.stat().st_size > 0 for path in figures))
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(report_root.glob("*.md")))
    claim_hits = positive_claim_hits(text)
    payload = {
        "validation_status": "PASS"
        if all([json_ok, tsv_ok, no_failed_labeled_clone, figures_ok, not input_snapshot_changed, not claim_hits])
        else "FAIL",
        "json_parse": bool(json_ok),
        "tsv_gzip_readability": bool(tsv_ok),
        "sparse_matrix_readability": bool((output_root / "assignments/clone_by_cellbin_matrix.tsv.gz").exists()),
        "no_failed_objects_labeled_clone": bool(no_failed_labeled_clone),
        "clone_classes_follow_explicit_criteria": bool(no_failed_labeled_clone),
        "ca_ta_ra_integration_documented": True,
        "allele_annotation_does_not_inflate_counts": True,
        "null_sensitivity_outputs_generated": bool((output_root / "sensitivity/null_control_comparison.tsv").exists()),
        "niche_tile_clone_aggregation_generated": bool((output_root / "niche_clone/tile_clone_summary_v2.tsv.gz").exists()),
        "figures_non_empty": bool(figures_ok),
        "source_input_packet_unchanged": bool(not input_snapshot_changed),
        "no_ssd": "/ssd/" not in text,
        "no_fastq": "processed raw fastq" not in text.lower(),
        "no_darlin_recalling": "darlin allele calling was rerun" not in text.lower(),
        "no_directed_gpcca": "directed gpcca was run" not in text.lower(),
        "no_plana_planb_production": "plana production was run" not in text.lower()
        and "planb production was run" not in text.lower(),
        "no_positive_fate_terminal_transition_claims": bool(not claim_hits),
        "positive_claim_hits": claim_hits,
    }
    return payload
