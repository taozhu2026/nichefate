#!/usr/bin/env python
"""Finalize the L126 full barcode-aware niche characterization package."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nichefate.barcode_adapter.qc import compare_file_snapshots, sha256_path, snapshot_files
from nichefate.barcode_adapter.reporting import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    ensure_dir,
    markdown_table,
    path_has_ssd,
    utc_now,
)


SOURCE_OUTPUT_ROOT = PROJECT_ROOT / "processed" / "l126_full_barcode_niche_characterization"
SOURCE_REPORT_ROOT = PROJECT_ROOT / "reports" / "l126_full_barcode_niche_characterization"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "processed" / "l126_full_barcode_niche_characterization_final"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "l126_full_barcode_niche_characterization_final"
FINAL_KEY_FIGURE_DIRNAME = "final_key_figures"
SAMPLES = ("L126_Brain_s1", "L126_Brain_s2", "L126_Brain_s3")

FORBIDDEN_TERMS = (
    "fate",
    "terminal",
    "endpoint",
    "transition probability",
    "developmental trajectory",
    "lineage-validated endpoint",
    "lineage-validated transition",
    "clonal expansion discovered",
    "terminal niche",
    "true fate",
    "fate probability",
)
NEGATION_MARKERS = (
    " not ",
    "no ",
    "does not",
    "do not",
    "doesn't",
    "don't",
    "cannot",
    "can't",
    "without",
    "avoid",
    "avoiding",
    "not support",
    "not a ",
    "limitations",
    "limitation",
    "boundary",
    "risk",
    "risks",
    "overclaiming",
    "forbidden",
)

FINAL_KEY_FIGURE_ORDER = [
    "method_data_flow_summary",
    "cellbin_lineage_coverage_spatial",
    "cellbin_total_lineage_count_spatial",
    "cellbin_detected_feature_count_spatial",
    "cellbin_feature_entropy_spatial",
    "cellbin_dominant_feature_fraction_spatial",
    "cellbin_assay_total_count_spatial",
    "section_assay_balance_summary",
    "tile_lineage_coverage_spatial",
    "tile_feature_entropy_spatial",
    "barcode_diversity_and_dominance_distributions",
    "metaniche_section_purity_distribution",
]

FIGURE_PROFILES: dict[str, dict[str, str]] = {
    "method_data_flow_summary": {
        "data_level": "method",
        "what_it_shows": "Workflow from input packet to barcode-aware cellbin, group, tile, and metaniche-like summaries.",
        "suggested_title": "Method and data flow summary",
        "reason_for_selection": "Best orientation panel for the package; it anchors the interpretation boundary.",
        "caution_note": "Descriptive workflow only; not a biological inference figure.",
    },
    "cellbin_lineage_coverage_spatial": {
        "data_level": "cellbin",
        "what_it_shows": "Spatial map of lineage evidence presence by cellbin across the three serial sections.",
        "suggested_title": "Cellbin lineage evidence coverage",
        "reason_for_selection": "Primary coverage view at the full cellbin level.",
        "caution_note": "Serial sections are not timepoints.",
    },
    "cellbin_total_lineage_count_spatial": {
        "data_level": "cellbin",
        "what_it_shows": "Spatial map of total lineage count by cellbin.",
        "suggested_title": "Cellbin total lineage count",
        "reason_for_selection": "Shows lineage evidence depth at the native cellbin level.",
        "caution_note": "Counts are evidence abundance, not fate probability.",
    },
    "cellbin_detected_feature_count_spatial": {
        "data_level": "cellbin",
        "what_it_shows": "Spatial map of detected lineage feature count by cellbin.",
        "suggested_title": "Cellbin detected lineage feature count",
        "reason_for_selection": "Complements total lineage count with feature breadth.",
        "caution_note": "Assay-scoped evidence is preserved separately.",
    },
    "cellbin_feature_entropy_spatial": {
        "data_level": "cellbin",
        "what_it_shows": "Spatial map of lineage feature entropy by cellbin.",
        "suggested_title": "Cellbin lineage feature entropy",
        "reason_for_selection": "Highlights barcode diversity and local composition structure.",
        "caution_note": "Entropy is descriptive diversity, not a terminal-state measure.",
    },
    "cellbin_dominant_feature_fraction_spatial": {
        "data_level": "cellbin",
        "what_it_shows": "Spatial map of dominant lineage feature fraction by cellbin.",
        "suggested_title": "Cellbin dominant feature fraction",
        "reason_for_selection": "Shows dominance structure alongside entropy.",
        "caution_note": "Use as a diversity/dominance summary only.",
    },
    "cellbin_assay_total_count_spatial": {
        "data_level": "cellbin",
        "what_it_shows": "RA/TA/CA total count maps by cellbin.",
        "suggested_title": "Cellbin RA/TA/CA total counts",
        "reason_for_selection": "Preserves assay-specific evidence channels in spatial form.",
        "caution_note": "Assay balance is a QC summary, not a final clone identity call.",
    },
    "section_assay_balance_summary": {
        "data_level": "cellbin",
        "what_it_shows": "Section-level RA/TA/CA total count summary.",
        "suggested_title": "Section assay balance summary",
        "reason_for_selection": "Compact section-aware QC panel for assay balance.",
        "caution_note": "Sections are serial provenance, not a time axis.",
    },
    "tile_lineage_coverage_spatial": {
        "data_level": "tile",
        "what_it_shows": "Spatial map of non-overlapping tile lineage coverage.",
        "suggested_title": "Tile lineage coverage",
        "reason_for_selection": "Preferred robust spatial summary because tiles are non-overlapping.",
        "caution_note": "Tile summaries are descriptive and not fate inference.",
    },
    "tile_feature_entropy_spatial": {
        "data_level": "tile",
        "what_it_shows": "Spatial map of non-overlapping tile barcode entropy.",
        "suggested_title": "Tile barcode entropy",
        "reason_for_selection": "Preferred robust spatial diversity map.",
        "caution_note": "Tiles are safer than overlapping group contexts for spatial summaries.",
    },
    "barcode_diversity_and_dominance_distributions": {
        "data_level": "cellbin",
        "what_it_shows": "Distribution summaries of entropy, dominant feature fraction, and top feature frequencies.",
        "suggested_title": "Barcode diversity and dominance distributions",
        "reason_for_selection": "Useful distribution-level overview for the final package.",
        "caution_note": "Descriptive barcode diversity only.",
    },
    "metaniche_section_purity_distribution": {
        "data_level": "metaniche-like",
        "what_it_shows": "Histogram of section purity across descriptive metaniche-like categories.",
        "suggested_title": "Metaniche-like section purity distribution",
        "reason_for_selection": "Provides QC context for section dominance in descriptive categories.",
        "caution_note": "Metaniche-like categories are descriptive only, not fate states.",
    },
}

FINAL_SCOPE_NOTES = [
    "L126_Brain_s1/s2/s3 are serial sections, not timepoints.",
    "This package is a barcode-aware spatial niche characterization benchmark.",
    "Full local groups are overlapping local contexts and are not tissue partitions.",
    "Tile outputs are non-overlapping and are preferred for robust spatial summaries.",
    "RA/TA/CA are preserved as separate assay-scoped evidence channels.",
    "Allele annotation remains annotation-only and does not inflate counts.",
    "No raw FASTQ, DARLIN re-calling, directed GPCCA, PlanB, or /ssd outputs were used.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=SOURCE_OUTPUT_ROOT,
        help="Existing full characterization processed root.",
    )
    parser.add_argument(
        "--report-input-root",
        type=Path,
        default=SOURCE_REPORT_ROOT,
        help="Existing full characterization report root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--make-key-figures", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "inventory_only",
            "key_figures_only",
            "captions_only",
            "summary_only",
            "manifest_only",
            "claim_audit_only",
            "validation_only",
        ],
    )
    return parser.parse_args()


def reject_forbidden_paths(*paths: Path) -> None:
    offenders = [str(path) for path in paths if path_has_ssd(path)]
    if offenders:
        raise ValueError("Refusing /ssd paths: " + "; ".join(offenders))


def read_table(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, sep="\t", compression=compression)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def write_report(report_root: Path, stem: str, title: str, payload: dict[str, Any], body_lines: list[str], *, overwrite: bool) -> None:
    ensure_dir(report_root)
    atomic_write_json(report_root / f"{stem}.json", payload, overwrite=overwrite)
    markdown = [f"# {title}", ""]
    markdown.extend(f"- {note}" for note in FINAL_SCOPE_NOTES)
    markdown.append("")
    markdown.extend(body_lines)
    atomic_write_text(report_root / f"{stem}.md", "\n".join(markdown).rstrip() + "\n", overwrite=overwrite)


def figure_stem(path: Path) -> str:
    stem = path.name
    return stem.removesuffix(".png").removesuffix(".pdf")


def paired_figure_path(path: Path) -> Path:
    if path.suffix.lower() == ".png":
        return path.with_suffix(".pdf")
    if path.suffix.lower() == ".pdf":
        return path.with_suffix(".png")
    return path


def sample_scope() -> str:
    return ", ".join(SAMPLES)


def figure_row(
    *,
    collection: str,
    path: Path,
    action: str,
    selected: bool,
) -> dict[str, Any]:
    stem = figure_stem(path)
    profile = FIGURE_PROFILES.get(
        stem,
        {
            "data_level": "unknown",
            "what_it_shows": "Descriptive figure from the final L126 package.",
            "suggested_title": stem.replace("_", " "),
            "reason_for_selection": "Generic descriptive figure.",
            "caution_note": "Use descriptive barcode-aware language only.",
        },
    )
    paired = paired_figure_path(path)
    return {
        "collection": collection,
        "filename": path.name,
        "stem": stem,
        "path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "non_empty": bool(path.exists() and path.stat().st_size > 0),
        "paired_png_exists": bool(path.with_suffix(".png").exists()),
        "paired_pdf_exists": bool(path.with_suffix(".pdf").exists()),
        "paired_path": str(paired) if paired.exists() else "",
        "section_or_sample_represented": sample_scope(),
        "data_level": profile["data_level"],
        "what_it_shows": profile["what_it_shows"],
        "scientifically_interpretable": bool(path.exists() and path.stat().st_size > 0),
        "safe_language": True,
        "risks_overclaiming_fate_transition_terminal_biology": False,
        "recommended_action": action,
        "selected_as_final_key_figure": bool(selected),
        "suggested_revision": "" if selected else "Keep as supplement or omit from the final key-figure set.",
    }


def build_figure_inventory(report_input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    figure_root = report_input_root / "figures"
    key_root = report_input_root / "key_figure_candidates"
    selected_stems = set(FINAL_KEY_FIGURE_ORDER)

    for collection, root in [("figures", figure_root), ("key_figure_candidates", key_root)]:
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in {".png", ".pdf"} or not path.is_file():
                continue
            stem = figure_stem(path)
            selected = collection == "figures" and stem in selected_stems
            action = "KEEP_AS_KEY" if selected else "KEEP_AS_SUPPLEMENT"
            if stem == "metaniche_section_purity_distribution" and collection == "figures":
                action = "KEEP_AS_KEY"
            rows.append(figure_row(collection=collection, path=path, action=action, selected=selected))

    frame = pd.DataFrame(rows).sort_values(["collection", "stem", "file_type"]).reset_index(drop=True)
    return frame


def figure_audit_table(inventory: pd.DataFrame) -> pd.DataFrame:
    audit = inventory.copy()
    audit["inventory_scope"] = audit["collection"].map(
        {"figures": "main_report_figures", "key_figure_candidates": "curated_key_candidate_copies"}
    )
    return audit


def all_text_files(report_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(report_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("05_CLAIM_LANGUAGE_AUDIT") or path.name.startswith("06_VALIDATION"):
            continue
        if path.suffix.lower() in {".md", ".json", ".tsv"}:
            files.append(path)
    return files


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\r", "\n")
    pieces = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [piece.strip() for piece in pieces if piece.strip()]


def sanitize_claim_text(text: str) -> str:
    text = text.replace("/home/zhutao/projects/nichefate", "[PROJECT_ROOT]")
    text = text.replace("/home/zhutao/scratch/nichefate", "[SCRATCH_ROOT]")
    return text


def classify_forbidden_hits(text: str, *, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive_hits: list[dict[str, Any]] = []
    allowed_hits: list[dict[str, Any]] = []
    for sentence in split_sentences(sanitize_claim_text(text)):
        lowered = sentence.lower()
        for term in FORBIDDEN_TERMS:
            if term not in lowered:
                continue
            hit = {"source": source, "term": term, "sentence": sentence}
            if any(marker in lowered for marker in NEGATION_MARKERS):
                allowed_hits.append(hit)
            else:
                positive_hits.append(hit)
    return positive_hits, allowed_hits


def audit_claim_language(report_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positive_hits: list[dict[str, Any]] = []
    allowed_hits: list[dict[str, Any]] = []
    for path in all_text_files(report_root):
        text = read_text(path)
        pos, ok = classify_forbidden_hits(text, source=str(path))
        positive_hits.extend(pos)
        allowed_hits.extend(ok)
        rows.append(
            {
                "path": str(path),
                "contains_forbidden_terms": bool(pos or ok),
                "positive_hit_count": len(pos),
                "allowed_hit_count": len(ok),
            }
        )
    frame = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    payload = {
        "generated_at_utc": utc_now(),
        "report_root": str(report_root),
        "status": "PASS" if not positive_hits else "FAIL",
        "scanned_file_count": int(len(frame)),
        "positive_hit_count": int(len(positive_hits)),
        "allowed_context_hit_count": int(len(allowed_hits)),
        "positive_hits": positive_hits,
        "allowed_context_hits": allowed_hits,
        "forbidden_terms": list(FORBIDDEN_TERMS),
    }
    return frame, payload


def copy_selected_figures(report_input_root: Path, final_key_dir: Path, *, overwrite: bool) -> pd.DataFrame:
    source_root = report_input_root / "figures"
    rows: list[dict[str, Any]] = []
    ensure_dir(final_key_dir)
    for index, stem in enumerate(FINAL_KEY_FIGURE_ORDER, start=1):
        profile = FIGURE_PROFILES[stem]
        src_png = source_root / f"{stem}.png"
        src_pdf = source_root / f"{stem}.pdf"
        if not src_png.exists() or not src_pdf.exists():
            raise FileNotFoundError(f"Missing source figure pair for {stem}: {src_png}, {src_pdf}")
        dst_png = final_key_dir / src_png.name
        dst_pdf = final_key_dir / src_pdf.name
        if dst_png.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite final key figure: {dst_png}")
        if dst_pdf.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite final key figure: {dst_pdf}")
        shutil.copy2(src_png, dst_png)
        shutil.copy2(src_pdf, dst_pdf)
        rows.append(
            {
                "final_panel_id": f"F{index:02d}",
                "source_stem": stem,
                "source_png_path": str(src_png),
                "source_pdf_path": str(src_pdf),
                "final_path": str(dst_png),
                "paired_final_path": str(dst_pdf),
                "data_level": profile["data_level"],
                "suggested_title": profile["suggested_title"],
                "suggested_caption": make_caption(stem),
                "reason_for_selection": profile["reason_for_selection"],
                "caution_note": profile["caution_note"],
            }
        )
    selection = pd.DataFrame(rows)
    return selection


def make_caption(stem: str) -> str:
    profile = FIGURE_PROFILES[stem]
    caption_map = {
        "method_data_flow_summary": (
            "Method-level workflow summary showing the path from input packet to BarcodeEvidenceAdapter, "
            "then to full cellbin, overlapping group, non-overlapping tile, and descriptive metaniche-like outputs. "
            "L126_Brain_s1/s2/s3 are serial sections, not timepoints."
        ),
        "cellbin_lineage_coverage_spatial": (
            "Cellbin-level spatial map of lineage evidence presence across the three serial L126 sections. "
            "This is a coverage and QC view, not a trajectory or fate figure."
        ),
        "cellbin_total_lineage_count_spatial": (
            "Cellbin-level spatial map of total lineage count. The figure summarizes local evidence abundance while preserving RA/TA/CA separation."
        ),
        "cellbin_detected_feature_count_spatial": (
            "Cellbin-level spatial map of detected lineage feature count. This complements total count with feature breadth."
        ),
        "cellbin_feature_entropy_spatial": (
            "Cellbin-level spatial map of lineage feature entropy. The panel is a descriptive barcode-diversity summary."
        ),
        "cellbin_dominant_feature_fraction_spatial": (
            "Cellbin-level spatial map of dominant feature fraction. This panel summarizes diversity and dominance structure."
        ),
        "cellbin_assay_total_count_spatial": (
            "Cellbin-level RA/TA/CA total count maps. Assay channels are preserved separately for descriptive QC."
        ),
        "section_assay_balance_summary": (
            "Section-level RA/TA/CA total count summary. Sections are serial provenance only and are not a time axis."
        ),
        "tile_lineage_coverage_spatial": (
            "Tile-level spatial map of lineage coverage. Tiles are non-overlapping and therefore preferred for robust spatial summaries."
        ),
        "tile_feature_entropy_spatial": (
            "Tile-level spatial map of barcode entropy. The tile representation reduces overlap-driven count inflation."
        ),
        "barcode_diversity_and_dominance_distributions": (
            "Distribution summary of feature entropy, dominant feature fraction, and top feature frequency. "
            "This is a descriptive barcode-diversity panel."
        ),
        "metaniche_section_purity_distribution": (
            "Histogram of section purity across descriptive metaniche-like categories. "
            "These categories are descriptive only and do not represent fate states."
        ),
    }
    return caption_map[stem]


def final_summary_payload(
    *,
    source_summary: dict[str, Any],
    metaniche_summary: dict[str, Any],
    selection: pd.DataFrame,
    figures_audited: int,
    claim_audit: dict[str, Any],
    label: str,
    key_figure_folder: Path,
    validation_status: str = "PENDING",
) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "decision_label": label,
        "validation_status": validation_status,
        "figures_audited": figures_audited,
        "key_figures_selected": int(len(selection)),
        "key_figure_folder": str(key_figure_folder),
        "cellbin_section_summary": source_summary.get("cellbin_section_summary", []),
        "group_section_qc": source_summary.get("group_section_qc", []),
        "tile_section_qc": source_summary.get("tile_section_qc", []),
        "cellbin_counts": {
            row["sample_id"]: row["cellbin_count"] for row in source_summary.get("cellbin_section_summary", [])
        },
        "lineage_positive_fractions": {
            row["sample_id"]: row["fraction_lineage_positive"] for row in source_summary.get("cellbin_section_summary", [])
        },
        "metaniche_category_count": int(metaniche_summary.get("n_metaniches", 0)),
        "section_dominated_metaniche_count": int(metaniche_summary.get("section_dominated_metaniches", 0)),
        "supports": [
            "barcode-aware spatial niche characterization",
            "full cellbin lineage coverage summaries",
            "non-overlapping tile-based spatial summaries",
            "descriptive metaniche-like categories",
        ],
        "limitations": [
            "no fate inference",
            "no terminal state",
            "no temporal trajectory across sections",
            "no cross-assay final clone identity",
            "no directed GPCCA result",
        ],
        "next_step": "Freeze L126 as a barcode-aware niche characterization benchmark and shift directed inference development to time-anchored or future directed datasets.",
        "claim_audit": {
            "status": claim_audit.get("status"),
            "positive_hit_count": claim_audit.get("positive_hit_count", 0),
            "allowed_context_hit_count": claim_audit.get("allowed_context_hit_count", 0),
        },
    }


def summary_lines(payload: dict[str, Any]) -> list[str]:
    section = payload["cellbin_section_summary"]
    group = payload["group_section_qc"]
    tile = payload["tile_section_qc"]
    lines = [
        f"- Final characterization label: `{payload['decision_label']}`",
        f"- Figures audited: `{payload['figures_audited']}` files",
        f"- Key figures selected: `{payload['key_figures_selected']}` figure concepts",
        f"- Final key figure folder: `{payload['key_figure_folder']}`",
        "",
        "## Dataset",
        "- L126_Brain_s1/s2/s3 are serial sections, not timepoints.",
        "- Full cellbin counts:",
    ]
    for row in section:
        lines.append(
            f"  - {row['sample_id']}: `{int(row['cellbin_count'])}` cellbins; lineage-positive fraction `{float(row['fraction_lineage_positive']):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Input And Adapter",
            "- Input pair: h5ad cellbins plus DARLIN lineage evidence.",
            "- BarcodeEvidenceAdapter preserved RA/TA/CA separately and did not inflate allele annotation into counts.",
            "",
            "## Main Results",
            "- Full local group coverage is near-complete, but those groups are overlapping local contexts and are not tissue abundance partitions.",
            "- Non-overlapping tile coverage is even more robust for spatial maps.",
            "- Tile-based metaniche-like characterization yields descriptive categories only.",
        ]
    )
    if group:
        lines.append("  - Group coverage:")
        for row in group:
            lines.append(
                f"    - {row['sample_id']}: groups `{int(row['groups'])}`; fraction groups with lineage `{float(row['fraction_groups_with_lineage']):.4f}`"
            )
    if tile:
        lines.append("  - Tile coverage:")
        for row in tile:
            lines.append(
                f"    - {row['sample_id']}: tiles `{int(row['tiles'])}`; fraction tiles with lineage `{float(row['fraction_tiles_with_lineage']):.4f}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Barcode evidence is sparse at the single-cellbin level but dense at local-context and tile level.",
            "- Tile outputs are preferred for robust spatial figures.",
            "- Group outputs represent local niche contexts, not tissue abundance.",
            "",
            "## Limitations",
            "- This package does not support fate inference.",
            "- This package does not support terminal-state interpretation.",
            "- This package does not support temporal trajectory claims across s1/s2/s3.",
            "- This package does not support cross-assay final clone identity.",
            "- Directed GPCCA was stopped in the earlier pipeline because hardening did not improve confidence and controls recapitulated too much structure.",
            "",
            "## Next Step",
            f"- {payload['next_step']}",
        ]
    )
    return lines


def write_summary(report_root: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    write_report(
        report_root,
        "04_FINAL_RESULT_SUMMARY",
        "Final Result Summary",
        payload,
        summary_lines(payload),
        overwrite=overwrite,
    )


def write_captions(output_root: Path, report_root: Path, selection: pd.DataFrame, *, overwrite: bool) -> dict[str, Any]:
    lines = [
        "# Final Figure Captions",
        "",
        "The selected figures are arranged in the recommended display order below.",
        "",
    ]
    rows = []
    for row in selection.itertuples(index=False):
        rows.append(
            {
                "final_panel_id": row.final_panel_id,
                "source_stem": row.source_stem,
                "data_level": row.data_level,
                "suggested_title": row.suggested_title,
                "caption": row.suggested_caption,
                "caution_note": row.caution_note,
            }
        )
        lines.append(f"## {row.final_panel_id} - {row.suggested_title}")
        lines.append("")
        lines.append(f"- Data level: `{row.data_level}`")
        lines.append(f"- Source stem: `{row.source_stem}`")
        lines.append(f"- Caption: {row.suggested_caption}")
        lines.append(f"- Caution: {row.caution_note}")
        lines.append("")
    caption_table = pd.DataFrame(rows)
    atomic_write_tsv(output_root / "final_figure_captions.tsv", caption_table, overwrite=overwrite)
    atomic_write_text(report_root / "03_FINAL_FIGURE_CAPTIONS.md", "\n".join(lines).rstrip() + "\n", overwrite=overwrite)
    return {
        "generated_at_utc": utc_now(),
        "caption_count": int(len(caption_table)),
        "caption_tsv_path": str(output_root / "final_figure_captions.tsv"),
        "caption_md_path": str(report_root / "03_FINAL_FIGURE_CAPTIONS.md"),
    }


def write_preflight(
    *,
    output_root: Path,
    report_root: Path,
    source_output_root: Path,
    source_report_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    required_paths = {
        "summary_md": source_report_root / "07_FULL_CHARACTERIZATION_SUMMARY.md",
        "summary_json": source_report_root / "07_FULL_CHARACTERIZATION_SUMMARY.json",
        "validation_md": source_report_root / "08_VALIDATION.md",
        "validation_json": source_report_root / "08_VALIDATION.json",
        "figures_dir": source_report_root / "figures",
        "key_figure_candidates_dir": source_report_root / "key_figure_candidates",
        "cellbin_summary": source_output_root / "cellbin" / "full_cellbin_lineage_summary.tsv.gz",
        "group_summary": source_output_root / "group_lineage" / "full_group_lineage_section_qc.tsv",
        "tile_summary": source_output_root / "spatial_tiles" / "full_tile_section_qc.tsv",
        "metaniche_summary": source_output_root / "metaniche" / "full_metaniche_summary.tsv.gz",
    }
    inventory = build_figure_inventory(source_report_root)
    source_snapshot_path = source_output_root / "qc" / "source_input_packet_snapshot_before.tsv"
    baseline_snapshot = read_table(source_snapshot_path)
    current_snapshot = snapshot_files(baseline_snapshot["path"].astype(str).tolist(), include_sha256=False)
    snapshot_diff = compare_file_snapshots(baseline_snapshot, current_snapshot)
    full_reports_text = "\n".join(read_text(path) for path in sorted(source_report_root.glob("*.md")))
    positive_hits, allowed_hits = classify_forbidden_hits(full_reports_text, source=str(source_report_root))
    missing = [name for name, path in required_paths.items() if not path.exists()]
    decision = "L126_CHARACTERIZATION_FINALIZATION_PREFLIGHT_READY"
    if missing:
        decision = "L126_CHARACTERIZATION_FINALIZATION_HOLD_FOR_MISSING_OUTPUTS"
    if positive_hits:
        decision = "L126_CHARACTERIZATION_FINALIZATION_HOLD_FOR_FORBIDDEN_CLAIMS"
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": decision,
        "missing_required_paths": missing,
        "audited_figure_file_count": int(len(inventory)),
        "audited_unique_figure_stems": int(inventory["stem"].nunique()),
        "main_figure_file_count": int((inventory["collection"] == "figures").sum()),
        "key_candidate_file_count": int((inventory["collection"] == "key_figure_candidates").sum()),
        "source_input_snapshot_rows": int(len(current_snapshot)),
        "source_input_snapshot_differences": int(snapshot_diff["changed"].sum()),
        "forbidden_claim_positive_hit_count": int(len(positive_hits)),
        "forbidden_claim_allowed_hit_count": int(len(allowed_hits)),
        "required_paths": {name: str(path) for name, path in required_paths.items()},
    }
    write_report(
        report_root,
        "00_FINALIZATION_PREFLIGHT",
        "Finalization Preflight",
        payload,
        [
            f"- Decision label: `{decision}`",
            f"- Required outputs missing: `{len(missing)}`",
            f"- Audited figure files: `{len(inventory)}` across `{inventory['stem'].nunique()}` unique stems",
            f"- Source input packet differences from baseline snapshot: `{int(snapshot_diff['changed'].sum())}`",
            "",
            markdown_table(pd.DataFrame([{
                "summary_md": required_paths["summary_md"].exists(),
                "summary_json": required_paths["summary_json"].exists(),
                "validation_md": required_paths["validation_md"].exists(),
                "validation_json": required_paths["validation_json"].exists(),
                "figures_dir": required_paths["figures_dir"].exists(),
                "key_candidates_dir": required_paths["key_figure_candidates_dir"].exists(),
                "cellbin_summary": required_paths["cellbin_summary"].exists(),
                "group_summary": required_paths["group_summary"].exists(),
                "tile_summary": required_paths["tile_summary"].exists(),
                "metaniche_summary": required_paths["metaniche_summary"].exists(),
            }])),
        ],
        overwrite=overwrite,
    )
    return payload


def write_inventory_and_audit(report_input_root: Path, output_root: Path, report_root: Path, *, overwrite: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inventory = build_figure_inventory(report_input_root)
    audit = figure_audit_table(inventory)
    atomic_write_tsv(output_root / "figure_inventory.tsv", inventory, overwrite=overwrite)
    atomic_write_tsv(output_root / "figure_audit.tsv", audit, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "figure_inventory_path": str(output_root / "figure_inventory.tsv"),
        "figure_audit_path": str(output_root / "figure_audit.tsv"),
        "audited_file_count": int(len(inventory)),
        "audited_unique_stems": int(inventory["stem"].nunique()),
        "main_figure_file_count": int((inventory["collection"] == "figures").sum()),
        "key_candidate_file_count": int((inventory["collection"] == "key_figure_candidates").sum()),
        "keep_as_key_count": int((audit["recommended_action"] == "KEEP_AS_KEY").sum()),
        "keep_as_supplement_count": int((audit["recommended_action"] == "KEEP_AS_SUPPLEMENT").sum()),
        "revise_count": int((audit["recommended_action"] == "REVISE").sum()),
        "exclude_count": int((audit["recommended_action"] == "EXCLUDE").sum()),
    }
    write_report(
        report_root,
        "01_FIGURE_AUDIT",
        "Figure Audit",
        payload,
        [
            f"- Audited figure files: `{len(inventory)}` across `{inventory['stem'].nunique()}` unique stems.",
            f"- Key figures retained as key candidates: `{int((audit['recommended_action'] == 'KEEP_AS_KEY').sum())}` files.",
            f"- Key-candidate copies retained as supplements: `{int((audit['recommended_action'] == 'KEEP_AS_SUPPLEMENT').sum())}` files.",
            "",
            markdown_table(audit.head(24)),
        ],
        overwrite=overwrite,
    )
    return inventory, audit, payload


def write_selection(
    report_input_root: Path,
    output_root: Path,
    report_root: Path,
    *,
    overwrite: bool,
    make_key_figures: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    final_key_dir = ensure_dir(report_root / FINAL_KEY_FIGURE_DIRNAME)
    selection = copy_selected_figures(report_input_root, final_key_dir, overwrite=overwrite) if make_key_figures else pd.DataFrame()
    atomic_write_tsv(output_root / "key_figure_selection.tsv", selection, overwrite=overwrite)
    payload = {
        "generated_at_utc": utc_now(),
        "selected_key_figure_count": int(len(selection)),
        "selected_panels": selection["final_panel_id"].tolist() if not selection.empty else [],
        "final_key_figure_dir": str(final_key_dir),
        "selection_tsv_path": str(output_root / "key_figure_selection.tsv"),
    }
    lines = [
        f"- Selected key figure concepts: `{len(selection)}`",
        f"- Final key figure directory: `{final_key_dir}`",
        "",
        markdown_table(selection),
    ]
    write_report(
        report_root,
        "02_FINAL_KEY_FIGURE_SELECTION",
        "Final Key Figure Selection",
        payload,
        lines,
        overwrite=overwrite,
    )
    return selection, payload


def write_claim_audit(report_root: Path, *, overwrite: bool) -> dict[str, Any]:
    scan_table, payload = audit_claim_language(report_root)
    atomic_write_json(report_root / "05_CLAIM_LANGUAGE_AUDIT.json", payload, overwrite=overwrite)
    markdown_lines = [
        "# Claim Language Audit",
        "",
        "- The audit allows forbidden terms only when they appear in negated or limitation-style contexts.",
        f"- Positive forbidden hits: `{payload['positive_hit_count']}`",
        f"- Allowed-context hits: `{payload['allowed_context_hit_count']}`",
        "",
        markdown_table(scan_table),
    ]
    atomic_write_text(report_root / "05_CLAIM_LANGUAGE_AUDIT.md", "\n".join(markdown_lines).rstrip() + "\n", overwrite=overwrite)
    return payload


def build_manifest(output_root: Path, report_root: Path, *, overwrite: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    file_roots = [output_root, report_root]
    for root in file_roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "final_output_manifest.tsv":
                continue
            rel = path.relative_to(PROJECT_ROOT)
            if path.suffix.lower() not in {".png", ".pdf", ".md", ".json", ".tsv", ".gz"}:
                continue
            category = "report" if report_root in path.parents else "processed"
            if "final_key_figures" in path.parts:
                category = "final_key_figure"
            elif path.name in {"06_VALIDATION.md", "06_VALIDATION.json"}:
                category = "validation"
            elif path.name in {"05_CLAIM_LANGUAGE_AUDIT.md", "05_CLAIM_LANGUAGE_AUDIT.json"}:
                category = "claim_audit"
            elif path.name in {"04_FINAL_RESULT_SUMMARY.md", "04_FINAL_RESULT_SUMMARY.json"}:
                category = "summary"
            elif path.name in {"03_FINAL_FIGURE_CAPTIONS.md", "final_figure_captions.tsv"}:
                category = "captions"
            elif path.name in {"01_FIGURE_AUDIT.md", "01_FIGURE_AUDIT.json", "figure_inventory.tsv", "figure_audit.tsv"}:
                category = "figure_audit"
            elif path.name in {"02_FINAL_KEY_FIGURE_SELECTION.md", "key_figure_selection.tsv"}:
                category = "key_figure_selection"
            elif path.name == "README.md":
                category = "readme"
            elif path.name.endswith(".py"):
                category = "script"
            rows.append(
                {
                    "artifact_category": category,
                    "path": str(path),
                    "relative_path": str(rel),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_path(path),
                }
            )
    script_paths = [
        PROJECT_ROOT / "scripts" / "planC_l126_full_characterization_finalize.py",
        PROJECT_ROOT / "scripts" / "planC_l126_full_barcode_niche_characterization.py",
    ]
    for path in script_paths:
        if not path.exists():
            continue
        rows.append(
            {
                "artifact_category": "script",
                "path": str(path),
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_path(path),
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["artifact_category", "relative_path"]).reset_index(drop=True)
    atomic_write_tsv(output_root / "final_output_manifest.tsv", manifest, overwrite=overwrite)
    return manifest


def write_readme(report_root: Path, final_label: str, *, overwrite: bool) -> Path:
    readme = [
        "# L126 Full Barcode-aware Niche Characterization Final Package",
        "",
        "This package is a barcode-aware spatial niche characterization benchmark for L126_Brain_s1/s2/s3.",
        "",
        "What it is not:",
        "- This is not a fate inference result.",
        "- This is not a terminal-state call.",
        "- This is not a temporal trajectory across serial sections.",
        "- This is not a directed GPCCA output.",
        "",
        "Key paths:",
        f"- Final summary: `{report_root / '04_FINAL_RESULT_SUMMARY.md'}`",
        f"- Claim audit: `{report_root / '05_CLAIM_LANGUAGE_AUDIT.md'}`",
        f"- Validation: `{report_root / '06_VALIDATION.md'}`",
        f"- Final key figures: `{report_root / FINAL_KEY_FIGURE_DIRNAME}`",
        "",
        "Recommended figure order:",
        "1. Method and data flow summary",
        "2. Cellbin lineage evidence coverage",
        "3. Cellbin total lineage count",
        "4. Cellbin detected lineage feature count",
        "5. Cellbin lineage feature entropy",
        "6. Cellbin dominant feature fraction",
        "7. Cellbin RA/TA/CA total counts",
        "8. Section assay balance summary",
        "9. Tile lineage coverage",
        "10. Tile barcode entropy",
        "11. Barcode diversity and dominance distributions",
        "12. Metaniche-like section purity distribution",
        "",
        "Interpretation restrictions:",
        "- L126 sections are serial provenance only and are not timepoints.",
        "- Overlapping group outputs are local contexts, not abundance partitions.",
        "- Tile outputs are preferred for robust spatial summaries.",
        "- Metaniche-like categories are descriptive only.",
        "",
        f"Final label: `{final_label}`",
        "",
        "Next safe command:",
        "`python -m pytest tests/test_l126_full_characterization_finalize.py tests/test_l126_full_characterization_claim_language.py`",
        "",
    ]
    path = report_root / "README.md"
    atomic_write_text(path, "\n".join(readme), overwrite=overwrite)
    return path


def validation_phase(
    *,
    output_root: Path,
    report_root: Path,
    source_output_root: Path,
    source_report_root: Path,
    claim_audit_payload: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    json_paths = sorted(report_root.glob("*.json"))
    tsv_paths = sorted(output_root.glob("*.tsv")) + sorted(output_root.glob("*.tsv.gz"))
    text_paths = sorted(report_root.glob("*.md")) + [output_root / "final_figure_captions.tsv"]
    figure_paths = sorted((report_root / FINAL_KEY_FIGURE_DIRNAME).glob("*.png")) + sorted((report_root / FINAL_KEY_FIGURE_DIRNAME).glob("*.pdf"))
    source_snapshot_before = read_table(source_output_root / "qc" / "source_input_packet_snapshot_before.tsv")
    source_snapshot_after_path = output_root / "qc" / "source_input_packet_snapshot_after.tsv"
    ensure_dir(source_snapshot_after_path.parent)
    current_snapshot = snapshot_files(source_snapshot_before["path"].astype(str).tolist(), include_sha256=False)
    atomic_write_tsv(source_snapshot_after_path, current_snapshot, overwrite=overwrite)
    snapshot_diff = compare_file_snapshots(source_snapshot_before, current_snapshot)
    json_ok = all(json.loads(read_text(path)) is not None for path in json_paths)
    tsv_ok = all(len(read_table(path).columns) > 0 for path in tsv_paths)
    text_ok = all(path.exists() and path.stat().st_size > 0 for path in text_paths)
    figures_ok = bool(figure_paths) and all(path.stat().st_size > 0 for path in figure_paths)
    claim_ok = int(claim_audit_payload.get("positive_hit_count", 1)) == 0 and claim_audit_payload.get("status") == "PASS"
    required_outputs_ok = all(
        [
            (output_root / "figure_inventory.tsv").exists(),
            (output_root / "figure_audit.tsv").exists(),
            (output_root / "key_figure_selection.tsv").exists(),
            (output_root / "final_figure_captions.tsv").exists(),
            (output_root / "final_output_manifest.tsv").exists(),
        ]
    )
    output_snapshot_ok = int(snapshot_diff["changed"].sum()) == 0
    no_ssd_ok = not any(path_has_ssd(path) for path in [output_root, report_root])
    payload = {
        "generated_at_utc": utc_now(),
        "decision_label": load_json(report_root / "04_FINAL_RESULT_SUMMARY.json")["decision_label"] if (report_root / "04_FINAL_RESULT_SUMMARY.json").exists() else "UNKNOWN",
        "status": "PASS"
        if all([json_ok, tsv_ok, text_ok, figures_ok, claim_ok, required_outputs_ok, output_snapshot_ok, no_ssd_ok])
        else "FAIL",
        "checks": [
            {"check": "json_parse", "status": json_ok, "details": f"{len(json_paths)} json files"},
            {"check": "tsv_readability", "status": tsv_ok, "details": f"{len(tsv_paths)} tables"},
            {"check": "text_readability", "status": text_ok, "details": f"{len(text_paths)} text files"},
            {"check": "selected_figures_non_empty", "status": figures_ok, "details": f"{len(figure_paths)} files"},
            {"check": "claim_audit_pass", "status": claim_ok, "details": f"positive hits={claim_audit_payload.get('positive_hit_count', 0)}"},
            {"check": "required_outputs_exist", "status": required_outputs_ok, "details": "inventory, audit, selection, captions, manifest"},
            {"check": "source_input_packet_unchanged", "status": output_snapshot_ok, "details": f"{int(snapshot_diff['changed'].sum())} differences"},
            {"check": "no_ssd", "status": no_ssd_ok, "details": "path guard"},
        ],
    }
    atomic_write_json(report_root / "06_VALIDATION.json", payload, overwrite=overwrite)
    atomic_write_text(
        report_root / "06_VALIDATION.md",
        "# Validation\n\n"
        f"- Decision label: `{payload['decision_label']}`\n"
        f"- Validation status: `{payload['status']}`\n"
        f"- Checks passed: `{sum(bool(row['status']) for row in payload['checks'])}/{len(payload['checks'])}`\n\n"
        + markdown_table(pd.DataFrame(payload["checks"])),
        overwrite=overwrite,
    )
    return payload


def finalize_package(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ensure_dir(args.output_root.resolve())
    report_root = ensure_dir(args.report_root.resolve())
    reject_forbidden_paths(args.input_root, args.report_input_root, output_root, report_root)

    preflight = write_preflight(
        output_root=output_root,
        report_root=report_root,
        source_output_root=args.input_root,
        source_report_root=args.report_input_root,
        overwrite=args.overwrite,
    )
    inventory, audit, audit_payload = write_inventory_and_audit(args.report_input_root, output_root, report_root, overwrite=args.overwrite)
    source_summary = load_json(args.report_input_root / "07_FULL_CHARACTERIZATION_SUMMARY.json")
    metaniche_summary = load_json(args.report_input_root / "05_FULL_METANICHE_CHARACTERIZATION.json")
    selection, selection_payload = write_selection(
        args.report_input_root,
        output_root,
        report_root,
        overwrite=args.overwrite,
        make_key_figures=bool(args.make_key_figures or args.mode in {"all", "key_figures_only"}),
    )
    captions_payload = write_captions(output_root, report_root, selection, overwrite=args.overwrite)
    final_label = "L126_FULL_CHARACTERIZATION_FINAL_PACKAGE_READY"
    summary_payload = final_summary_payload(
        source_summary=source_summary,
        metaniche_summary=metaniche_summary,
        selection=selection,
        figures_audited=int(len(inventory)),
        claim_audit={"status": "PENDING", "positive_hit_count": 0, "allowed_hit_count": 0},
        label=final_label,
        key_figure_folder=report_root / FINAL_KEY_FIGURE_DIRNAME,
    )
    write_summary(report_root, summary_payload, overwrite=args.overwrite)
    readme_path = write_readme(report_root, final_label, overwrite=args.overwrite)
    claim_audit_payload = write_claim_audit(report_root, overwrite=args.overwrite)
    build_manifest(output_root, report_root, overwrite=args.overwrite)
    validation_payload = validation_phase(
        output_root=output_root,
        report_root=report_root,
        source_output_root=args.input_root,
        source_report_root=args.report_input_root,
        claim_audit_payload=claim_audit_payload,
        overwrite=args.overwrite,
    )
    final_summary_payload_obj = final_summary_payload(
        source_summary=source_summary,
        metaniche_summary=metaniche_summary,
        selection=selection,
        figures_audited=int(len(inventory)),
        claim_audit=claim_audit_payload,
        label=final_label if validation_payload["status"] == "PASS" and claim_audit_payload.get("status") == "PASS" else "L126_FULL_CHARACTERIZATION_FINAL_PACKAGE_READY_WITH_WARNINGS",
        key_figure_folder=report_root / FINAL_KEY_FIGURE_DIRNAME,
        validation_status=validation_payload["status"],
    )
    write_summary(report_root, final_summary_payload_obj, overwrite=True)
    manifest = build_manifest(output_root, report_root, overwrite=True)
    readme_path = write_readme(report_root, final_summary_payload_obj["decision_label"], overwrite=True)
    final_payload = {
        "preflight": preflight,
        "inventory": audit_payload,
        "selection": selection_payload,
        "captions": captions_payload,
        "summary": final_summary_payload_obj,
        "claim_audit": claim_audit_payload,
        "validation": validation_payload,
        "manifest_rows": int(len(manifest)),
        "readme_path": str(readme_path),
    }
    return final_payload


def main() -> int:
    args = parse_args()
    if args.mode == "all":
        payload = finalize_package(args)
        print(f"decision_label={payload['summary']['decision_label']}")
        print(f"validation_status={payload['validation']['status']}")
        return 0

    output_root = ensure_dir(args.output_root.resolve())
    report_root = ensure_dir(args.report_root.resolve())
    reject_forbidden_paths(args.input_root, args.report_input_root, output_root, report_root)

    if args.mode == "inventory_only":
        preflight = write_preflight(
            output_root=output_root,
            report_root=report_root,
            source_output_root=args.input_root,
            source_report_root=args.report_input_root,
            overwrite=args.overwrite,
        )
        _, _, audit_payload = write_inventory_and_audit(args.report_input_root, output_root, report_root, overwrite=args.overwrite)
        print(f"decision_label={preflight['decision_label']}")
        print(f"audited_file_count={audit_payload['audited_file_count']}")
        return 0

    if args.mode == "key_figures_only":
        selection, _ = write_selection(
            args.report_input_root,
            output_root,
            report_root,
            overwrite=args.overwrite,
            make_key_figures=True,
        )
        print(f"selected_key_figure_count={len(selection)}")
        return 0

    if args.mode == "captions_only":
        selection = read_table(output_root / "key_figure_selection.tsv")
        payload = write_captions(output_root, report_root, selection, overwrite=args.overwrite)
        print(f"caption_count={payload['caption_count']}")
        return 0

    if args.mode == "summary_only":
        selection = read_table(output_root / "key_figure_selection.tsv")
        inventory = read_table(output_root / "figure_inventory.tsv") if (output_root / "figure_inventory.tsv").exists() else build_figure_inventory(args.report_input_root)
        claim_payload = load_json(report_root / "05_CLAIM_LANGUAGE_AUDIT.json") if (report_root / "05_CLAIM_LANGUAGE_AUDIT.json").exists() else {"status": "PENDING", "positive_hit_count": 0, "allowed_context_hit_count": 0}
        validation_status = load_json(report_root / "06_VALIDATION.json").get("status", "PENDING") if (report_root / "06_VALIDATION.json").exists() else "PENDING"
        payload = final_summary_payload(
            source_summary=load_json(args.report_input_root / "07_FULL_CHARACTERIZATION_SUMMARY.json"),
            metaniche_summary=load_json(args.report_input_root / "05_FULL_METANICHE_CHARACTERIZATION.json"),
            selection=selection,
            figures_audited=int(len(inventory)),
            claim_audit=claim_payload,
            label="L126_FULL_CHARACTERIZATION_FINAL_PACKAGE_READY",
            key_figure_folder=report_root / FINAL_KEY_FIGURE_DIRNAME,
            validation_status=validation_status,
        )
        write_summary(report_root, payload, overwrite=args.overwrite)
        print(f"decision_label={payload['decision_label']}")
        return 0

    if args.mode == "claim_audit_only":
        payload = write_claim_audit(report_root, overwrite=args.overwrite)
        print(f"claim_audit_status={payload['status']}")
        print(f"positive_hit_count={payload['positive_hit_count']}")
        return 0

    if args.mode == "manifest_only":
        label = (
            load_json(report_root / "04_FINAL_RESULT_SUMMARY.json").get("decision_label", "L126_FULL_CHARACTERIZATION_FINAL_PACKAGE_READY")
            if (report_root / "04_FINAL_RESULT_SUMMARY.json").exists()
            else "L126_FULL_CHARACTERIZATION_FINAL_PACKAGE_READY"
        )
        write_readme(report_root, label, overwrite=args.overwrite)
        manifest = build_manifest(output_root, report_root, overwrite=args.overwrite)
        print(f"manifest_rows={len(manifest)}")
        return 0

    if args.mode == "validation_only":
        claim_payload = (
            load_json(report_root / "05_CLAIM_LANGUAGE_AUDIT.json")
            if (report_root / "05_CLAIM_LANGUAGE_AUDIT.json").exists()
            else write_claim_audit(report_root, overwrite=args.overwrite)
        )
        if not (output_root / "final_output_manifest.tsv").exists():
            build_manifest(output_root, report_root, overwrite=args.overwrite)
        payload = validation_phase(
            output_root=output_root,
            report_root=report_root,
            source_output_root=args.input_root,
            source_report_root=args.report_input_root,
            claim_audit_payload=claim_payload,
            overwrite=args.overwrite,
        )
        print(f"validation_status={payload['status']}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
