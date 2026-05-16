"""Production preflight helpers for the full PlanA-K parameter lock.

The functions in this module are intentionally inspect-only. They collect
metadata, generate draft configs, and write command blueprints without running
full M2.5, GPCCA, Slurm, DARLIN, BranchSBM, or any raw/frozen-output mutation.
"""

from __future__ import annotations

import getpass
import json
import math
import platform
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd

from .io import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_tsv,
    disk_usage,
    ensure_dir,
    file_summary,
    git_branch,
    git_root,
    git_status_short,
    read_memory_info,
    utc_now,
)
from .reporting import dataframe_to_markdown
from .schemas import (
    M1_BY_SLICE_ROOT,
    M2_BY_SLICE_ROOT,
    M2_SCHEMA_PATH,
    PLAN_A_K_PRODUCTION_SCRATCH_ROOT,
    PRODUCTION_PREFLIGHT_ROOT,
    PROJECT_ROOT,
)


DECISION_LABELS = {
    "DIRECT_FULL_RUN_READY",
    "DIRECT_FULL_RUN_READY_WITH_RESOURCE_CAUTION",
    "NEEDS_SCALE_CONTROLLED_FALLBACK",
    "NOT_READY_MAJOR_BLOCKERS",
}

FORBIDDEN_ACTIONS = [
    "DARLIN processing",
    "raw data modification",
    "frozen P_fate output modification",
    "full production M2.5 execution",
    "full GPCCA execution",
    "BranchSBM training",
    "Slurm submission",
    "/ssd output writes",
    "git add/commit/push/PR actions",
]

PRODUCTION_METADATA_LOCK = {
    "m1_by_slice_file_count": 58,
    "m2_by_slice_file_count": 58,
    "slice_ids_match": True,
    "total_m2_anchors_from_parquet_metadata": 1_439_542,
    "timepoints": ["D0", "D3", "D9", "D21", "D35"],
    "m2_output_column_count": 775,
    "m2_numeric_feature_column_count": 765,
    "m2_metadata_column_count": 10,
    "safe_feature_mode_column_count": 600,
    "required_join_keys": ["slice_id", "anchor_index", "anchor_cell_id"],
    "required_m1_coordinate_columns": ["x", "y"],
}

REPRODUCIBILITY_REQUIREMENTS = [
    "exact feature list",
    "scaler parameters or scaler object",
    "PCA components or PCA object",
    "training sample manifest",
    "random seed",
    "software environment record",
]

PRODUCTION_ORDER = [
    "full M2.5",
    "full M2.5 QC",
    "full Kmix_A",
    "full kernel QC",
    "full GPCCA",
    "full macrostate annotation",
]


def _import_parquet() -> Any:
    import pyarrow.parquet as pq

    return pq


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_full_m2_schema(schema_path: Path = M2_SCHEMA_PATH) -> dict[str, Any]:
    data = _load_json(schema_path)
    metadata_columns = list(data.get("metadata_columns", []))
    numeric_feature_columns = list(data.get("numeric_feature_columns", []))
    output_columns = list(data.get("output_columns", []))
    if not output_columns:
        output_columns = [*metadata_columns, *numeric_feature_columns]
    data.update(
        {
            "schema_path": str(schema_path),
            "exists": schema_path.exists(),
            "metadata_columns": metadata_columns,
            "numeric_feature_columns": numeric_feature_columns,
            "output_columns": output_columns,
            "metadata_column_count": int(data.get("metadata_column_count", len(metadata_columns))),
            "numeric_feature_column_count": int(
                data.get("numeric_feature_column_count", len(numeric_feature_columns))
            ),
            "output_column_count": int(data.get("output_column_count", len(output_columns))),
        }
    )
    return data


def _read_first_parquet_value(path: Path, column: str) -> Any:
    pq = _import_parquet()
    parquet = pq.ParquetFile(path)
    if column not in parquet.schema_arrow.names:
        return None
    for batch in parquet.iter_batches(columns=[column], batch_size=1):
        frame = batch.to_pandas()
        if not frame.empty:
            return frame.iloc[0][column]
    return None


def _read_parquet_head(path: Path, columns: list[str], batch_size: int) -> pd.DataFrame:
    pq = _import_parquet()
    parquet = pq.ParquetFile(path)
    available = [column for column in columns if column in parquet.schema_arrow.names]
    if not available:
        return pd.DataFrame(columns=columns)
    for batch in parquet.iter_batches(columns=available, batch_size=batch_size):
        return batch.to_pandas()
    return pd.DataFrame(columns=available)


def collect_production_environment_payload() -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "hostname": platform.node() or "unknown",
        "user": getpass.getuser(),
        "pwd": str(Path.cwd()),
        "git_root": git_root(),
        "git_branch": git_branch(),
        "git_status_short": git_status_short(),
        "disk_usage": {
            "repo_root": disk_usage(PROJECT_ROOT),
            "/home": disk_usage(Path("/home")),
            "production_scratch_root_parent": disk_usage(PLAN_A_K_PRODUCTION_SCRATCH_ROOT.parent),
        },
        "memory": read_memory_info(),
        "guardrails": {
            "forbidden_actions": FORBIDDEN_ACTIONS,
            "preflight_only": True,
            "uses_ssd_outputs": False,
        },
    }


def build_recovery_note_payload() -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "interruption": "unexpected status 503 Service Unavailable: auth_unavailable",
        "interpretation": "Codex/API auth interruption; no code failure was found during recovery inspection.",
        "phase0_recovery_findings": {
            "schema_constants_present": True,
            "production_preflight_module_present_before_resume": False,
            "production_preflight_scripts_present_before_resume": False,
            "production_preflight_tests_present_before_resume": False,
            "production_preflight_report_dir_present_before_resume": False,
            "partial_syntax_invalid_files_found": False,
        },
        "remaining_phases_at_resume": [
            "add production_preflight helpers",
            "export helpers through package and compatibility facade",
            "add dry-run scripts",
            "add synthetic tests",
            "generate lightweight reports and draft configs",
            "run validation",
        ],
    }


def recovery_note_markdown(payload: dict[str, Any]) -> str:
    findings = payload["phase0_recovery_findings"]
    rows = pd.DataFrame(
        [{"item": key, "value": value} for key, value in findings.items()]
    )
    return dedent(
        f"""
        # PlanA-K Production Preflight Recovery Note

        - Interruption: `{payload["interruption"]}`
        - Recovery interpretation: {payload["interpretation"]}

        ## Findings

        {dataframe_to_markdown(rows)}

        ## Remaining Work

        {chr(10).join(f"- {item}" for item in payload["remaining_phases_at_resume"])}
        """
    ).strip() + "\n"


def discover_full_input_availability(
    m1_root: Path = M1_BY_SLICE_ROOT,
    m2_root: Path = M2_BY_SLICE_ROOT,
    schema_path: Path = M2_SCHEMA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema = load_full_m2_schema(schema_path)
    required_keys = set(PRODUCTION_METADATA_LOCK["required_join_keys"])
    required_coords = set(PRODUCTION_METADATA_LOCK["required_m1_coordinate_columns"])
    m1_files = {path.parent.name: path for path in sorted(m1_root.glob("*/niche_features_*.parquet"))}
    m2_files = {path.parent.name: path for path in sorted(m2_root.glob("*/m2_representation_*.parquet"))}

    rows: list[dict[str, Any]] = []
    timepoint_totals: dict[str, int] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    pq = None
    for slice_id in sorted(set(m1_files) | set(m2_files)):
        m1_path = m1_files.get(slice_id)
        m2_path = m2_files.get(slice_id)
        row: dict[str, Any] = {
            "slice_id": slice_id,
            "m1_path": str(m1_path) if m1_path else "",
            "m2_path": str(m2_path) if m2_path else "",
            "m1_exists": m1_path is not None,
            "m2_exists": m2_path is not None,
            "m1_has_xy": False,
            "m2_has_required_join_keys": False,
            "m2_rows": 0,
            "m2_columns": 0,
            "timepoint": None,
            "inspection_note": "",
        }
        try:
            if pq is None:
                pq = _import_parquet()
            if m1_path:
                m1_parquet = pq.ParquetFile(m1_path)
                m1_columns = set(m1_parquet.schema_arrow.names)
                row["m1_rows"] = int(m1_parquet.metadata.num_rows)
                row["m1_has_xy"] = required_coords.issubset(m1_columns)
            if m2_path:
                m2_parquet = pq.ParquetFile(m2_path)
                m2_columns = set(m2_parquet.schema_arrow.names)
                row["m2_rows"] = int(m2_parquet.metadata.num_rows)
                row["m2_columns"] = int(len(m2_columns))
                row["m2_has_required_join_keys"] = required_keys.issubset(m2_columns)
                row["timepoint"] = _read_first_parquet_value(m2_path, "time")
                if row["timepoint"] is not None:
                    timepoint_totals[str(row["timepoint"])] = (
                        timepoint_totals.get(str(row["timepoint"]), 0) + int(row["m2_rows"])
                    )
            row["inspection_note"] = "Parquet metadata and one time value inspected only"
        except Exception as exc:
            row["inspection_note"] = f"metadata inspection failed: {exc}"
            warnings.append(f"{slice_id}: {exc}")
        rows.append(row)

    frame = pd.DataFrame(rows)
    observed = {
        "m1_by_slice_file_count": int(len(m1_files)),
        "m2_by_slice_file_count": int(len(m2_files)),
        "slice_ids_match": set(m1_files) == set(m2_files),
        "total_m2_anchors_from_parquet_metadata": int(frame["m2_rows"].sum()) if not frame.empty else 0,
        "timepoints": sorted(timepoint_totals, key=_timepoint_sort_key),
        "m2_output_column_counts": sorted(frame.loc[frame["m2_exists"], "m2_columns"].dropna().astype(int).unique().tolist())
        if not frame.empty
        else [],
        "m2_schema_exists": bool(schema.get("exists")),
        "m2_schema_output_column_count": int(schema.get("output_column_count", 0)),
        "m2_schema_numeric_feature_column_count": int(schema.get("numeric_feature_column_count", 0)),
        "m2_schema_metadata_column_count": int(schema.get("metadata_column_count", 0)),
        "m2_required_join_keys_present_all_slices": bool(frame["m2_has_required_join_keys"].all())
        if not frame.empty
        else False,
        "m1_coordinate_columns_present_all_slices": bool(frame["m1_has_xy"].all()) if not frame.empty else False,
        "timepoint_anchor_totals": timepoint_totals,
    }
    if observed["m1_by_slice_file_count"] != PRODUCTION_METADATA_LOCK["m1_by_slice_file_count"]:
        blockers.append("M1 by-slice file count differs from locked metadata finding.")
    if observed["m2_by_slice_file_count"] != PRODUCTION_METADATA_LOCK["m2_by_slice_file_count"]:
        blockers.append("M2 by-slice file count differs from locked metadata finding.")
    if not observed["slice_ids_match"]:
        blockers.append("M1/M2 slice IDs do not match.")
    if observed["total_m2_anchors_from_parquet_metadata"] != PRODUCTION_METADATA_LOCK["total_m2_anchors_from_parquet_metadata"]:
        blockers.append("Total M2 anchors differs from locked Parquet metadata finding.")
    if observed["timepoints"] != PRODUCTION_METADATA_LOCK["timepoints"]:
        blockers.append("Observed timepoints differ from locked production decision.")
    if not observed["m2_required_join_keys_present_all_slices"]:
        blockers.append("Required M2 join keys are missing from at least one slice schema.")
    if not observed["m1_coordinate_columns_present_all_slices"]:
        blockers.append("M1 coordinate columns x/y are missing from at least one slice schema.")
    if not observed["m2_schema_exists"]:
        blockers.append("M2 full feature schema is missing.")

    summary = {
        "generated_at_utc": utc_now(),
        "m1_root": str(m1_root),
        "m2_root": str(m2_root),
        "schema_path": str(schema_path),
        "locked_metadata_findings": PRODUCTION_METADATA_LOCK,
        "observed": observed,
        "blockers": blockers,
        "warnings": warnings,
        "ready_for_parameter_lock": not blockers,
    }
    return frame, summary


def _timepoint_sort_key(value: str) -> int:
    text = str(value)
    if text.startswith("D") and text[1:].isdigit():
        return int(text[1:])
    return 10_000


def full_input_availability_markdown(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    observed = summary["observed"]
    preview_columns = [
        "slice_id",
        "timepoint",
        "m1_exists",
        "m2_exists",
        "m1_has_xy",
        "m2_has_required_join_keys",
        "m2_rows",
        "m2_columns",
    ]
    preview = frame[[column for column in preview_columns if column in frame.columns]].head(20)
    return dedent(
        f"""
        # Full Input Availability

        - M1 by-slice files: {observed["m1_by_slice_file_count"]}
        - M2 by-slice files: {observed["m2_by_slice_file_count"]}
        - Slice IDs match: {observed["slice_ids_match"]}
        - Total M2 anchors from Parquet metadata: {observed["total_m2_anchors_from_parquet_metadata"]:,}
        - Timepoints: {", ".join(observed["timepoints"])}
        - Required M2 join keys present on all slices: {observed["m2_required_join_keys_present_all_slices"]}
        - M1 x/y present on all slices: {observed["m1_coordinate_columns_present_all_slices"]}
        - Ready for parameter lock: {summary["ready_for_parameter_lock"]}

        This inventory uses file discovery, Parquet metadata, schemas, and one
        timepoint value per M2 file. It does not load the full M1/M2 tables.

        ## Preview

        {dataframe_to_markdown(preview)}
        """
    ).strip() + "\n"


def _strip_m2_scale_prefix(column: str) -> tuple[str | None, str]:
    parts = column.split("__", 1)
    if len(parts) == 2 and parts[0].startswith("radius_"):
        return parts[0], parts[1]
    return None, column


def classify_production_feature_column(column: str, metadata_columns: set[str]) -> str:
    if column in metadata_columns:
        return "metadata"
    _, base = _strip_m2_scale_prefix(column)
    lowered = base.lower()
    if base.startswith(("ct_l1__", "ct_l2__", "ct_l3__")):
        return "niche_composition_features"
    if base.endswith("_entropy"):
        return "entropy_features"
    if base.startswith("emb_mean_pc"):
        return "embedding_mean_features"
    if base.startswith("emb_var_pc"):
        return "embedding_variance_features"
    if base == "n_neighbors":
        return "neighborhood_count_features"
    if any(token in lowered for token in ["distance", "density", "topology"]):
        return "spatial_topology_density_features"
    if any(token in lowered for token in ["fate", "endpoint", "darlin"]):
        return "excluded_leakage_or_deferred_features"
    return "technical_or_unknown_features"


def build_full_feature_lock(
    schema_path: Path = M2_SCHEMA_PATH,
    production_root: Path = PLAN_A_K_PRODUCTION_SCRATCH_ROOT,
    random_seed: int = 271_828,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    schema = load_full_m2_schema(schema_path)
    metadata_columns = set(schema.get("metadata_columns", []))
    output_columns = list(schema.get("output_columns", []))
    safe_groups = {
        "niche_composition_features",
        "entropy_features",
        "embedding_mean_features",
    }
    grouped: dict[str, list[str]] = {}
    selected_features: list[str] = []
    for column in output_columns:
        group = classify_production_feature_column(column, metadata_columns)
        grouped.setdefault(group, []).append(column)
        if group in safe_groups:
            selected_features.append(column)

    rows = []
    for group, columns in sorted(grouped.items()):
        rows.append(
            {
                "feature_group": group,
                "column_count": len(columns),
                "selected_for_safe_feature_lock": group in safe_groups,
                "selected_column_count": len(columns) if group in safe_groups else 0,
                "example_columns": ";".join(columns[:8]),
            }
        )
    frame = pd.DataFrame(rows)
    warnings = []
    if len(selected_features) != PRODUCTION_METADATA_LOCK["safe_feature_mode_column_count"]:
        warnings.append(
            "Safe feature count differs from the locked metadata-only finding "
            f"({len(selected_features)} observed)."
        )

    feature_lock_config = {
        "config_version": "planA_k_full_m2_5_feature_lock_draft_v1",
        "dry_run_draft": True,
        "feature_mode": "safe",
        "random_seed": random_seed,
        "schema_path": str(schema_path),
        "metadata_columns": list(schema.get("metadata_columns", [])),
        "feature_columns": selected_features,
        "feature_column_count": len(selected_features),
        "excluded_feature_policy": {
            "metadata": "preserve for joins, labels, QC, and annotation; do not cluster on directly",
            "embedding_variance_and_density": "defer until full Kmix_A QC confirms no domination",
            "fate_endpoint_darlin": "exclude from M2.5 coarsening",
        },
        "training_sample_manifest": {
            "required": True,
            "path": str(production_root / "full_m2_5" / "training_sample_manifest.tsv"),
            "policy": "stratified_by_timepoint_slice_with_D35_included",
        },
        "scaler_lock": {
            "required": True,
            "path": str(production_root / "full_m2_5" / "scaler.joblib"),
        },
        "pca_lock": {
            "required": True,
            "path": str(production_root / "full_m2_5" / "pca.joblib"),
            "components_path": str(production_root / "full_m2_5" / "pca_components.npy"),
        },
        "software_environment_record": {
            "required": True,
            "path": str(production_root / "full_m2_5" / "software_environment.json"),
        },
        "reproducibility_requirements": REPRODUCIBILITY_REQUIREMENTS,
    }
    payload = {
        "generated_at_utc": utc_now(),
        "schema": schema,
        "feature_mode": "safe",
        "safe_feature_column_count": len(selected_features),
        "safe_feature_column_examples": selected_features[:20],
        "warnings": warnings,
        "feature_lock_config": feature_lock_config,
        "rows": frame.to_dict(orient="records"),
    }
    return frame, payload, feature_lock_config


def feature_lock_markdown(frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    config = payload["feature_lock_config"]
    return dedent(
        f"""
        # Full M2.5 Feature Lock Audit

        - Feature mode: `{payload["feature_mode"]}`
        - Safe feature columns: {payload["safe_feature_column_count"]}
        - Metadata columns preserved outside clustering: {len(config["metadata_columns"])}
        - Reproducibility lock requirements: {", ".join(REPRODUCIBILITY_REQUIREMENTS)}

        Safe mode selects neighborhood composition, entropy, and embedding-mean
        features only. Fate, endpoint, DARLIN-derived, metadata, variance, and
        density/topology fields are excluded or deferred for QC.

        {dataframe_to_markdown(frame)}
        """
    ).strip() + "\n"


def adaptive_metaniche_count_estimate(
    total_anchors: int,
    feature_count: int,
    timepoint_anchor_totals: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_scale = max(0.75, min(1.25, feature_count / 600 if feature_count else 1.0))
    target = int(round((math.sqrt(max(total_anchors, 1)) * feature_scale) / 64) * 64)
    target = max(512, min(2048, target))
    lower = max(256, int(round(target * 0.67 / 64) * 64))
    upper = min(4096, int(round(target * 1.5 / 64) * 64))
    rows = []
    for timepoint in PRODUCTION_METADATA_LOCK["timepoints"]:
        anchors = int(timepoint_anchor_totals.get(timepoint, 0))
        per_target = int(round(math.sqrt(max(anchors, 1)) / 16) * 16) if anchors else 0
        rows.append(
            {
                "timepoint": timepoint,
                "anchors_from_metadata": anchors,
                "adaptive_metaniche_count_estimate": max(32, per_target) if anchors else 0,
                "qc_requirement": "D35-specific terminal/sink QC required"
                if timepoint == "D35"
                else "standard timepoint representation QC",
            }
        )
    frame = pd.DataFrame(rows)
    payload = {
        "total_anchors": total_anchors,
        "feature_count": feature_count,
        "target_metaniche_count": target,
        "candidate_metaniche_count_range": [lower, target, upper],
        "timepoint_rows": frame.to_dict(orient="records"),
        "d35_policy": "Include D35 in full production and do not inherit the tiny k=4 D0-D21 pilot interpretation.",
    }
    return frame, payload


def build_full_m2_5_coarsening_strategy(
    input_summary: dict[str, Any],
    feature_lock_payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = input_summary["observed"]
    estimate_frame, estimate_payload = adaptive_metaniche_count_estimate(
        total_anchors=int(observed["total_m2_anchors_from_parquet_metadata"]),
        feature_count=int(feature_lock_payload["safe_feature_column_count"]),
        timepoint_anchor_totals=observed["timepoint_anchor_totals"],
    )
    rows = [
        {
            "stage": "feature_lock",
            "parameter": "feature_mode",
            "draft_value": "safe",
            "rationale": "Uses the locked 600-feature safe mode.",
        },
        {
            "stage": "scaling",
            "parameter": "scaler",
            "draft_value": "StandardScaler persisted under scratch",
            "rationale": "Scaler parameters are required for reproducibility.",
        },
        {
            "stage": "pca",
            "parameter": "pca_components",
            "draft_value": "fit on locked training sample and persist object/components",
            "rationale": "Full production must reuse an exact PCA lock.",
        },
        {
            "stage": "coarsening",
            "parameter": "metaniche_count_candidates",
            "draft_value": ",".join(map(str, estimate_payload["candidate_metaniche_count_range"])),
            "rationale": "Adaptive estimate from anchor count and feature count.",
        },
        {
            "stage": "D35_QC",
            "parameter": "late_timepoint_policy",
            "draft_value": "include_D35",
            "rationale": "D35 is included and must receive dedicated terminal/sink QC.",
        },
    ]
    frame = pd.DataFrame(rows)
    payload = {
        "generated_at_utc": utc_now(),
        "strategy": "full_metadata_locked_safe_feature_metaniche_coarsening",
        "estimate": estimate_payload,
        "timepoint_estimates": estimate_frame.to_dict(orient="records"),
        "d35_qc_requirements": [
            "D35 metaniche count",
            "D21->D35 edge coverage",
            "D35 sink/closed-class behavior",
            "whether D35 dominates late-enriched macrostates",
            "do not inherit tiny k=4 interpretation from D0-D21 pilot",
        ],
        "reproducibility_requirements": REPRODUCIBILITY_REQUIREMENTS,
        "rows": frame.to_dict(orient="records"),
    }
    return frame, payload


def m2_5_strategy_markdown(frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    estimate = payload["estimate"]
    return dedent(
        f"""
        # Full M2.5 Coarsening Strategy

        - Target metaniche count estimate: {estimate["target_metaniche_count"]}
        - Candidate count range: {", ".join(map(str, estimate["candidate_metaniche_count_range"]))}
        - D35 policy: {estimate["d35_policy"]}

        ## Draft Strategy Parameters

        {dataframe_to_markdown(frame)}

        ## Timepoint Estimates

        {dataframe_to_markdown(pd.DataFrame(payload["timepoint_estimates"]))}
        """
    ).strip() + "\n"


def bounded_coordinate_join_audit(
    m1_root: Path = M1_BY_SLICE_ROOT,
    m2_root: Path = M2_BY_SLICE_ROOT,
    sample_rows_per_slice: int = 256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    join_keys = PRODUCTION_METADATA_LOCK["required_join_keys"]
    columns_m2 = [*join_keys, "time", "time_day", "mouse_id"]
    columns_m1 = [*join_keys, "scale", "x", "y"]
    m2_files = {path.parent.name: path for path in sorted(m2_root.glob("*/m2_representation_*.parquet"))}
    for slice_id, m2_path in m2_files.items():
        m1_path = m1_root / slice_id / f"niche_features_{slice_id}.parquet"
        row = {
            "slice_id": slice_id,
            "m2_sample_rows": 0,
            "m1_sample_rows": 0,
            "joined_rows": 0,
            "join_coverage": 0.0,
            "m2_duplicate_key_rows": 0,
            "m1_duplicate_key_rows": 0,
            "status": "unchecked",
            "note": "",
        }
        try:
            m2_sample = _read_parquet_head(m2_path, columns_m2, sample_rows_per_slice)
            m1_sample = _read_parquet_head(m1_path, columns_m1, max(sample_rows_per_slice * 6, 1024))
            if "scale" in m1_sample.columns:
                m1_sample = m1_sample[m1_sample["scale"].astype(str) == "radius_x2"].copy()
            m1_sample = m1_sample.head(sample_rows_per_slice)
            m2_keys = m2_sample[join_keys].dropna().copy()
            m1_keys = m1_sample[[*join_keys, "x", "y"]].dropna().copy()
            joined = m2_keys.merge(m1_keys, on=join_keys, how="left", indicator=True)
            coverage = float((joined["_merge"] == "both").mean()) if len(joined) else 0.0
            row.update(
                {
                    "m2_sample_rows": int(len(m2_keys)),
                    "m1_sample_rows": int(len(m1_keys)),
                    "joined_rows": int((joined["_merge"] == "both").sum()) if len(joined) else 0,
                    "join_coverage": coverage,
                    "m2_duplicate_key_rows": int(m2_keys.duplicated(join_keys).sum()),
                    "m1_duplicate_key_rows": int(m1_keys.duplicated(join_keys).sum()),
                    "status": "sample_pass" if coverage >= 0.999 else "sample_warning",
                    "note": "bounded key/coordinate sample only; full production validation still required",
                }
            )
        except Exception as exc:
            row.update({"status": "sample_unavailable", "note": str(exc)})
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {
        "generated_at_utc": utc_now(),
        "sample_rows_per_slice": sample_rows_per_slice,
        "slice_count_sampled": int(len(frame)),
        "min_sample_join_coverage": float(frame["join_coverage"].min()) if not frame.empty else 0.0,
        "sample_duplicate_key_rows": int(
            frame[["m2_duplicate_key_rows", "m1_duplicate_key_rows"]].sum().sum()
        )
        if not frame.empty
        else 0,
        "preflight_scope": "bounded sample join check only",
    }
    return frame, summary


def build_coordinate_join_contract(
    bounded_frame: pd.DataFrame,
    bounded_summary: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = pd.DataFrame(
        [
            {
                "contract_item": "join_key",
                "requirement": "+".join(PRODUCTION_METADATA_LOCK["required_join_keys"]),
                "blocking_rule": "duplicate keys block production",
            },
            {
                "contract_item": "coordinate_columns",
                "requirement": "x+y from M1 by-slice niche_features parquet",
                "blocking_rule": "missing coordinate columns block production",
            },
            {
                "contract_item": "coverage_threshold",
                "requirement": "full production join coverage >= 99.9%",
                "blocking_rule": "coverage below 99.9% blocks production",
            },
            {
                "contract_item": "validation_timing",
                "requirement": "full validation across all slices before coarsening",
                "blocking_rule": "M2.5 cannot proceed before full validation passes",
            },
        ]
    )
    payload = {
        "generated_at_utc": utc_now(),
        "recommended_join_key": PRODUCTION_METADATA_LOCK["required_join_keys"],
        "primary_coordinate_source": str(M1_BY_SLICE_ROOT),
        "preflight_bounded_sample": bounded_summary,
        "preflight_bounded_sample_rows": bounded_frame.to_dict(orient="records"),
        "production_blocking_rules": {
            "duplicate_join_keys_block": True,
            "minimum_join_coverage": 0.999,
            "full_join_validation_before_coarsening": True,
        },
        "rows": rows.to_dict(orient="records"),
    }
    return rows, payload


def evaluate_coordinate_join_validation(
    join_coverage: float,
    duplicate_join_key_rows: int,
    minimum_coverage: float = 0.999,
) -> dict[str, Any]:
    blockers = []
    if duplicate_join_key_rows > 0:
        blockers.append("duplicate join keys")
    if join_coverage < minimum_coverage:
        blockers.append("join coverage below 99.9%")
    return {
        "join_coverage": float(join_coverage),
        "duplicate_join_key_rows": int(duplicate_join_key_rows),
        "minimum_coverage": float(minimum_coverage),
        "production_blocked": bool(blockers),
        "blockers": blockers,
    }


def coordinate_join_contract_markdown(frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    sample = payload["preflight_bounded_sample"]
    return dedent(
        f"""
        # Full Coordinate Join Contract

        - Recommended key: `{"+".join(payload["recommended_join_key"])}`
        - Primary coordinate source: `{payload["primary_coordinate_source"]}`
        - Bounded sample minimum coverage: {sample["min_sample_join_coverage"]:.6f}
        - Production blocking coverage threshold: 99.9%

        The preflight may use bounded join checks. Full production M2.5 must
        perform full join validation across all slices before coarsening.

        {dataframe_to_markdown(frame)}
        """
    ).strip() + "\n"


def build_full_kmix_A_config(
    production_root: Path = PLAN_A_K_PRODUCTION_SCRATCH_ROOT,
    report_root: Path = PRODUCTION_PREFLIGHT_ROOT,
) -> dict[str, Any]:
    timepoints = PRODUCTION_METADATA_LOCK["timepoints"]
    return {
        "config_version": "planA_k_full_kmix_A_draft_v1",
        "dry_run_draft": True,
        "timepoints": timepoints,
        "forward_edges": [
            {"source": source, "target": target}
            for source, target in zip(timepoints[:-1], timepoints[1:])
        ],
        "include_d35": True,
        "forward_top_k": {
            "candidate": 20,
            "qc_grid": [10, 20, 30],
            "locked_final_value": None,
            "selection_rule": "select final top_k only after full Kmix_A QC",
        },
        "mixture_weights": {
            "forward": 0.85,
            "within_time": 0.10,
            "regularization": 0.05,
            "status": "draft_from_pilot_stabilized_Kmix_A",
        },
        "inputs": {
            "full_m2_5_root": str(production_root / "full_m2_5"),
            "feature_lock_config": str(PROJECT_ROOT / "configs" / "planA_k" / "full_m2_5_feature_lock.draft.json"),
            "coordinate_join_contract": str(report_root / "04_full_coordinate_join_contract.json"),
        },
        "outputs": {
            "production_kmix_root": str(production_root / "full_kmix_A"),
            "lightweight_qc_report_root": str(report_root / "full_kmix_A_qc"),
        },
        "d35_qc_requirements": [
            "D35 metaniche count",
            "D21->D35 edge coverage",
            "D35 sink/closed-class behavior",
            "whether D35 dominates late-enriched macrostates",
            "do not inherit tiny k=4 interpretation from D0-D21 pilot",
        ],
    }


def kmix_A_plan_markdown(config: dict[str, Any]) -> str:
    edge_text = ", ".join(f"{edge['source']}->{edge['target']}" for edge in config["forward_edges"])
    return dedent(
        f"""
        # Full Kmix_A Construction Plan

        - Timepoints: {", ".join(config["timepoints"])}
        - Forward edges: {edge_text}
        - `forward_top_k=20` status: production candidate, not final lock
        - QC grid: {", ".join(map(str, config["forward_top_k"]["qc_grid"]))}
        - D35 included: {config["include_d35"]}

        Full Kmix_A must run after full M2.5 and full M2.5 QC. The final
        forward top-k is selected only after full Kmix_A QC.
        """
    ).strip() + "\n"


def build_gpcca_feasibility(
    strategy_payload: dict[str, Any],
    input_summary: dict[str, Any],
) -> dict[str, Any]:
    states = int(strategy_payload["estimate"]["target_metaniche_count"])
    top_k_candidate = 20
    within_k = 10
    estimated_sparse_nnz = int(states * (top_k_candidate + within_k + 1))
    dense_workspace_gib = round((states * states * 8 * 4) / (1024**3), 3)
    memory = read_memory_info()
    available_gib = float(memory.get("mem_available_gib", 0.0)) if memory.get("available") else 0.0
    blockers = list(input_summary.get("blockers", []))
    if blockers:
        decision = "NOT_READY_MAJOR_BLOCKERS"
    elif available_gib and dense_workspace_gib > max(1.0, available_gib * 0.5):
        decision = "NEEDS_SCALE_CONTROLLED_FALLBACK"
    else:
        decision = "DIRECT_FULL_RUN_READY_WITH_RESOURCE_CAUTION"
    return {
        "generated_at_utc": utc_now(),
        "decision_label": decision,
        "estimated_state_count": states,
        "forward_top_k_candidate": top_k_candidate,
        "within_time_top_k_assumption": within_k,
        "estimated_sparse_nnz": estimated_sparse_nnz,
        "estimated_dense_workspace_gib": dense_workspace_gib,
        "memory": memory,
        "resource_caution": [
            "Full GPCCA is not run by this preflight.",
            "Estimate sparse nnz and potential dense workspace before execution.",
            "If local dense workspace risk increases after full Kmix_A QC, use a scale-controlled fallback or Slurm-style execution plan.",
        ],
        "blockers": blockers,
    }


def gpcca_feasibility_markdown(payload: dict[str, Any]) -> str:
    return dedent(
        f"""
        # Full GPCCA Feasibility

        - Decision label: `{payload["decision_label"]}`
        - Estimated metaniche states: {payload["estimated_state_count"]}
        - Estimated sparse nnz: {payload["estimated_sparse_nnz"]:,}
        - Estimated dense workspace: {payload["estimated_dense_workspace_gib"]} GiB

        Full GPCCA is intentionally not executed in this preflight. If resource
        risk increases after full Kmix_A QC, use a scale-controlled fallback or
        Slurm-style execution plan rather than running an uncontrolled local job.
        """
    ).strip() + "\n"


def build_full_run_blueprint(
    decision_label: str,
    production_root: Path = PLAN_A_K_PRODUCTION_SCRATCH_ROOT,
    report_root: Path = PRODUCTION_PREFLIGHT_ROOT,
) -> dict[str, Any]:
    if decision_label not in DECISION_LABELS:
        raise ValueError(f"Invalid decision label: {decision_label}")
    full_m2_5_root = production_root / "full_m2_5"
    full_kmix_root = production_root / "full_kmix_A"
    full_gpcca_root = production_root / "full_gpcca"
    macrostate_root = production_root / "full_macrostate_annotation"
    commands = [
        {
            "order": 1,
            "phase": "full M2.5",
            "command": (
                "conda run --no-capture-output -n omicverse python "
                "scripts/planA_k_23_run_full_m2_5_production.py "
                "--feature-lock configs/planA_k/full_m2_5_feature_lock.draft.json "
                f"--output-root {full_m2_5_root} --seed 271828"
            ),
            "production_output_root": str(full_m2_5_root),
            "lightweight_report_root": str(report_root / "full_m2_5_qc"),
        },
        {
            "order": 2,
            "phase": "full M2.5 QC",
            "command": (
                "conda run --no-capture-output -n omicverse python "
                "scripts/planA_k_24_full_m2_5_qc.py "
                f"--m2-5-root {full_m2_5_root} "
                "--coordinate-join-contract reports/planA_k_production_preflight/04_full_coordinate_join_contract.json "
                f"--output-dir {report_root / 'full_m2_5_qc'}"
            ),
            "production_output_root": str(full_m2_5_root),
            "lightweight_report_root": str(report_root / "full_m2_5_qc"),
        },
        {
            "order": 3,
            "phase": "full Kmix_A",
            "command": (
                "conda run --no-capture-output -n omicverse python "
                "scripts/planA_k_25_build_full_kmix_A.py "
                "--config configs/planA_k/full_kmix_A.draft.yaml "
                f"--output-root {full_kmix_root}"
            ),
            "production_output_root": str(full_kmix_root),
            "lightweight_report_root": str(report_root / "full_kmix_A_qc"),
        },
        {
            "order": 4,
            "phase": "full kernel QC",
            "command": (
                "conda run --no-capture-output -n omicverse python "
                "scripts/planA_k_26_full_kernel_qc.py "
                f"--kmix-root {full_kmix_root} --output-dir {report_root / 'full_kmix_A_qc'}"
            ),
            "production_output_root": str(full_kmix_root),
            "lightweight_report_root": str(report_root / "full_kmix_A_qc"),
        },
        {
            "order": 5,
            "phase": "full GPCCA",
            "command": (
                "conda run --no-capture-output -n omicverse python "
                "scripts/planA_k_27_run_full_gpcca.py "
                f"--kmix-root {full_kmix_root} --output-root {full_gpcca_root}"
            ),
            "production_output_root": str(full_gpcca_root),
            "lightweight_report_root": str(report_root / "full_gpcca_qc"),
        },
        {
            "order": 6,
            "phase": "full macrostate annotation",
            "command": (
                "conda run --no-capture-output -n omicverse python "
                "scripts/planA_k_28_annotate_full_macrostates.py "
                f"--gpcca-root {full_gpcca_root} --output-root {macrostate_root} "
                f"--report-dir {report_root / 'full_macrostate_annotation'}"
            ),
            "production_output_root": str(macrostate_root),
            "lightweight_report_root": str(report_root / "full_macrostate_annotation"),
        },
    ]
    return {
        "generated_at_utc": utc_now(),
        "decision_label": decision_label,
        "preflight_only": True,
        "commands_execute_in_preflight": False,
        "production_order": PRODUCTION_ORDER,
        "lightweight_report_root": str(report_root),
        "production_output_root": str(production_root),
        "next_safe_command": commands[0]["command"],
        "commands": commands,
        "output_path_policy": {
            "repo_reports_and_configs_allowed": True,
            "production_matrices_parquet_kmix_gpcca_under_scratch": True,
            "ssd_outputs_allowed": False,
        },
    }


def blueprint_markdown(payload: dict[str, Any]) -> str:
    rows = pd.DataFrame(payload["commands"])
    return dedent(
        f"""
        # Full Production Run Blueprint

        - Decision label: `{payload["decision_label"]}`
        - Commands execute in preflight: {payload["commands_execute_in_preflight"]}
        - Production output root: `{payload["production_output_root"]}`
        - Lightweight report root: `{payload["lightweight_report_root"]}`
        - Next safe command: `{payload["next_safe_command"]}`

        The order is fixed: {" -> ".join(payload["production_order"])}.

        {dataframe_to_markdown(rows[["order", "phase", "command", "production_output_root", "lightweight_report_root"]])}
        """
    ).strip() + "\n"


def build_preflight_payload(key_files: list[Path] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if key_files is None:
        key_files = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "reproducibility_guide.md",
            M2_SCHEMA_PATH,
            M1_BY_SLICE_ROOT,
            M2_BY_SLICE_ROOT,
        ]
    inventory = pd.DataFrame([file_summary(path) for path in key_files])
    payload = {
        "generated_at_utc": utc_now(),
        "environment": collect_production_environment_payload(),
        "key_file_inventory": inventory.to_dict(orient="records"),
        "scope": {
            "goal": "PlanA-K full production preflight and parameter lock.",
            "preflight_only": True,
            "forbidden_actions": FORBIDDEN_ACTIONS,
        },
    }
    return inventory, payload


def preflight_markdown(inventory: pd.DataFrame, payload: dict[str, Any]) -> str:
    env = payload["environment"]
    return dedent(
        f"""
        # PlanA-K Production Preflight

        - Generated at: {payload["generated_at_utc"]}
        - Git branch: {env["git_branch"]}
        - Git status entries: {len(env["git_status_short"])}
        - Preflight only: {payload["scope"]["preflight_only"]}
        - Uses /ssd outputs: {env["guardrails"]["uses_ssd_outputs"]}

        ## Key File Inventory

        {dataframe_to_markdown(inventory)}
        """
    ).strip() + "\n"


def summary_markdown(payload: dict[str, Any]) -> str:
    return dedent(
        f"""
        # PlanA-K Production Preflight Summary

        - Decision label: `{payload["decision_label"]}`
        - Next safe command: `{payload["next_safe_command"]}`
        - M1 files: {payload["input_summary"]["observed"]["m1_by_slice_file_count"]}
        - M2 files: {payload["input_summary"]["observed"]["m2_by_slice_file_count"]}
        - M2 anchors: {payload["input_summary"]["observed"]["total_m2_anchors_from_parquet_metadata"]:,}
        - Safe feature columns: {payload["feature_summary"]["safe_feature_column_count"]}
        - D35 included in Kmix planning: {payload["kmix_config"]["include_d35"]}
        - Preflight executed production work: False

        Full order remains: {" -> ".join(PRODUCTION_ORDER)}.
        """
    ).strip() + "\n"


def _render_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in [":", "#", "{", "}", "[", "]", ","]):
        return json.dumps(text)
    return text


def render_simple_yaml(data: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(render_simple_yaml(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_render_yaml_scalar(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for value in data:
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(render_simple_yaml(value, indent + 2))
            else:
                lines.append(f"{prefix}- {_render_yaml_scalar(value)}")
        return "\n".join(lines)
    return f"{prefix}{_render_yaml_scalar(data)}"


def write_production_preflight_outputs(
    output_dir: Path = PRODUCTION_PREFLIGHT_ROOT,
    config_dir: Path = PROJECT_ROOT / "configs" / "planA_k",
    overwrite: bool = False,
    dry_run: bool = True,
    bounded_join_sample_rows: int = 256,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    config_dir = ensure_dir(config_dir)

    recovery_payload = build_recovery_note_payload()
    atomic_write_text(output_dir / "00_recovery_note.md", recovery_note_markdown(recovery_payload), overwrite=overwrite)
    atomic_write_json(output_dir / "00_recovery_note.json", recovery_payload, overwrite=overwrite)

    preflight_inventory, preflight_payload = build_preflight_payload()
    preflight_payload["dry_run"] = dry_run
    atomic_write_text(output_dir / "00_preflight.md", preflight_markdown(preflight_inventory, preflight_payload), overwrite=overwrite)
    atomic_write_json(output_dir / "00_preflight.json", preflight_payload, overwrite=overwrite)

    input_frame, input_summary = discover_full_input_availability()
    atomic_write_text(output_dir / "01_full_input_availability.md", full_input_availability_markdown(input_frame, input_summary), overwrite=overwrite)
    atomic_write_tsv(output_dir / "01_full_input_availability.tsv", input_frame, overwrite=overwrite)
    atomic_write_json(output_dir / "01_full_input_availability.json", {"summary": input_summary, "rows": input_frame.to_dict(orient="records")}, overwrite=overwrite)

    feature_frame, feature_payload, feature_config = build_full_feature_lock()
    atomic_write_text(output_dir / "02_feature_lock_audit.md", feature_lock_markdown(feature_frame, feature_payload), overwrite=overwrite)
    atomic_write_tsv(output_dir / "02_feature_lock_audit.tsv", feature_frame, overwrite=overwrite)
    atomic_write_json(output_dir / "02_feature_lock_audit.json", feature_payload, overwrite=overwrite)
    atomic_write_json(config_dir / "full_m2_5_feature_lock.draft.json", feature_config, overwrite=overwrite)

    strategy_frame, strategy_payload = build_full_m2_5_coarsening_strategy(input_summary, feature_payload)
    atomic_write_text(output_dir / "03_full_m2_5_coarsening_strategy.md", m2_5_strategy_markdown(strategy_frame, strategy_payload), overwrite=overwrite)
    atomic_write_tsv(output_dir / "03_full_m2_5_coarsening_strategy.tsv", strategy_frame, overwrite=overwrite)
    atomic_write_json(output_dir / "03_full_m2_5_coarsening_strategy.json", strategy_payload, overwrite=overwrite)

    bounded_frame, bounded_summary = bounded_coordinate_join_audit(sample_rows_per_slice=bounded_join_sample_rows)
    contract_frame, contract_payload = build_coordinate_join_contract(bounded_frame, bounded_summary)
    atomic_write_text(output_dir / "04_full_coordinate_join_contract.md", coordinate_join_contract_markdown(contract_frame, contract_payload), overwrite=overwrite)
    atomic_write_tsv(output_dir / "04_full_coordinate_join_contract.tsv", contract_frame, overwrite=overwrite)
    atomic_write_json(output_dir / "04_full_coordinate_join_contract.json", contract_payload, overwrite=overwrite)

    kmix_config = build_full_kmix_A_config()
    atomic_write_text(output_dir / "05_full_kmix_A_construction_plan.md", kmix_A_plan_markdown(kmix_config), overwrite=overwrite)
    atomic_write_json(output_dir / "05_full_kmix_A_construction_plan.json", kmix_config, overwrite=overwrite)
    atomic_write_text(config_dir / "full_kmix_A.draft.yaml", render_simple_yaml(kmix_config) + "\n", overwrite=overwrite)

    gpcca_payload = build_gpcca_feasibility(strategy_payload, input_summary)
    atomic_write_text(output_dir / "06_full_gpcca_feasibility.md", gpcca_feasibility_markdown(gpcca_payload), overwrite=overwrite)
    atomic_write_json(output_dir / "06_full_gpcca_feasibility.json", gpcca_payload, overwrite=overwrite)

    blueprint_payload = build_full_run_blueprint(gpcca_payload["decision_label"])
    atomic_write_text(output_dir / "07_full_production_run_blueprint.md", blueprint_markdown(blueprint_payload), overwrite=overwrite)
    atomic_write_json(output_dir / "07_full_production_run_blueprint.json", blueprint_payload, overwrite=overwrite)

    summary_payload = {
        "generated_at_utc": utc_now(),
        "decision_label": gpcca_payload["decision_label"],
        "next_safe_command": blueprint_payload["next_safe_command"],
        "input_summary": input_summary,
        "feature_summary": {
            "safe_feature_column_count": feature_payload["safe_feature_column_count"],
            "warnings": feature_payload["warnings"],
        },
        "strategy_summary": strategy_payload["estimate"],
        "coordinate_join_summary": contract_payload["preflight_bounded_sample"],
        "kmix_config": kmix_config,
        "gpcca_feasibility": gpcca_payload,
        "blueprint": blueprint_payload,
        "guardrail_confirmation": {
            "raw_data_processed": False,
            "darlin_data_processed": False,
            "frozen_p_fate_modified": False,
            "ssd_outputs_used": False,
            "slurm_submitted": False,
            "full_m2_5_run": False,
            "full_gpcca_run": False,
            "branchsbm_trained": False,
            "git_write_actions": False,
        },
    }
    atomic_write_text(output_dir / "00_PRODUCTION_PREFLIGHT_SUMMARY.md", summary_markdown(summary_payload), overwrite=overwrite)
    atomic_write_json(output_dir / "00_PRODUCTION_PREFLIGHT_SUMMARY.json", summary_payload, overwrite=overwrite)
    return summary_payload


def write_full_run_blueprint_outputs(
    output_dir: Path = PRODUCTION_PREFLIGHT_ROOT,
    overwrite: bool = False,
    decision_label: str = "DIRECT_FULL_RUN_READY_WITH_RESOURCE_CAUTION",
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    payload = build_full_run_blueprint(decision_label=decision_label)
    atomic_write_text(output_dir / "07_full_production_run_blueprint.md", blueprint_markdown(payload), overwrite=overwrite)
    atomic_write_json(output_dir / "07_full_production_run_blueprint.json", payload, overwrite=overwrite)
    return payload


__all__ = [name for name in globals() if not name.startswith("__")]
