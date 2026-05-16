"""Rare-state preservation audit helpers for PlanA-K metaniches."""

from __future__ import annotations

import getpass
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse import csgraph

from .schemas import *
from .io import *
from .reporting import *
from .validation import *
from .kernel_qc import *
from .coordinates import *


def rare_state_preservation_audit(
    pilot_root: Path = PILOT_OUTPUT_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchor_map = load_pilot_anchor_map(pilot_root)
    if anchor_map.empty:
        frame = pd.DataFrame(
            [
                {
                    "label_column": "none",
                    "label_value": "none",
                    "label_count": 0,
                    "metaniche_count_with_label": 0,
                    "largest_metaniche_fraction": 0.0,
                    "collapsed_warning": True,
                    "status": "BLOCKED",
                }
            ]
        )
        return frame, {"rare_state_audit_available": False, "reason": "pilot anchor map missing"}
    size_by_metaniche = anchor_map["metaniche_id"].value_counts()
    median_size = float(size_by_metaniche.median()) if not size_by_metaniche.empty else 0.0
    rows: list[dict[str, Any]] = []
    for label_column in ["cell_type_l1", "cell_type_l2", "cell_type_l3"]:
        if label_column not in anchor_map.columns:
            continue
        counts = anchor_map[label_column].astype(str).value_counts()
        rare_threshold = max(10, int(np.ceil(0.01 * len(anchor_map))))
        for label, count in counts[counts < rare_threshold].items():
            subset = anchor_map[anchor_map[label_column].astype(str) == label]
            by_mn = subset["metaniche_id"].value_counts()
            if by_mn.empty:
                continue
            largest_id = by_mn.index[0]
            largest_count = int(by_mn.iloc[0])
            largest_fraction = float(largest_count / count)
            largest_size = int(size_by_metaniche.get(largest_id, 0))
            collapsed = largest_fraction >= 0.75 and largest_size > max(20, 2 * median_size)
            rows.append(
                {
                    "label_column": label_column,
                    "label_value": label,
                    "label_count": int(count),
                    "rare_threshold": int(rare_threshold),
                    "metaniche_count_with_label": int(by_mn.size),
                    "largest_metaniche_id": largest_id,
                    "largest_metaniche_count": largest_count,
                    "largest_metaniche_fraction": largest_fraction,
                    "largest_metaniche_size": largest_size,
                    "collapsed_warning": bool(collapsed),
                    "status": "WARN" if collapsed else "PASS",
                }
            )
    frame = pd.DataFrame(rows)
    payload = {
        "generated_at_utc": utc_now(),
        "rare_state_audit_available": True,
        "sampled_anchor_count": int(len(anchor_map)),
        "rare_label_count": int(len(frame)),
        "collapsed_warning_count": int(frame["collapsed_warning"].sum()) if not frame.empty else 0,
        "rare_state_definition": "sampled label count < max(10, 1% of sampled anchors)",
        "biological_claims_allowed": False,
    }
    return frame, payload


def rare_state_audit_markdown(frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    if not payload.get("rare_state_audit_available"):
        return f"# Rare-State Preservation Audit\n\nBlocked: {payload.get('reason')}\n"
    preview = frame.sort_values(["collapsed_warning", "largest_metaniche_fraction"], ascending=[False, False]).head(30) if not frame.empty else frame
    return dedent(
        f"""
        # Rare-State Preservation Audit

        - Sampled anchors: {payload["sampled_anchor_count"]:,}
        - Rare labels: {payload["rare_label_count"]}
        - Collapsed warnings: {payload["collapsed_warning_count"]}
        - Rare-state definition: {payload["rare_state_definition"]}
        - Biological claims allowed: {payload["biological_claims_allowed"]}

        {dataframe_to_markdown(preview)}
        """
    ).strip() + "\n"


__all__ = [name for name in globals() if not name.startswith("__")]
