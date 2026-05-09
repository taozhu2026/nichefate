#!/usr/bin/env python
"""Define the M3 transition evidence and sampled-kernel contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import load_config
from nichefate.transition import evidence_schema_columns, resolve_transition_feature_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_transition_kernel.yaml")
    return parser.parse_args()


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: Path(value)
        for key, value in config["paths"].items()
        if isinstance(value, str) and value.startswith("/")
    }


def _write_contract(
    path: Path,
    config: dict[str, Any],
    groups: dict[str, list[str]],
    time_pairs: list[dict[str, Any]],
) -> None:
    edge_cfg = config["candidate_edges"]
    expected_edges = (
        len(time_pairs)
        * int(edge_cfg["max_source_niches_per_pair"])
        * int(edge_cfg["k_candidates"])
    )
    retrieval = edge_cfg["retrieval_feature_groups"]
    rerank = config["cost"]["rerank_feature_groups"]
    lines = [
        "# M3 Transition Evidence Contract",
        "",
        "M3 is a direction-aware transition evidence layer. It is not a full",
        "Markov kernel, GPCCA run, fate-probability run, or model-training step.",
        "",
        "## Supported Downstream Consumers",
        "",
        "- Markov-GPCCA baseline: consumes future row-stochastic directed kernels.",
        "- Branched NicheFlow: consumes weighted source-target niche pairs.",
        "- Current mode: pseudo-lineage evidence from time, state continuity, composition, spatial summaries, and topology.",
        "- Future mode: barcode-derived lineage evidence can populate the same evidence schema.",
        "",
        "## Candidate Retrieval And Cost",
        "",
        f"- Neighbor backend used in this stage: `{edge_cfg['neighbor_backend']}`.",
        f"- Future backend placeholders: {', '.join(edge_cfg['future_supported_backends'])}.",
        f"- KNN retrieval feature groups: {', '.join(retrieval)}.",
        f"- Rerank/cost feature groups: {', '.join(rerank)}.",
        "- Feature matrices are standardized before KNN retrieval using combined sampled source/target column statistics.",
        "- `combined_cost` is computed only from scaled evidence columns.",
        "- Spatial summary and topology evidence are used for reranking/cost, not the default initial retrieval feature space.",
        f"- Candidate K: {edge_cfg['k_candidates']}.",
        f"- Expected sampled edge count upper bound: {expected_edges}.",
        "",
        "## Direction And Sampling",
        "",
        "- `time_delta` is required and inferred from metadata-derived time-day values.",
        "- Candidate construction is sample-aware and metadata-preserving.",
        "- v1 is not longitudinal sample-paired; source and target mouse/sample ids are preserved but not assumed to be true pairs.",
        "",
        "## Probabilities And Coupling Weights",
        "",
        "- `row_normalized_transition_prob` is a local candidate-set transition probability, not the full global Markov transition matrix P.",
        "- `raw_edge_weight` and `mass_adjusted_weight` are preserved for future unbalanced transport and Branched NicheFlow pseudo-pair supervision.",
        "- v1 mass, growth, and unbalanced terms are neutral placeholders and do not claim proliferation or apoptosis modeling.",
        "- Within-time regularization is a contract placeholder/diagnostic in v1.",
        "",
        "## Feature Group Sizes",
        "",
    ]
    for group, columns in groups.items():
        lines.append(f"- {group}: {len(columns)} columns")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    with paths["m2_schema"].open("r", encoding="utf-8") as handle:
        m2_schema = json.load(handle)
    with (reports_dir / "m3_time_pairs.json").open("r", encoding="utf-8") as handle:
        time_pairs = json.load(handle)

    groups = resolve_transition_feature_groups(m2_schema, config["feature_groups"])
    feature_payload = {
        "feature_groups": groups,
        "retrieval_feature_groups": config["candidate_edges"]["retrieval_feature_groups"],
        "rerank_feature_groups": config["cost"]["rerank_feature_groups"],
        "retrieval_feature_columns": [
            column
            for group in config["candidate_edges"]["retrieval_feature_groups"]
            for column in groups[group]
        ],
        "rerank_feature_columns": {
            group: groups[group] for group in config["cost"]["rerank_feature_groups"]
        },
    }
    feature_path = reports_dir / "m3_feature_groups.json"
    feature_path.write_text(json.dumps(feature_payload, indent=2) + "\n", encoding="utf-8")

    schema_payload = {
        "evidence_columns": evidence_schema_columns(),
        "evidence_mode_values": ["pseudo_lineage", "barcode_supervised_future"],
        "combined_cost_rule": "scaled evidence columns only",
        "candidate_probability_scope": "local candidate set, not full global transition matrix",
    }
    evidence_path = reports_dir / "m3_evidence_schema.json"
    evidence_path.write_text(json.dumps(schema_payload, indent=2) + "\n", encoding="utf-8")

    contract_path = reports_dir / "m3_transition_contract.md"
    _write_contract(contract_path, config, groups, time_pairs)
    print(f"Wrote transition contract: {contract_path}")
    print(f"Wrote feature groups: {feature_path}")
    print(f"Wrote evidence schema: {evidence_path}")
    print(f"FEATURE_GROUPS {len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
