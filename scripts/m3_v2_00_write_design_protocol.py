#!/usr/bin/env python
"""Write the M3/M4-v2 design justification and M3-v2 pilot protocol.

This is a report-only stage. It reads frozen M2/M3/M4/M4E summaries and writes
design documents under m3_v2_design. It does not build M3-v2 edges, assemble
M4A-v2 matrices, compute M4C-v2 propagation, or modify v1 production outputs.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path("/home/zhutao/scratch/nichefate")
OUT = ROOT / "m3_v2_design"
REPORTS = OUT / "reports"
PILOT_ROOT = ROOT / "m3_v2_pilot"

M2_AUDIT_CSV = ROOT / "m3" / "reports" / "m3_m2_input_audit.csv"
M2_AUDIT_MD = ROOT / "m3" / "reports" / "m3_m2_input_audit.md"
M3_CONTRACT = ROOT / "m3" / "reports" / "m3_transition_contract.md"
M3_FEATURE_GROUPS = ROOT / "m3" / "reports" / "m3_feature_groups.json"
M3_TIME_PAIRS = ROOT / "m3" / "reports" / "m3_time_pairs.json"
M3_FINAL_HANDOFF = ROOT / "m3" / "reports" / "m3_full_m3_final_handoff_to_m4a.md"
M4A_REPORT = ROOT / "m4a" / "reports" / "m4a_assembly_report.md"
M4B_REPORT = ROOT / "m4b" / "reports" / "m4b_terminal_macrostate_design_report.md"
M4C_REPORT = ROOT / "m4c" / "reports" / "m4c_markov_fate_final_review.md"
M4E_FREEZE_REPORT = ROOT / "m4e" / "reports" / "m4c_v1_baseline_dynamic_niche_fate_freeze_report.md"
M4E_NEXT_STEP = ROOT / "m4e" / "reports" / "m4e_next_step_after_endpoint_refinement.md"
M4E_QC_NOTE = ROOT / "m4e" / "reports" / "m4e03_leiden_heatmap_qc_note.md"

CONTEXT_FILES = [
    M2_AUDIT_MD,
    M2_AUDIT_CSV,
    M3_CONTRACT,
    M3_FEATURE_GROUPS,
    M3_TIME_PAIRS,
    M3_FINAL_HANDOFF,
    M4A_REPORT,
    M4B_REPORT,
    M4C_REPORT,
    M4E_FREEZE_REPORT,
    M4E_NEXT_STEP,
    M4E_QC_NOTE,
]


METRICS = [
    {
        "metric_name": "top-target Leiden_neigh consistency",
        "rationale": "Tests whether v2 connects sources to biologically coherent M4E Leiden neighborhoods instead of arbitrary nearest targets.",
        "input_needed": "M3-v1 and M3-v2 top targets joined to M4E node_neighborhood_annotation.leiden_neigh.",
        "preferred_direction": "higher consistency without collapse",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "top-target fine cell cluster consistency",
        "rationale": "Checks whether target choices preserve plausible fine cell identity while still allowing state transitions.",
        "input_needed": "Top target anchors joined to M4E or M2 fine cell cluster metadata.",
        "preferred_direction": "higher or biologically explainable shifts",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "source-target refined endpoint plausibility",
        "rationale": "Uses frozen M4E endpoint refinement to test whether transitions point toward plausible endpoint-attraction structure.",
        "input_needed": "Source and target dominant/refined endpoint annotations from M4E and M4C-v1.",
        "preferred_direction": "higher plausibility with preserved mixed/rare endpoint traceability",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "transition entropy / top1 concentration",
        "rationale": "Detects whether v2 becomes too diffuse or collapses into a single high-probability target.",
        "input_needed": "Per-source M3-v1 and M3-v2 local transition probabilities.",
        "preferred_direction": "moderate entropy and non-degenerate top1 probability",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "spatial smoothness of predicted dominant endpoint",
        "rationale": "Tests whether nearby source niches have coherent predicted dominant endpoints without forcing spatial over-smoothing.",
        "input_needed": "Source coordinates, local spatial graph or neighborhood assignments, predicted dominant endpoint.",
        "preferred_direction": "higher local smoothness without loss of neighborhood separation",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "slice/mouse collapse diagnostics",
        "rationale": "Directly targets a known M3-v1 risk where target mass can concentrate in one slice or mouse.",
        "input_needed": "Source/target slice_id and mouse_id for M3-v1 and M3-v2 edges.",
        "preferred_direction": "lower top slice/mouse fraction and higher target slice/mouse entropy",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "target-neighborhood diversity",
        "rationale": "Tests whether soft gates preserve enough target neighborhood support for biological alternatives.",
        "input_needed": "Target Leiden_neigh distribution per source or source stratum.",
        "preferred_direction": "higher diversity than collapsed v1 cases, bounded by biological plausibility",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "stability across random seeds/subsamples",
        "rationale": "Ensures the pilot result is not an artifact of a particular source subset or random candidate retrieval sample.",
        "input_needed": "Repeated v2 pilot runs over fixed seeds and stratified subsamples.",
        "preferred_direction": "higher agreement across runs",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "endpoint-attraction agreement with M4C-v1",
        "rationale": "Benchmarks v2 against the frozen M4C-v1 endpoint-attraction baseline without assuming identical behavior.",
        "input_needed": "M4C-v1 node summaries and v2 target/endpoint predictions.",
        "preferred_direction": "preserve major M4C-v1 signals while reducing artifacts",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "plasticity enrichment in transition/ulcer/repair-like neighborhoods",
        "rationale": "Checks whether v2 gives interpretable plasticity signal in neighborhoods expected to be dynamic.",
        "input_needed": "M4C-v1 normalized plasticity, M4E Leiden neighborhoods, v2 transition summaries.",
        "preferred_direction": "higher or clearer enrichment in repair-like neighborhoods",
        "required_for_decision": "no",
    },
    {
        "metric_name": "same-anchor-cell-type, different-neighborhood fate separation",
        "rationale": "Tests the central niche-level claim that microenvironment changes can separate fate tendencies within the same cell type.",
        "input_needed": "Cell type, Leiden_neigh, source anchor, and predicted endpoint distribution.",
        "preferred_direction": "stronger separation without cell-type confounding",
        "required_for_decision": "yes",
    },
    {
        "metric_name": "computational runtime/memory",
        "rationale": "Determines whether v2 is feasible before any full production run is considered.",
        "input_needed": "Runtime, peak RSS, edge count, candidate count, and output disk usage.",
        "preferred_direction": "within pilot budget and scalable to full M3 with ANN/sharding",
        "required_for_decision": "yes",
    },
]

DECISION_CATEGORIES = [
    "adopt_v2_for_full_production",
    "revise_v2_and_repeat_pilot",
    "keep_v1_as_main_baseline",
    "keep_v1_and_v2_as_complementary",
    "defer_until_barcode",
]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(name: str, text: str) -> Path:
    path = REPORTS / name
    path.write_text(text.rstrip() + "\n")
    return path


def write_metrics_csv() -> Path:
    path = REPORTS / "m3_v1_vs_v2_benchmark_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric_name",
                "rationale",
                "input_needed",
                "preferred_direction",
                "required_for_decision",
            ],
        )
        writer.writeheader()
        writer.writerows(METRICS)
    return path


def context_inventory() -> list[dict[str, Any]]:
    inventory = []
    for path in CONTEXT_FILES:
        inventory.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return inventory


def choose_pilot_slices(rows: list[dict[str, str]]) -> tuple[list[str], list[str], int]:
    d9 = [row for row in rows if row.get("time") == "D9"]
    d21 = [row for row in rows if row.get("time") == "D21"]
    by_mouse: dict[str, dict[str, str]] = {}
    for row in d9:
        mouse = row.get("mouse_id", "")
        if mouse not in by_mouse or int(row.get("rows", "0")) > int(by_mouse[mouse].get("rows", "0")):
            by_mouse[mouse] = row
    source = [row["slice_id"] for row in sorted(by_mouse.values(), key=lambda value: value["mouse_id"])]
    source_rows = sum(int(row.get("rows", "0")) for row in by_mouse.values())
    target = [row["slice_id"] for row in sorted(d21, key=lambda value: value["slice_id"])]
    return source, target, source_rows


def feature_group_counts(feature_groups: dict[str, Any]) -> dict[str, int]:
    groups = feature_groups.get("feature_groups", {})
    return {name: len(columns) for name, columns in groups.items()}


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def design_justification(
    time_pairs: list[dict[str, Any]],
    counts: dict[str, int],
    source_slices: list[str],
    target_slices: list[str],
    source_rows: int,
) -> str:
    time_pair_summary = ", ".join(
        f"{pair.get('source_time')}->{pair.get('target_time')}" for pair in time_pairs
    )
    feature_summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    return f"""# M3/M4-v2 Design Justification

## Current Frozen v1 Baseline

M1/M2 are stable and are not being redesigned. M2 provides the anchor-centered niche representation consumed by M3. The current M3-v1 layer is a pseudo-only, time-coupled transition edge construction over adjacent time pairs ({time_pair_summary}). It writes candidate source-target edge shards and local row-normalized transition probabilities, not a global Markov object by itself.

M4A-v1 is the corresponding anchor-level sparse Markov matrix assembly from frozen M3-v1 edge shards. M4B/M4C-v1 use candidate endpoint niche clusters and endpoint-attraction / fate-propagation over that v1 matrix. M4E endpoint annotation, refinement, and figure QC make M4C-v1 interpretable enough to keep as a frozen baseline comparator.

M4D/pyGPCCA remains paused and not validated. That failure does not invalidate M4C-v1 because M4C-v1 uses the v1 endpoint-attraction propagation baseline rather than a validated GPCCA macrostate discovery layer.

## Why v2 Is Worth Testing

M3-v2 is motivated by algorithmic rigor and future barcode compatibility, not by a claim that M3-v1 is wrong. M3-v1 is useful and remains the baseline comparator. The v2 pilot asks whether a more modular transition kernel can preserve v1 interpretability while reducing known robustness risks.

Known M3-v1 risks to test:

- Weighted multi-feature cost can be scale-sensitive.
- Molecular state, cell-type composition, and niche composition may be collinear.
- Manual evidence weights can dominate edge probability.
- Future barcode evidence would be hard to interpret if treated as just another weighted distance.
- Current v1 kernel is useful, but needs robustness benchmarking against slice/mouse collapse and target concentration diagnostics.

## M3-v2 Concept

M3-v2 should separate primary state distance from modular soft gates:

```text
P_fate_v2(i -> j) proportional to
  exp(-d_state(i,j) / tau_i)
  * G_time(i,j)
  * G_composition(i,j)
  * G_spatial_topology(i,j)
  * G_slice_mouse(i,j)
  * G_barcode(i,j)
```

The primary distance `d_state` should come from a low-dimensional state embedding derived from M2 representation. Available M3-v1 feature groups include: {feature_summary}. The v2 design should keep anchor-centered niche as the state unit and avoid becoming a cell-level CellRank reimplementation.

## Method Rationale

This is design rationale, not a citation-complete manuscript section. Citation insertion is a future manuscript task.

- Multiview transition evidence should be modular rather than collapsed into one opaque weighted distance.
- CellRank-like kernel/estimator separation motivates distinguishing transition construction from downstream fate or macrostate analysis.
- CoSpar-like state plus lineage coherence motivates future barcode evidence as directionality/coupling evidence rather than post-hoc annotation.
- NicheFlow-like microenvironment modeling supports keeping anchor-centered niche as the state unit.
- M3-v2 should remain niche-level and should not become a cell-level CellRank reimplementation.

## Pilot Scope Recommendation

Recommended small pilot: `D9 -> D21`.

Recommended pilot source slices:

{bullet_list(source_slices)}

Recommended pilot target slices:

{bullet_list(target_slices)}

The selected source set covers multiple D9 samples/mice with about {source_rows:,} source anchors before any optional fixed subsampling. Use `/home/zhutao/scratch/nichefate/m3_v2_pilot/` for future pilot outputs. Do not overwrite M3-v1 outputs.

## M4 Implications

If the pilot succeeds, the next steps are staged: generate M3-v2 pilot edges, assemble an M4A-v2 pilot sparse matrix, optionally run M4C-v2 mini propagation on the pilot, and compare against frozen M4C-v1. Only after review should full M3-v2 -> M4A-v2 -> M4C-v2 production be considered.

K_gpcca is separate from `P_fate_v2`. It should be treated as a later RealTime-like niche kernel pilot and should not block M3-v2 pilot design or implementation.

## Final Recommendation

Implement the M3-v2 small pilot runner and focused tests only. Preserve M3-v1 and M4C-v1 as frozen baselines.
"""


def limitations_report() -> str:
    return """# M3-v1 Limitations And What M3-v2 Tests

## What v1 Already Provides

M3-v1 provides a completed pseudo-only transition evidence layer. Its edge shards are row-normalized per source candidate set, preserve source/target sample metadata, and support M4A-v1 assembly. M4C-v1 and M4E make the endpoint-attraction baseline interpretable enough for benchmarking.

## Limitations To Benchmark, Not Assumed Failures

- Scale sensitivity: v1 collapses multiple scaled evidence distances into one weighted cost.
- Collinearity: molecular state, cell-type composition, and niche composition can encode overlapping biology.
- Weight dominance: manually chosen evidence weights can dominate the probability landscape.
- Slice/mouse artifacts: v1 final QC retained warning-only slice/mouse collapse diagnostics across production shards.
- Barcode extensibility: future DARLIN clone evidence should be directionality/coupling evidence, not another opaque distance term.

## What v2 Tests

M3-v2 tests whether a primary state-cost plus soft-gating design can preserve biological interpretability, reduce obvious sample collapse, avoid target degeneracy, and make future barcode evidence explicit. A successful pilot must compare against v1 on the same source/target scope and must not be adopted simply because it is more sophisticated.
"""


def kernel_definition(counts: dict[str, int]) -> str:
    feature_summary = "\n".join(f"- `{name}`: {count} existing M3-v1 columns" for name, count in sorted(counts.items()))
    return f"""# M3-v2 Kernel Definition

## Kernel Form

M3-v2 is defined as primary state cost plus soft gating:

```text
P_fate_v2(i -> j) proportional to
  exp(-d_state(i,j) / tau_i)
  * G_time(i,j)
  * G_composition(i,j)
  * G_spatial_topology(i,j)
  * G_slice_mouse(i,j)
  * G_barcode(i,j)
```

This report defines the formula only. It does not implement or run it.

## Terms

- `d_state(i,j)`: primary molecular/niche-state distance, preferably from a low-dimensional state embedding derived from M2 representation.
- `tau_i`: source-adaptive temperature or local bandwidth estimated from the local target candidate distance distribution for source `i`.
- `G_time(i,j)`: adjacent-time directionality gate. It is zero or near-zero for invalid time pairs and positive for valid adjacent forward pairs.
- `G_composition(i,j)`: soft compatibility gate for niche/cell-type composition. It should modulate plausible transitions without becoming the main distance.
- `G_spatial_topology(i,j)`: plausibility gate from local spatial/topological features.
- `G_slice_mouse(i,j)`: sample/slice compatibility or anti-collapse gate.
- `G_barcode(i,j)`: neutral in pseudo-only pilot. Reserved for processed DARLIN clone/barcode evidence.

## Required Existing Inputs

- Source and target anchor identifiers, slice IDs, mouse IDs, time labels, and time days.
- M2 representation columns for state, composition, entropy, spatial summary, and topology.
- M4E Leiden neighborhood annotations and refined endpoint mapping for benchmark joins.
- Frozen M3-v1 candidate edges for comparator metrics.
- Frozen M4C-v1 endpoint-attraction summaries for endpoint agreement metrics.

Existing M3-v1 feature group inventory:

{feature_summary}

## Future Barcode Inputs

Future barcode inputs should arrive as processed clone/barcode tables, not raw DARLIN reads. Minimum future fields are cell clone assignment, clone confidence, niche clone composition, clone-supported source-target coupling, and an explicit `G_barcode` or `P_barcode` term.
"""


def pilot_protocol(source_slices: list[str], target_slices: list[str], source_rows: int) -> str:
    return f"""# M3-v2 Small Pilot Protocol

## Scope

Run a conservative `D9 -> D21` pilot only after this design stage. Do not run full M3-v2 production.

Recommended output root for the future pilot:

```text
{PILOT_ROOT}
```

## Inputs

- Existing M2 by-slice representation parquet files.
- Existing M4E neighborhood annotations and refined endpoint labels.
- Frozen M3-v1 D9->D21 edge shards for comparator metrics.
- Frozen M4C-v1 endpoint-attraction / plasticity summaries for benchmarking.

## Source And Target Selection

Recommended D9 source slices:

{bullet_list(source_slices)}

Recommended D21 target slices:

{bullet_list(target_slices)}

The selected D9 source slices contain about {source_rows:,} anchors before optional subsampling. If runtime or memory is constrained, use a fixed stratified subset of 50,000 source anchors. Otherwise target 75,000-100,000 source anchors.

## Pilot Runs

- Run at least two random seeds for candidate retrieval/subsampling.
- Keep pseudo-only mode: `G_barcode = 1`.
- Preserve all source/target slice and mouse metadata.
- Write M3-v2 pilot outputs under `m3_v2_pilot`, never under M3-v1 directories.
- Do not assemble M4A-v2 or compute M4C-v2 unless explicitly approved later.

## Acceptance Checks

- Candidate probabilities are finite, non-negative, and locally row-normalized.
- Target distribution does not collapse into a few targets, slices, or mice.
- Biological interpretability is preserved or improved relative to M3-v1.
- Runtime and peak memory are feasible for a larger staged run.
"""


def decision_gate() -> str:
    return f"""# M3-v2 Decision Gate

## Decision Categories

{bullet_list(f'`{category}`' for category in DECISION_CATEGORIES)}

## Adopt v2 Only If

- v2 improves or preserves biological interpretability relative to v1.
- v2 reduces obvious slice/mouse artifact.
- v2 does not collapse all transitions into a few targets.
- v2 is stable across seed and subsample.
- v2 is computationally feasible.
- v2 is easier to extend to barcode evidence.

## Non-Adoption Conditions

Keep v1 as the main baseline if v2 is less interpretable, less stable, computationally excessive, or mostly changes outputs without explaining why. Keep v1 and v2 as complementary if v2 captures plausible alternative biology but does not dominate v1 across required metrics.

## Explicit Rule

M3-v2 should not replace M3-v1 merely because it is more sophisticated.
"""


def barcode_contract() -> str:
    return """# M3-v2 Barcode Compatibility Contract

## Boundary

Raw DARLIN reads should be processed by official or lab-standard DARLIN preprocessing first. NicheFate should consume processed clone/barcode tables, not raw sequencing reads.

## Expected Future Barcode Evidence

- Cell clone assignment.
- Niche clone composition.
- Clone-supported source-target coupling.
- Explicit `G_barcode` or `P_barcode` evidence for the M3-v2 kernel.

## Data Modes

- `pseudo-only`: no barcode evidence; `G_barcode = 1`.
- `barcode-only`: transition coupling is driven by processed clone evidence for scoped analyses.
- `hybrid`: state distance and soft gates are combined with clone-supported coupling.

## Compatibility Requirement

M3-v2 should be designed so pseudo-only mode can be upgraded to hybrid mode without rewriting M1/M2. Barcode evidence should be modular directionality/coupling evidence, not a post-hoc label and not just another weighted distance column.
"""


def write_summary(
    generated: list[Path],
    source_slices: list[str],
    target_slices: list[str],
    source_rows: int,
    counts: dict[str, int],
    context_files: list[dict[str, Any]],
) -> Path:
    path = REPORTS / "m3_v2_design_summary.json"
    generated_with_summary = [*generated, path]
    summary = {
        "schema_version": "m3_m4_v2_design_summary_v1",
        "generated_date": date.today().isoformat(),
        "task_type": "report_only_design_protocol",
        "output_root": str(OUT),
        "reports_root": str(REPORTS),
        "generated_files": [str(path) for path in generated_with_summary],
        "context_files": context_files,
        "v1_baseline": {
            "m1_m2_stable": True,
            "m3_v1_role": "pseudo-only time-coupled transition edge construction",
            "m4c_v1_role": "frozen pseudo-only endpoint-attraction / fate-propagation baseline",
            "v1_is_comparator": True,
        },
        "m3_v2_kernel": {
            "formula": "exp(-d_state(i,j)/tau_i) * G_time * G_composition * G_spatial_topology * G_slice_mouse * G_barcode",
            "barcode_gate_in_pseudo_only_pilot": "neutral",
            "feature_group_counts": counts,
        },
        "recommended_pilot": {
            "time_pair": "D9->D21",
            "future_output_root": str(PILOT_ROOT),
            "source_slices": source_slices,
            "target_slices": target_slices,
            "source_rows_before_optional_subsample": source_rows,
            "fallback_fixed_source_subset": 50000,
        },
        "benchmark_metrics_count": len(METRICS),
        "decision_categories": DECISION_CATEGORIES,
        "next_engineering_step": "implement M3-v2 small pilot runner and focused tests only",
        "constraints": {
            "no_full_m3_v2_production": True,
            "no_m4a_v2_assembly": True,
            "no_m4c_v2_propagation": True,
            "no_pygpcca": True,
            "no_k_gpcca_production": True,
            "no_upstream_v1_modification": True,
        },
        "citation_policy": "Design rationale only; literature citation insertion is a future manuscript task.",
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    m2_rows = read_csv_rows(M2_AUDIT_CSV)
    time_pairs = read_json(M3_TIME_PAIRS, [])
    feature_groups = read_json(M3_FEATURE_GROUPS, {"feature_groups": {}})
    counts = feature_group_counts(feature_groups)
    source_slices, target_slices, source_rows = choose_pilot_slices(m2_rows)
    context_files = context_inventory()

    generated = [
        write_text(
            "m3_m4_v2_design_justification.md",
            design_justification(time_pairs, counts, source_slices, target_slices, source_rows),
        ),
        write_text("m3_v1_limitations_and_what_v2_tests.md", limitations_report()),
        write_text("m3_v2_kernel_definition.md", kernel_definition(counts)),
        write_text("m3_v2_small_pilot_protocol.md", pilot_protocol(source_slices, target_slices, source_rows)),
        write_metrics_csv(),
        write_text("m3_v2_decision_gate.md", decision_gate()),
        write_text("m3_v2_barcode_compatibility_contract.md", barcode_contract()),
    ]
    generated.append(write_summary(generated, source_slices, target_slices, source_rows, counts, context_files))

    print(f"reports: {REPORTS}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
