import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from nichefate.transition import (
    CandidateNeighborBackendStatus,
    build_candidate_neighbors,
    inspect_candidate_neighbor_backend,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


m3_08 = load_script("m3_08_inspect_ann_backends.py")
m3_09 = load_script("m3_09_design_ann_validation.py")
m3_10 = load_script("m3_10_design_slurm_full_m3_strategy.py")


def toy_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "reports_dir": str(tmp_path / "reports"),
            "logs_dir": str(tmp_path / "logs"),
            "use_ssd": False,
        },
        "candidate_edges": {
            "max_source_niches_per_pair": 5,
            "retrieval_metric": "euclidean",
        },
        "full_m3": {
            "enabled": False,
            "execution_mode": "design_only",
            "write_global_kernel": False,
            "output_root": str(tmp_path / "by_pair"),
            "max_memory_gb_warning": 1,
            "retrieval_feature_groups": ["retrieval"],
            "rerank_feature_groups": ["rerank"],
        },
    }


def toy_shards() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_time": "t0",
                "target_time": "t1",
                "source_slice_id": "s_big",
                "source_slice_file": "s_big.m0.h5ad",
                "source_rows": 10,
                "target_time_rows": 20,
                "candidate_k": 30,
                "expected_edge_rows": 300,
            },
            {
                "source_time": "t0",
                "target_time": "t1",
                "source_slice_id": "s_small",
                "source_slice_file": "s_small.m0.h5ad",
                "source_rows": 4,
                "target_time_rows": 20,
                "candidate_k": 30,
                "expected_edge_rows": 120,
            },
        ]
    )


def test_required_neighbor_backends_remain_available() -> None:
    source = np.array([[0.0, 0.0], [2.0, 0.0]])
    target = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    exact = build_candidate_neighbors(source, target, k=2, backend="sklearn_exact")
    chunked = build_candidate_neighbors(source, target, k=2, backend="numpy_chunked", chunk_size=1)

    assert exact.backend == "sklearn_exact"
    assert chunked.backend == "numpy_chunked"
    np.testing.assert_array_equal(exact.indices, chunked.indices)
    np.testing.assert_allclose(exact.distances, chunked.distances)


def test_unavailable_optional_ann_backend_returns_clean_status(monkeypatch) -> None:
    original = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "faiss":
            return None
        return original(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    status = inspect_candidate_neighbor_backend("faiss", run_toy_check=True)
    result = build_candidate_neighbors(
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 0.0]]),
        k=1,
        backend="faiss",
    )

    assert isinstance(status, CandidateNeighborBackendStatus)
    assert status.available is False
    assert "not importable" in status.reason
    assert isinstance(result, CandidateNeighborBackendStatus)
    assert result.available is False


def test_ann_inspection_is_toy_level_only() -> None:
    availability = m3_08.inspect_backends("euclidean")

    assert set(availability["backend"]) >= {"sklearn_exact", "numpy_chunked", "faiss"}
    assert set(availability["check_scope"]) == {"toy_in_memory_only"}


def test_ann_validation_plan_writes_expected_columns(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    pilot_metrics = pd.DataFrame(
        [{"source_time": "t0", "target_time": "t1", "completed_shards": 1, "failed_shards": 0}]
    )
    runtime = pd.DataFrame(
        [{"source_time": "t0", "target_time": "t1", "conservative_projection_hours": 0.1}]
    )

    validation = m3_09.design_validation_shards(toy_shards(), pilot_metrics, runtime, config)
    report_path = tmp_path / "validation.md"
    m3_09.write_validation_plan(report_path, validation, pilot_metrics)
    text = report_path.read_text(encoding="utf-8")

    assert list(validation.columns) == m3_09.VALIDATION_COLUMNS
    assert validation.loc[validation["recommended"], "source_slice_id"].iloc[0] == "s_small"
    assert "recall@30 >= 0.8" in text
    assert "DESIGN" not in "".join(validation["source_slice_id"])


def test_slurm_strategy_writes_shard_table_and_non_submitting_template(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    shards = toy_shards()
    shards["target_rows"] = shards["target_time_rows"]
    memory = m3_10.estimate_target_time_memory(shards, config, 3, 2)
    strategy = m3_10.build_strategy_table(toy_shards(), config, memory)
    report_path = tmp_path / "strategy.md"
    template_path = tmp_path / "m3_full_shard_array.sbatch"
    strategy_csv = tmp_path / "m3_slurm_array_shards.csv"

    strategy.to_csv(strategy_csv, index=False)
    m3_10.write_strategy_report(report_path, strategy, memory, "toy dimensions")
    m3_10.write_slurm_template(template_path, config, strategy_csv)
    template_lines = template_path.read_text(encoding="utf-8").splitlines()
    non_directive = "\n".join(line for line in template_lines if not line.startswith("#SBATCH")).lower()

    assert set(m3_10.STRATEGY_COLUMNS) <= set(strategy.columns)
    assert strategy["array_task_index"].tolist() == [1, 2]
    assert "sbatch " not in non_directive
    assert "srun " not in non_directive
    assert "scancel" not in non_directive


def test_generated_reports_have_no_executable_downstream_commands(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    shards = toy_shards()
    shards["target_rows"] = shards["target_time_rows"]
    memory = m3_10.estimate_target_time_memory(shards, config, 3, 2)
    strategy = m3_10.build_strategy_table(toy_shards(), config, memory)
    report_path = tmp_path / "strategy.md"
    template_path = tmp_path / "m3_full_shard_array.sbatch"
    m3_10.write_strategy_report(report_path, strategy, memory, "toy dimensions")
    m3_10.write_slurm_template(template_path, config, tmp_path / "shards.csv")

    executable_lines = []
    for path in [report_path, template_path]:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lower()
            if stripped.startswith(("python ", "conda ", "rscript ", "snakemake ")):
                executable_lines.append(stripped)
    text = "\n".join(executable_lines)

    for token in ["gpcca", "fate", "branched", "nicheflow", "m5", "regulator", "markov"]:
        assert token not in text
    for path in strategy["output_dir"].astype(str):
        assert all(token not in path.lower() for token in ["gpcca", "fate", "branched", "m5", "regulator"])


def test_forbidden_import_scan_allows_optional_lazy_ann_names() -> None:
    paths = [
        PROJECT_ROOT / "src" / "nichefate" / "transition.py",
        PROJECT_ROOT / "scripts" / "m3_08_inspect_ann_backends.py",
        PROJECT_ROOT / "scripts" / "m3_09_design_ann_validation.py",
        PROJECT_ROOT / "scripts" / "m3_10_design_slurm_full_m3_strategy.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for package in ["squidpy", "spatialdata", "harmonypy", "scvi"]:
        assert f"import {package}" not in text
        assert f"from {package}" not in text


def test_no_dataset_specific_hard_coding_in_m3_core() -> None:
    paths = [
        PROJECT_ROOT / "src" / "nichefate" / "transition.py",
        PROJECT_ROOT / "scripts" / "m3_08_inspect_ann_backends.py",
        PROJECT_ROOT / "scripts" / "m3_09_design_ann_validation.py",
        PROJECT_ROOT / "scripts" / "m3_10_design_slurm_full_m3_strategy.py",
        PROJECT_ROOT / "configs" / "m3_transition_kernel.yaml",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for token in ["Moffitt", "Cadinu", "DSS", "colon", "Day35", "Sample_type"]:
        assert token not in text
