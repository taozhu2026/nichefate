from pathlib import Path

import pandas as pd

from nichefate import m4d_supernode as m4d


def inspect_without_standard_gpcca(monkeypatch) -> pd.DataFrame:
    real_find_spec = m4d.importlib.util.find_spec

    def fake_find_spec(module_name: str):
        if module_name in {"pygpcca", "cellrank"}:
            return None
        return real_find_spec(module_name)

    monkeypatch.setattr(m4d.importlib.util, "find_spec", fake_find_spec)
    return m4d.inspect_backend_availability()


def test_backend_report_does_not_crash_without_pygpcca_or_cellrank(monkeypatch) -> None:
    backend = inspect_without_standard_gpcca(monkeypatch)

    selected_backend, result_label, true_gpcca_available = m4d.selected_backend_label(backend)

    assert selected_backend == m4d.NO_TRUE_GPCCA_BACKEND
    assert result_label == m4d.NO_TRUE_GPCCA_BACKEND_LABEL
    assert true_gpcca_available is False
    assert not backend["selected"].astype(bool).any()
    assert not backend.loc[backend["backend"].isin(m4d.TRUE_GPCCA_BACKENDS), "available"].astype(bool).any()


def test_scipy_fallback_is_diagnostic_only(monkeypatch) -> None:
    backend = inspect_without_standard_gpcca(monkeypatch)
    fallback = backend.loc[backend["backend"] == m4d.DIAGNOSTIC_FALLBACK_BACKEND].iloc[0]

    assert bool(fallback["available"]) in {True, False}
    assert bool(fallback["true_gpcca_backend"]) is False
    assert bool(fallback["selected"]) is False
    assert "diagnostic" in fallback["result_label"]
    assert "not true GPCCA" in fallback["result_label"]


def test_backend_availability_markdown_does_not_claim_true_gpcca(monkeypatch) -> None:
    backend = inspect_without_standard_gpcca(monkeypatch)
    text = m4d.backend_availability_markdown(backend)

    assert "True GPCCA backend available: `False`" in text
    assert "scipy_pcca_like_diagnostic_fallback" in text
    assert "will use a **PCCA-like coarse spectral fallback**" not in text
    assert "true GPCCA macrostates" not in text


def test_standard_gpcca_report_paths_are_reports_only(tmp_path: Path) -> None:
    outputs = m4d.m4d_output_paths(
        {
            "output_root": tmp_path / "m4d",
            "reports_dir": tmp_path / "m4d" / "reports",
        }
    )
    report_keys = [
        "backend_md",
        "backend_csv",
        "standard_gpcca_backend_plan",
        "cellrank_feasibility",
        "standard_gpcca_next_step",
    ]
    forbidden = ["absorption", "fate_probability", "branched", "branchsbm", "m5", "regulator"]

    for key in report_keys:
        path = outputs[key]
        assert path.parent == tmp_path / "m4d" / "reports"
        assert not any(token in path.name.lower() for token in forbidden)


def test_standard_gpcca_review_reports_include_recommendation_without_downstream_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = inspect_without_standard_gpcca(monkeypatch)
    outputs = m4d.m4d_output_paths(
        {
            "output_root": tmp_path / "m4d",
            "reports_dir": tmp_path / "m4d" / "reports",
        }
    )

    m4d.write_standard_gpcca_review_reports(outputs, backend)

    for key in ["standard_gpcca_backend_plan", "cellrank_feasibility", "standard_gpcca_next_step"]:
        assert outputs[key].is_file()
    recommendation = outputs["standard_gpcca_next_step"].read_text(encoding="utf-8")
    assert "A. create isolated `nichefate-gpcca` environment" in recommendation
    assert "B. run pyGPCCA on the supernode Markov matrix" in recommendation
    assert "C. use `scipy_pcca_like_diagnostic_fallback` only" in recommendation
    for forbidden_path in ["node_projected", "fate_probability_node_summary", "regulator_output"]:
        assert forbidden_path not in recommendation
