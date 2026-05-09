import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "m4d_01b_setup_standard_gpcca_env.sh"
VALIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "m4d_01b_validate_standard_gpcca_backend.py"
SPEC = importlib.util.spec_from_file_location("m4d_01b_validate_standard_gpcca_backend", VALIDATE_SCRIPT)
m4d01b = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = m4d01b
SPEC.loader.exec_module(m4d01b)


def test_setup_script_targets_only_isolated_gpcca_environment() -> None:
    text = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "ENV_NAME=\"nichefate-gpcca\"" in text
    assert "env_logs" in text
    assert "conda create" in text
    assert "python=3.12" in text
    assert not re.search(r"conda\s+(create|install|run)[^\n]*omicverse", text)
    assert "scipy_pcca_like_diagnostic_fallback" in text


def test_output_paths_are_report_only_and_exclude_forbidden_downstream_targets(tmp_path: Path) -> None:
    outputs = m4d01b.output_paths(tmp_path / "reports")
    forbidden = [
        "full_node",
        "P_forward_no_terminal_selfloops",
        "absorption",
        "fate_probability",
        "regulator",
        "m5",
        "branched_nicheflow",
        "branchsbm",
    ]

    assert set(outputs) == {
        "environment_md",
        "environment_json",
        "pygpcca_md",
        "cellrank_md",
        "recommendation_md",
    }
    for path in outputs.values():
        assert path.parent == tmp_path / "reports"
        assert not any(token in path.name.lower() for token in forbidden)


def test_reports_distinguish_standard_gpcca_from_diagnostic_scipy() -> None:
    payload = {
        "recommendation": {
            "immediate_recommendation": "A. pyGPCCA on supernode Markov chain",
            "ordered_route": [
                "A. pyGPCCA on supernode Markov chain",
                "B. CellRank GPCCA with custom/precomputed kernel",
                "C. scipy_pcca_like_diagnostic_fallback only if standard backend cannot be established",
            ],
            "next_stage_after_success": "M4D-01c: build supernode Markov chain",
        }
    }

    text = m4d01b.recommendation_markdown(payload)

    assert "pyGPCCA on supernode Markov chain" in text
    assert "CellRank GPCCA with custom/precomputed kernel" in text
    assert "scipy fallback remains diagnostic only" in text


def test_pygpcca_toy_validation_output_schema() -> None:
    class FakeGPCCA:
        def __init__(self, matrix, z="LM", method="brandts"):
            self.matrix = matrix

        def optimize(self, n_macrostates):
            n_states = int(self.matrix.shape[0])
            self.memberships = np.ones((n_states, int(n_macrostates))) / float(n_macrostates)
            self.macrostate_assignment = np.zeros(n_states, dtype=int)
            self.coarse_grained_transition_matrix = np.eye(int(n_macrostates))
            self.n_m = int(n_macrostates)
            return self

    result = m4d01b.run_gpcca_validation(
        SimpleNamespace(GPCCA=FakeGPCCA),
        np.eye(4),
        2,
        "toy",
        "dense_ndarray",
    )

    assert result["success"]
    assert result["memberships_shape"] == [4, 2]
    assert result["macrostate_assignment_shape"] == [4]
    assert result["coarse_grained_transition_matrix_shape"] == [2, 2]
    assert result["n_macrostates_observed"] == 2


def test_cellrank_feasibility_report_schema_when_import_unavailable() -> None:
    packages = {"cellrank": {"import_ok": False, "error": "ModuleNotFoundError: cellrank"}}
    result = m4d01b.inspect_cellrank({}, packages)

    assert set(result) == {
        "import_ok",
        "status",
        "error",
        "gpcca_estimator_found",
        "gpcca_estimator_path",
        "precomputed_kernel_found",
        "precomputed_kernel_path",
        "custom_kernel_base_found",
        "custom_kernel_base_path",
        "direct_matrix_constructor_success",
        "direct_matrix_constructor_error",
        "precomputed_or_custom_kernel_feasible",
    }
    assert result["status"] == "not_feasible"
    assert not result["precomputed_or_custom_kernel_feasible"]


def test_scope_guards_exclude_forbidden_m4d01b_work() -> None:
    guards = m4d01b.SCOPE_GUARDS

    assert guards["environment_interface_validation_only"] is True
    assert guards["no_full_node_gpcca"] is True
    assert guards["no_full_m4a_transition_matrix_loaded_for_gpcca"] is True
    assert guards["no_absorption_probability"] is True
    assert guards["no_fate_probability"] is True
    assert guards["no_regulator_analysis"] is True
    assert guards["no_m5"] is True
    assert guards["no_branched_nicheflow_training"] is True
    assert guards["no_branchsbm_training"] is True
    assert guards["scipy_fallback_is_diagnostic_only"] is True
