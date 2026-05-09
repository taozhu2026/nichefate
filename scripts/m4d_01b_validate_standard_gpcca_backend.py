#!/usr/bin/env python
"""Validate the isolated M4D-01b standard GPCCA backend environment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "configs/m4d_markov_macrostate_visualization.yaml"
DEFAULT_REPORTS_DIR = Path("/home/zhutao/scratch/nichefate/m4d/reports")
ENV_NAME = "nichefate-gpcca"
DIAGNOSTIC_FALLBACK_BACKEND = "scipy_pcca_like_diagnostic_fallback"
PACKAGE_MODULES = {
    "pygpcca": "pygpcca",
    "cellrank": "cellrank",
    "scipy": "scipy",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "sklearn",
    "anndata": "anndata",
    "scanpy": "scanpy",
    "matplotlib": "matplotlib",
}
SCOPE_GUARDS = {
    "environment_interface_validation_only": True,
    "no_full_node_gpcca": True,
    "no_full_m4a_transition_matrix_loaded_for_gpcca": True,
    "no_absorption_probability": True,
    "no_fate_probability": True,
    "no_regulator_analysis": True,
    "no_m5": True,
    "no_branched_nicheflow_training": True,
    "no_branchsbm_training": True,
    "omicverse_modified": False,
    "scipy_fallback_is_diagnostic_only": True,
}
ALGORITHM_PRIORITY = [
    "primary: pyGPCCA on transition matrices",
    "secondary: CellRank GPCCA / PrecomputedKernel / custom kernel feasibility",
    f"diagnostic only: {DIAGNOSTIC_FALLBACK_BACKEND}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return [json_safe(item) for item in value.tolist()]
    except Exception:  # noqa: BLE001
        pass
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def assert_no_ssd_path(path: Path, label: str) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved == "/ssd" or resolved.startswith("/ssd/"):
        raise ValueError(f"Refusing to use disallowed storage for {label}: {path}")


def configured_reports_dir(config_path: Path) -> Path:
    if not config_path.is_file():
        return DEFAULT_REPORTS_DIR
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*reports_dir:\s*(\S+)\s*$", text)
    if not match:
        return DEFAULT_REPORTS_DIR
    reports_dir = Path(match.group(1)).expanduser()
    assert_no_ssd_path(reports_dir, "paths.reports_dir")
    return reports_dir


def output_paths(reports_dir: Path) -> dict[str, Path]:
    return {
        "environment_md": reports_dir / "m4d_standard_gpcca_environment_report.md",
        "environment_json": reports_dir / "m4d_standard_gpcca_environment_report.json",
        "pygpcca_md": reports_dir / "m4d_pygpcca_toy_validation.md",
        "cellrank_md": reports_dir / "m4d_cellrank_integration_feasibility.md",
        "recommendation_md": reports_dir / "m4d_standard_gpcca_next_step_recommendation.md",
    }


def import_status(module_name: str) -> tuple[Any | None, dict[str, Any]]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return None, {
            "module": module_name,
            "import_ok": False,
            "version": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return module, {
        "module": module_name,
        "import_ok": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "error": "",
    }


def collect_package_versions() -> tuple[dict[str, Any], dict[str, Any]]:
    modules: dict[str, Any] = {}
    packages: dict[str, Any] = {}
    for package_name, module_name in PACKAGE_MODULES.items():
        module, status = import_status(module_name)
        modules[package_name] = module
        packages[package_name] = status
    return modules, packages


def toy_stochastic_matrix(np_module: Any) -> Any:
    return np_module.array(
        [
            [0.88, 0.10, 0.02, 0.00, 0.00, 0.00],
            [0.12, 0.84, 0.04, 0.00, 0.00, 0.00],
            [0.03, 0.07, 0.85, 0.05, 0.00, 0.00],
            [0.00, 0.00, 0.05, 0.85, 0.07, 0.03],
            [0.00, 0.00, 0.00, 0.04, 0.84, 0.12],
            [0.00, 0.00, 0.00, 0.02, 0.10, 0.88],
        ],
        dtype=float,
    )


def toy_absorbing_matrix(np_module: Any) -> Any:
    return np_module.array(
        [
            [0.70, 0.20, 0.10, 0.00, 0.00],
            [0.10, 0.70, 0.10, 0.10, 0.00],
            [0.00, 0.10, 0.70, 0.10, 0.10],
            [0.00, 0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00],
        ],
        dtype=float,
    )


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def run_gpcca_validation(
    pygpcca_module: Any,
    matrix: Any,
    n_macrostates: int,
    label: str,
    input_kind: str,
    method: str = "brandts",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": label,
        "input_kind": input_kind,
        "method": method,
        "n_states": int(matrix.shape[0]),
        "n_macrostates_requested": int(n_macrostates),
        "success": False,
        "error": "",
        "memberships_shape": None,
        "macrostate_assignment_shape": None,
        "coarse_grained_transition_matrix_shape": None,
        "n_macrostates_observed": None,
    }
    try:
        gpcca = pygpcca_module.GPCCA(matrix, z="LM", method=method)
        gpcca.optimize(int(n_macrostates))
        memberships = getattr(gpcca, "memberships", None)
        assignments = getattr(gpcca, "macrostate_assignment", None)
        coarse = getattr(gpcca, "coarse_grained_transition_matrix", None)
        observed = getattr(gpcca, "n_m", None)
        result.update(
            {
                "success": True,
                "memberships_shape": shape_of(memberships),
                "macrostate_assignment_shape": shape_of(assignments),
                "coarse_grained_transition_matrix_shape": shape_of(coarse),
                "n_macrostates_observed": int(observed) if observed is not None else None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def safe_gpcca_subprocess_env() -> dict[str, str]:
    tmpdir = Path("/tmp/nichefate_m4d01b_gpcca_tmp")
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in [
        "TMPDIR",
        "TEMP",
        "TMP",
        "OMPI_MCA_orte_tmpdir_base",
        "OMPI_MCA_prte_tmpdir_base",
        "PRTE_MCA_prte_tmpdir_base",
        "PMIX_MCA_pmix_tmpdir_base",
    ]:
        env[key] = str(tmpdir)
    return env


def run_sparse_csr_subprocess() -> dict[str, Any]:
    code = r'''
import json

import numpy as np
import pygpcca
import scipy.sparse as sp

matrix = np.array(
    [
        [0.88, 0.10, 0.02, 0.00, 0.00, 0.00],
        [0.12, 0.84, 0.04, 0.00, 0.00, 0.00],
        [0.03, 0.07, 0.85, 0.05, 0.00, 0.00],
        [0.00, 0.00, 0.05, 0.85, 0.07, 0.03],
        [0.00, 0.00, 0.00, 0.04, 0.84, 0.12],
        [0.00, 0.00, 0.00, 0.02, 0.10, 0.88],
    ],
    dtype=float,
)
result = {
    "label": "toy_stochastic_sparse_csr",
    "input_kind": "scipy_csr",
    "method": "krylov",
    "n_states": 6,
    "n_macrostates_requested": 2,
    "success": False,
    "error": "",
    "memberships_shape": None,
    "macrostate_assignment_shape": None,
    "coarse_grained_transition_matrix_shape": None,
    "n_macrostates_observed": None,
}
try:
    gpcca = pygpcca.GPCCA(sp.csr_matrix(matrix), z="LM", method="krylov")
    gpcca.optimize(2)
    result.update(
        {
            "success": True,
            "memberships_shape": list(getattr(gpcca, "memberships").shape),
            "macrostate_assignment_shape": list(getattr(gpcca, "macrostate_assignment").shape),
            "coarse_grained_transition_matrix_shape": list(getattr(gpcca, "coarse_grained_transition_matrix").shape),
            "n_macrostates_observed": int(getattr(gpcca, "n_m")),
        }
    )
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
'''
    base_result = {
        "label": "toy_stochastic_sparse_csr",
        "input_kind": "scipy_csr",
        "method": "krylov",
        "n_states": 6,
        "n_macrostates_requested": 2,
        "success": False,
        "error": "",
        "memberships_shape": None,
        "macrostate_assignment_shape": None,
        "coarse_grained_transition_matrix_shape": None,
        "n_macrostates_observed": None,
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            env=safe_gpcca_subprocess_env(),
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        base_result["error"] = f"TimeoutExpired: sparse CSR pyGPCCA validation exceeded {exc.timeout} seconds"
        return base_result
    if completed.returncode != 0:
        stderr = completed.stderr.strip().replace("\n", " | ")
        stdout = completed.stdout.strip().replace("\n", " | ")
        base_result["error"] = f"subprocess_returncode_{completed.returncode}: stderr={stderr}; stdout={stdout}"
        return base_result
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        base_result["error"] = f"json_parse_failed: {type(exc).__name__}: {exc}; stdout={completed.stdout.strip()}"
        return base_result


def validate_pygpcca(modules: dict[str, Any], packages: dict[str, Any]) -> dict[str, Any]:
    pygpcca_module = modules.get("pygpcca")
    np_module = modules.get("numpy")
    scipy_module = modules.get("scipy")
    if not packages["pygpcca"]["import_ok"]:
        return {
            "import_ok": False,
            "critical_success": False,
            "error": packages["pygpcca"]["error"],
            "toy_stochastic": None,
            "toy_absorbing": None,
            "sparse_csr": None,
            "dense_conversion_for_toy": None,
            "sparse_csr_supported_directly": False,
            "dense_conversion_required_for_toy": False,
        }
    if np_module is None or scipy_module is None:
        return {
            "import_ok": True,
            "critical_success": False,
            "error": "numpy or scipy was unavailable after package import collection",
            "toy_stochastic": None,
            "toy_absorbing": None,
            "sparse_csr": None,
            "dense_conversion_for_toy": None,
            "sparse_csr_supported_directly": False,
            "dense_conversion_required_for_toy": False,
        }

    stochastic = toy_stochastic_matrix(np_module)
    absorbing = toy_absorbing_matrix(np_module)
    toy_result = run_gpcca_validation(pygpcca_module, stochastic, 2, "toy_stochastic", "dense_ndarray")
    absorbing_result = run_gpcca_validation(pygpcca_module, absorbing, 2, "toy_absorbing", "dense_ndarray")
    sparse_result = run_sparse_csr_subprocess()
    dense_conversion = None
    if not sparse_result["success"]:
        dense_conversion = run_gpcca_validation(
            pygpcca_module,
            stochastic,
            2,
            "toy_stochastic_dense_after_sparse_failure",
            "dense_ndarray",
        )
    critical_success = bool(toy_result["success"] and absorbing_result["success"])
    return {
        "import_ok": True,
        "critical_success": critical_success,
        "error": "",
        "toy_stochastic": toy_result,
        "toy_absorbing": absorbing_result,
        "sparse_csr": sparse_result,
        "dense_conversion_for_toy": dense_conversion,
        "sparse_csr_supported_directly": bool(sparse_result["success"]),
        "dense_conversion_required_for_toy": bool((not sparse_result["success"]) and dense_conversion and dense_conversion["success"]),
    }


def inspect_submodule(module_name: str, symbol: str) -> tuple[bool, str, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return False, "", f"{module_name}: {type(exc).__name__}: {exc}"
    if getattr(module, symbol, None) is None:
        return False, "", ""
    return True, f"{module_name}.{symbol}", ""


def inspect_cellrank(modules: dict[str, Any], packages: dict[str, Any]) -> dict[str, Any]:
    if not packages["cellrank"]["import_ok"]:
        return {
            "import_ok": False,
            "status": "not_feasible",
            "error": packages["cellrank"]["error"],
            "gpcca_estimator_found": False,
            "gpcca_estimator_path": "",
            "precomputed_kernel_found": False,
            "precomputed_kernel_path": "",
            "custom_kernel_base_found": False,
            "custom_kernel_base_path": "",
            "direct_matrix_constructor_success": False,
            "direct_matrix_constructor_error": "",
            "precomputed_or_custom_kernel_feasible": False,
        }

    estimator_paths: list[str] = []
    precomputed_paths: list[str] = []
    custom_paths: list[str] = []
    import_notes: list[str] = []
    for module_name in ["cellrank.estimators", "cellrank.tl.estimators"]:
        found, path, note = inspect_submodule(module_name, "GPCCA")
        if path:
            estimator_paths.append(path)
        if note:
            import_notes.append(note)
    for module_name in ["cellrank.kernels", "cellrank.tl.kernels"]:
        found, path, note = inspect_submodule(module_name, "PrecomputedKernel")
        if path:
            precomputed_paths.append(path)
        if note:
            import_notes.append(note)
        found, path, note = inspect_submodule(module_name, "Kernel")
        if path:
            custom_paths.append(path)
        if note:
            import_notes.append(note)

    direct_success = False
    direct_error = ""
    if estimator_paths:
        try:
            gpcca_class = getattr(importlib.import_module(estimator_paths[0].rsplit(".", 1)[0]), "GPCCA")
            np_module = modules["numpy"]
            sparse = importlib.import_module("scipy.sparse")
            _ = gpcca_class(sparse.csr_matrix(toy_stochastic_matrix(np_module)))
            direct_success = True
        except Exception as exc:  # noqa: BLE001
            direct_error = f"{type(exc).__name__}: {exc}"

    feasible = bool(estimator_paths) and bool(precomputed_paths or custom_paths or direct_success)
    status = "feasible" if feasible else "partially_feasible" if estimator_paths else "not_feasible"
    return {
        "import_ok": True,
        "status": status,
        "error": "; ".join(import_notes),
        "gpcca_estimator_found": bool(estimator_paths),
        "gpcca_estimator_path": ";".join(estimator_paths),
        "precomputed_kernel_found": bool(precomputed_paths),
        "precomputed_kernel_path": ";".join(precomputed_paths),
        "custom_kernel_base_found": bool(custom_paths),
        "custom_kernel_base_path": ";".join(custom_paths),
        "direct_matrix_constructor_success": direct_success,
        "direct_matrix_constructor_error": direct_error,
        "precomputed_or_custom_kernel_feasible": feasible,
    }


def recommendation(pygpcca: dict[str, Any], cellrank: dict[str, Any]) -> dict[str, Any]:
    if pygpcca["critical_success"]:
        immediate = "A. pyGPCCA on supernode Markov chain"
    elif cellrank.get("precomputed_or_custom_kernel_feasible"):
        immediate = "B. CellRank GPCCA with custom/precomputed kernel"
    else:
        immediate = "standard backend setup is incomplete; do not run M4D production GPCCA"
    return {
        "immediate_recommendation": immediate,
        "ordered_route": [
            "A. pyGPCCA on supernode Markov chain",
            "B. CellRank GPCCA with custom/precomputed kernel",
            f"C. {DIAGNOSTIC_FALLBACK_BACKEND} only if standard backend cannot be established",
        ],
        "next_stage_after_success": "M4D-01c: build supernode Markov chain and run standard pyGPCCA on the coarse transition matrix",
    }


def environment_payload(config_path: Path) -> dict[str, Any]:
    return {
        "stage": "M4D-01b Standard GPCCA Environment Setup and Interface Validation",
        "environment_name": ENV_NAME,
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "config_path": str(config_path),
        "generated_at_utc": utc_now_iso(),
    }


def build_payload(config_path: Path) -> dict[str, Any]:
    modules, packages = collect_package_versions()
    pygpcca = validate_pygpcca(modules, packages)
    cellrank = inspect_cellrank(modules, packages)
    return {
        "schema_version": "m4d_standard_gpcca_environment_report_v1",
        "environment": environment_payload(config_path),
        "packages": packages,
        "algorithm_priority": ALGORITHM_PRIORITY,
        "pygpcca_validation": pygpcca,
        "cellrank_feasibility": cellrank,
        "scope_guards": SCOPE_GUARDS,
        "recommendation": recommendation(pygpcca, cellrank),
    }


def package_summary_lines(packages: dict[str, Any]) -> list[str]:
    lines = []
    for name, status in packages.items():
        lines.append(
            f"- {name}: import_ok=`{bool(status['import_ok'])}`, "
            f"version=`{status['version']}`, error=`{status['error']}`"
        )
    return lines


def environment_markdown(payload: dict[str, Any]) -> str:
    env = payload["environment"]
    pygpcca = payload["pygpcca_validation"]
    cellrank = payload["cellrank_feasibility"]
    lines = [
        "# M4D-01b Standard GPCCA Environment Report",
        "",
        f"- generated at UTC: {env['generated_at_utc']}",
        f"- environment name: `{env['environment_name']}`",
        f"- python executable: `{env['python_executable']}`",
        f"- pyGPCCA imported: `{payload['packages']['pygpcca']['import_ok']}`",
        f"- CellRank imported: `{payload['packages']['cellrank']['import_ok']}`",
        f"- pyGPCCA toy validation critical success: `{pygpcca['critical_success']}`",
        f"- CellRank route status: `{cellrank['status']}`",
        "",
        "## Algorithm Priority",
    ]
    lines.extend(f"- {item}" for item in payload["algorithm_priority"])
    lines.extend(["", "## Package Versions"])
    lines.extend(package_summary_lines(payload["packages"]))
    lines.extend(["", "## Scope Guards"])
    lines.extend(f"- `{key}`: `{value}`" for key, value in payload["scope_guards"].items())
    return "\n".join(lines) + "\n"


def pygpcca_markdown(payload: dict[str, Any]) -> str:
    validation = payload["pygpcca_validation"]
    lines = [
        "# M4D pyGPCCA Toy Validation",
        "",
        f"- import ok: `{validation['import_ok']}`",
        f"- critical success: `{validation['critical_success']}`",
        f"- sparse CSR supported directly: `{validation['sparse_csr_supported_directly']}`",
        f"- dense conversion required for toy only: `{validation['dense_conversion_required_for_toy']}`",
        "",
        "## Results",
    ]
    for key in ["toy_stochastic", "toy_absorbing", "sparse_csr", "dense_conversion_for_toy"]:
        value = validation.get(key)
        if value is None:
            lines.append(f"- {key}: not run")
        else:
            lines.append(
                f"- {key}: success=`{value['success']}`, input_kind=`{value['input_kind']}`, "
                f"method=`{value['method']}`, "
                f"memberships_shape=`{value['memberships_shape']}`, error=`{value['error']}`"
            )
    return "\n".join(lines) + "\n"


def cellrank_markdown(payload: dict[str, Any]) -> str:
    cellrank = payload["cellrank_feasibility"]
    lines = [
        "# M4D CellRank Integration Feasibility",
        "",
        "This stage inspects CellRank GPCCA and precomputed/custom kernel feasibility only. It does not compute fate probabilities, absorption probabilities, terminal states, drivers, or regulator outputs.",
        "",
        f"- import ok: `{cellrank['import_ok']}`",
        f"- status: `{cellrank['status']}`",
        f"- GPCCA estimator found: `{cellrank['gpcca_estimator_found']}`",
        f"- GPCCA estimator path: `{cellrank['gpcca_estimator_path']}`",
        f"- PrecomputedKernel found: `{cellrank['precomputed_kernel_found']}`",
        f"- PrecomputedKernel path: `{cellrank['precomputed_kernel_path']}`",
        f"- custom Kernel base found: `{cellrank['custom_kernel_base_found']}`",
        f"- custom Kernel base path: `{cellrank['custom_kernel_base_path']}`",
        f"- direct matrix constructor success: `{cellrank['direct_matrix_constructor_success']}`",
        f"- direct matrix constructor error: `{cellrank['direct_matrix_constructor_error']}`",
        f"- precomputed/custom route feasible: `{cellrank['precomputed_or_custom_kernel_feasible']}`",
        f"- notes: {cellrank['error']}",
    ]
    return "\n".join(lines) + "\n"


def recommendation_markdown(payload: dict[str, Any]) -> str:
    recommendation_payload = payload["recommendation"]
    lines = [
        "# M4D Standard GPCCA Next-Step Recommendation",
        "",
        f"- immediate recommendation: {recommendation_payload['immediate_recommendation']}",
        "",
        "## Ordered Route",
    ]
    lines.extend(recommendation_payload["ordered_route"])
    lines.extend(
        [
            "",
            "## Next Stage",
            f"- {recommendation_payload['next_stage_after_success']}",
            "",
            "## Explicit Non-Goals",
            "- no full-node GPCCA",
            "- no absorption probability",
            "- no fate probability",
            "- no Branched NicheFlow / BranchSBM",
            "- no M5",
            "- no regulator analysis",
            "- scipy fallback remains diagnostic only",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(paths: dict[str, Path], payload: dict[str, Any]) -> None:
    atomic_write_json(paths["environment_json"], payload)
    atomic_write_text(paths["environment_md"], environment_markdown(payload))
    atomic_write_text(paths["pygpcca_md"], pygpcca_markdown(payload))
    atomic_write_text(paths["cellrank_md"], cellrank_markdown(payload))
    atomic_write_text(paths["recommendation_md"], recommendation_markdown(payload))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    reports_dir = configured_reports_dir(config_path)
    paths = output_paths(reports_dir)
    payload = build_payload(config_path)
    write_reports(paths, payload)

    pygpcca_ok = bool(payload["pygpcca_validation"]["critical_success"])
    cellrank_import_ok = bool(payload["packages"]["cellrank"]["import_ok"])
    if not pygpcca_ok or not cellrank_import_ok:
        print(
            "M4D-01b standard GPCCA validation failed: "
            f"pygpcca_ok={pygpcca_ok}; cellrank_import_ok={cellrank_import_ok}"
        )
        return 2
    print(
        "M4D-01b standard GPCCA validation complete: "
        f"pygpcca_ok={pygpcca_ok}; "
        f"sparse_csr_supported_directly={payload['pygpcca_validation']['sparse_csr_supported_directly']}; "
        f"cellrank_status={payload['cellrank_feasibility']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
