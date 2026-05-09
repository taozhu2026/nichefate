#!/usr/bin/env python
"""Check the lightweight M0 development environment."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nichefate.io import expected_raw_files, load_config, paths_from_config
from nichefate.utils import package_available, status_line

PACKAGE_MODULES = {
    "scanpy": "scanpy",
    "anndata": "anndata",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "h5py": "h5py",
    "zarr": "zarr",
    "pyarrow": "pyarrow",
    "pyyaml": "yaml",
    "tqdm": "tqdm",
    "networkx": "networkx",
    "pynndescent": "pynndescent",
    "omicverse": "omicverse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/m0_merfish_colitis.yaml",
        help="Path to the M0 YAML config.",
    )
    return parser.parse_args()


def check_paths(config: dict[str, object]) -> None:
    paths = paths_from_config(config)
    use_ssd = bool(config.get("paths", {}).get("use_ssd", False))

    print("\nConfigured storage paths:")
    for key, path in paths.items():
        if key == "future_ssd_output_dir" and not use_ssd:
            print(f"[DISABLED] {key} - {path}")
            continue
        exists = path.exists()
        print(status_line(key, exists, str(path)))

    print(status_line("use_ssd", not use_ssd, str(use_ssd)))

    output_root = paths["output_dir"]
    print("\nM0 output subdirectories:")
    for label in ("processed", "by_time", "by_slice", "graphs", "reports", "logs"):
        output_path = output_root / label
        print(status_line(label, output_path.is_dir(), str(output_path)))

    raw_root = paths["raw_dir"]
    required, optional = expected_raw_files(config)
    print("\nExpected raw files:")
    for filename in required:
        file_path = raw_root / filename
        print(status_line(filename, file_path.is_file(), str(file_path)))
    for filename in optional:
        file_path = raw_root / filename
        print(status_line(f"{filename} (optional)", file_path.is_file(), str(file_path)))


def check_packages(config: dict[str, object]) -> None:
    print("\nPython packages:")
    modules = dict(PACKAGE_MODULES)
    environment = config.get("environment", {})
    if isinstance(environment, dict):
        if environment.get("require_harmony", False):
            modules["harmonypy"] = "harmonypy"
        if environment.get("require_squidpy", False):
            modules["squidpy"] = "squidpy"
        if environment.get("require_spatialdata", False):
            modules["spatialdata"] = "spatialdata"
    for package_name, module_name in modules.items():
        print(status_line(package_name, package_available(module_name)))


def main() -> int:
    args = parse_args()

    print("nichefate M0 environment check")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"python: {sys.executable}")
    print(f"python_version: {platform.python_version()}")
    print(f"platform: {platform.platform()}")

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(status_line("config", False, str(exc)))
        return 1

    print(status_line("config", True, str(args.config)))
    check_paths(config)
    check_packages(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
