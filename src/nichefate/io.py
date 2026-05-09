"""Configuration, path, and AnnData I/O helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only outside the env
    yaml = None


def project_root() -> Path:
    """Return the repository root for an editable checkout."""

    return Path(__file__).resolve().parents[2]


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    return config_path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    if yaml is None:
        raise RuntimeError("PyYAML is required to load nichefate config files.")

    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def paths_from_config(config: dict[str, Any]) -> dict[str, Path]:
    """Return configured M0 paths as Path objects."""

    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Config section 'paths' must be a mapping.")

    return {
        key: Path(value)
        for key, value in paths.items()
        if isinstance(value, str) and value.startswith("/")
    }


def storage_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Return storage-like paths, supporting old and new config schemas."""

    if "paths" in config:
        paths = paths_from_config(config)
        return {
            "raw_dataset_root": paths["raw_dir"],
            "external_root": paths["external_dir"],
            "active_output_root": paths["output_dir"],
            "cache_root": paths["output_dir"].parent / "cache",
            "tmp_root": paths["output_dir"].parent / "tmp",
            "future_ssd_output_root": paths["future_ssd_output_dir"],
        }

    storage = config.get("storage", {})
    if not isinstance(storage, dict):
        raise ValueError("Config section 'storage' must be a mapping.")

    paths: dict[str, Path] = {}
    for key, value in storage.items():
        if isinstance(value, str) and value.startswith("/"):
            paths[key] = Path(value)
    return paths


def expected_raw_files(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return required and optional raw file names from config."""

    raw_files = config.get("raw_files", {})
    if not isinstance(raw_files, dict):
        raise ValueError("Config section 'raw_files' must be a mapping.")

    if "adata_main" in raw_files:
        required = [
            raw_files["adata_main"],
            raw_files["adata_day35"],
            raw_files["readme"],
        ]
        optional = [raw_files["ligand_receptor"]]
        return required, optional

    required = raw_files.get("required", [])
    optional = raw_files.get("optional", [])
    return list(required), list(optional)


def ensure_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Create configured lightweight directories and return key paths."""

    paths = paths_from_config(config)
    output_dir = paths["output_dir"]
    created = {
        "raw_dir": paths["raw_dir"],
        "external_dir": paths["external_dir"],
        "output_dir": output_dir,
        "processed": output_dir / "processed",
        "by_time": output_dir / "by_time",
        "by_slice": output_dir / "by_slice",
        "graphs": output_dir / "graphs",
        "reports": output_dir / "reports",
        "logs": output_dir / "logs",
        "cache": output_dir.parent / "cache",
        "tmp": output_dir.parent / "tmp",
    }
    for path in created.values():
        path.mkdir(parents=True, exist_ok=True)
    return created


def file_size_gb(path: str | Path) -> float:
    """Return file size in GiB, or 0 for a missing path."""

    file_path = Path(path)
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024**3)


def read_h5ad(path: str | Path, backed: str | None = None):
    """Read an AnnData h5ad file."""

    import anndata as ad

    return ad.read_h5ad(Path(path), backed=backed)


def write_h5ad_safely(adata: Any, path: str | Path) -> Path:
    """Write an AnnData file via a temporary sibling and atomic replace."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    adata.write_h5ad(temp_path)
    os.replace(temp_path, output_path)
    return output_path
