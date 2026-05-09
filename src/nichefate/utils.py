"""Small utility helpers shared by scripts and modules."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path


def package_available(module_name: str) -> bool:
    """Return whether a Python module can be imported."""

    return importlib.util.find_spec(module_name) is not None


def require_directory(path: str | Path, label: str) -> Path:
    """Validate that a directory exists."""

    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {directory}")
    return directory


def status_line(label: str, ok: bool, detail: str = "") -> str:
    """Format a simple check result."""

    status = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    return f"[{status}] {label}{suffix}"


def setup_file_logger(name: str, log_path: str | Path) -> logging.Logger:
    """Create a simple file logger."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def safe_filename(value: object) -> str:
    """Return a filesystem-safe filename stem."""

    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
