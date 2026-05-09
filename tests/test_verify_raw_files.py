import importlib.util
from pathlib import Path

import h5py


def load_verify_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m0_00b_verify_raw_files.py"
    spec = importlib.util.spec_from_file_location("m0_00b_verify_raw_files", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_verify_file_detects_html_like_file(tmp_path: Path) -> None:
    module = load_verify_module()
    html_path = tmp_path / "README.md"
    html_path.write_bytes(b"<!DOCTYPE html><html>Forbidden</html>")

    result = module.verify_file("README.md", html_path)

    assert result["html_like"] is True
    assert result["ok"] is False
    assert any("HTML" in error for error in result["errors"])


def test_verify_file_checks_hdf5_magic_and_structure(tmp_path: Path) -> None:
    module = load_verify_module()
    h5ad_path = tmp_path / "adata.h5ad"
    with h5py.File(h5ad_path, "w") as handle:
        handle.create_dataset("X", data=[[1.0, 2.0]])
        handle.create_group("obs")
        handle.create_group("var")

    result = module.verify_file("adata.h5ad", h5ad_path)

    assert result["hdf5_magic"] is True
    assert result["h5py_open_ok"] is True
    assert result["h5ad_structure"]["top_level_keys"] == ["X", "obs", "var"]
    assert result["ok"] is False
    assert any("too small" in error for error in result["errors"])
