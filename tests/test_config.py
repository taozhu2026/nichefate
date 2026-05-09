from pathlib import Path

from nichefate.io import expected_raw_files, load_config, paths_from_config


def test_config_uses_nichefate_paths() -> None:
    config = load_config("configs/m0_merfish_colitis.yaml")
    paths = paths_from_config(config)

    assert config["project"]["name"] == "nichefate"
    assert config["project"]["root"] == "/home/zhutao/projects/nichefate"
    assert paths["raw_dir"].as_posix() == (
        "/data/zhutao/datasets/merfish_colitis_moffitt_2024/raw"
    )
    assert paths["output_dir"].as_posix() == "/data/zhutao/work/nichefate/m0"
    assert paths["future_ssd_output_dir"].as_posix() == "/ssd/zhutao/nichefate/m0"
    assert config["paths"]["use_ssd"] is False


def test_config_does_not_reference_old_project_name() -> None:
    root = Path(__file__).resolve().parents[1]
    old_name = "nichefate" + "-bridge"
    checked_suffixes = {".md", ".py", ".yaml", ".yml", ".txt", ".toml"}
    offenders = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in checked_suffixes:
            if old_name in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(root).as_posix())
    assert offenders == []


def test_expected_raw_files_from_v1_config() -> None:
    config = load_config("configs/m0_merfish_colitis.yaml")
    required, optional = expected_raw_files(config)

    assert required == ["adata.h5ad", "adata_day35.h5ad", "README.md"]
    assert optional == ["ligand_receptor_pair_masterlist.csv"]
    assert set(config["download"]["files"]) == set(required + optional)
