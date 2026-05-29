from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .input_contract import (
    ALLELE_ANNOTATION_REQUIRED_COLUMNS,
    LINEAGE_EVIDENCE_REQUIRED_COLUMNS,
    REQUIRED_H5AD_LAYERS,
    REQUIRED_H5AD_OBS_FIELDS,
    REQUIRED_H5AD_OBSM,
    BarcodeInputContract,
)


MANIFEST_NAME = "L126_brain_barcode_aware_input_packet.manifest.tsv"
TRANSFER_CONTRACT_NAME = "nichefate_barcode_adapter_input_contract.json"


@dataclass(frozen=True)
class PacketPaths:
    root: Path
    archive: Path | None
    manifest: Path
    transfer_contract: Path
    h5ad_files: tuple[Path, ...]
    primary_evidence: Path
    allele_annotation: Path
    report_files: tuple[Path, ...]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = destination / member.name
            if not _is_within(target, destination):
                raise ValueError(f"Refusing unsafe tar member path: {member.name}")
        handle.extractall(destination)


def prepare_packet_root(
    contract: BarcodeInputContract,
    *,
    extract_if_needed: bool = True,
) -> PacketPaths:
    """Locate the transferred L126 packet and unpack it once if required."""

    packet_root = contract.packet_root
    if packet_root is None:
        packet_root = Path("/home/zhutao/scratch/nichefate/spatio_darlin_public_brain_hpc/input_packet")
    packet_root = packet_root.expanduser().resolve()

    archive = contract.packet_archive
    if archive is None:
        candidate = Path("/data/zhutao/transfer/L126_brain_barcode_aware_input_packet.tar.gz")
        archive = candidate if candidate.exists() else None
    if archive is not None:
        archive = archive.expanduser().resolve()

    manifest = packet_root / "processed" / "transfer" / MANIFEST_NAME
    if not manifest.exists() and extract_if_needed:
        if archive is None or not archive.exists():
            raise FileNotFoundError("L126 packet archive is missing and packet root is not unpacked")
        _safe_extract_tar(archive, packet_root)

    paths = packet_paths(packet_root, contract, archive)
    missing = [str(path) for path in required_packet_files(paths) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required packet files: " + "; ".join(missing))
    return paths


def packet_paths(
    packet_root: str | Path,
    contract: BarcodeInputContract,
    archive: Path | None = None,
) -> PacketPaths:
    root = Path(packet_root).expanduser().resolve()
    manifest = root / "processed" / "transfer" / MANIFEST_NAME
    transfer_contract = root / "processed" / "transfer" / TRANSFER_CONTRACT_NAME
    return PacketPaths(
        root=root,
        archive=archive,
        manifest=manifest,
        transfer_contract=transfer_contract,
        h5ad_files=tuple(root / relative for relative in contract.expected_h5ad_files),
        primary_evidence=root / contract.primary_evidence_file,
        allele_annotation=root / contract.allele_annotation_file,
        report_files=tuple(root / relative for relative in contract.expected_reports),
    )


def required_packet_files(paths: PacketPaths) -> tuple[Path, ...]:
    return (
        paths.manifest,
        paths.transfer_contract,
        paths.primary_evidence,
        paths.allele_annotation,
        *paths.h5ad_files,
        *paths.report_files,
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_cellbin_lineage_evidence(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", compression="gzip")
    _require_columns(frame, LINEAGE_EVIDENCE_REQUIRED_COLUMNS, "cellbin lineage evidence")
    frame["count"] = pd.to_numeric(frame["count"], errors="raise")
    return frame


def load_feature_allele_annotation(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", compression="gzip")
    _require_columns(frame, ALLELE_ANNOTATION_REQUIRED_COLUMNS, "feature allele annotation")
    frame["n_alleles_for_feature"] = pd.to_numeric(
        frame["n_alleles_for_feature"],
        errors="coerce",
    ).fillna(0).astype(int)
    return frame


def _close_h5ad(adata: Any) -> None:
    file_obj = getattr(adata, "file", None)
    close = getattr(file_obj, "close", None)
    if callable(close):
        close()


def load_l126_h5ad_packet(
    paths: PacketPaths,
    contract: BarcodeInputContract,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Read h5ad metadata in backed mode and return the cellbin index."""

    import anndata as ad

    rows: list[pd.DataFrame] = []
    info_rows: list[dict[str, Any]] = []
    expected_samples = set(contract.sample_list)
    for path in paths.h5ad_files:
        adata = ad.read_h5ad(path, backed="r")
        try:
            obs = adata.obs.reset_index(drop=False).copy()
            missing_obs = [column for column in REQUIRED_H5AD_OBS_FIELDS if column not in obs.columns]
            missing_layers = [layer for layer in REQUIRED_H5AD_LAYERS if layer not in adata.layers.keys()]
            missing_obsm = [key for key in REQUIRED_H5AD_OBSM if key not in adata.obsm.keys()]
            if missing_obs or missing_layers or missing_obsm:
                raise ValueError(
                    f"{path.name} is not Round 1 compatible; "
                    f"missing obs={missing_obs}, layers={missing_layers}, obsm={missing_obsm}"
                )
            sample_values = sorted(obs["sample_id"].astype(str).unique().tolist())
            unexpected = sorted(set(sample_values) - expected_samples)
            if unexpected:
                raise ValueError(f"{path.name} contains unexpected samples: {unexpected}")
            keep = ["sample_id", "slice_id", "cellbin_id", "x", "y"]
            if "section_order" in obs.columns:
                keep.insert(2, "section_order")
            else:
                obs["section_order"] = sample_values[0].rsplit("_s", 1)[-1] if sample_values else ""
                keep.insert(2, "section_order")
            cellbins = obs[keep].copy()
            cellbins["section_order"] = pd.to_numeric(
                cellbins["section_order"],
                errors="coerce",
            ).astype("Int64")
            rows.append(cellbins)
            info_rows.append(
                {
                    "path": str(path),
                    "sample_id": ";".join(sample_values),
                    "n_obs": int(adata.n_obs),
                    "n_vars": int(adata.n_vars),
                    "has_counts_layer": "counts" in adata.layers.keys(),
                    "has_spatial_obsm": "spatial" in adata.obsm.keys(),
                    "required_obs_present": True,
                    "readback_ok": True,
                }
            )
        finally:
            _close_h5ad(adata)

    index = pd.concat(rows, ignore_index=True)
    duplicate_count = int(index.duplicated(["sample_id", "slice_id", "cellbin_id"]).sum())
    if duplicate_count:
        raise ValueError(f"h5ad cellbin index has duplicate primary join keys: {duplicate_count}")
    return index, info_rows
