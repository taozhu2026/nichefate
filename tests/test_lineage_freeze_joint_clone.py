from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import sparse

from nichefate.darlin import (
    build_joint_clone_assignment,
    build_joint_clone_matrix,
    build_validated_joint_clone_summary,
)
from nichefate.lineage import aggregate_joint_clone_to_units


def _valid_alleles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mosaic_allele": "CA_23_24del",
                "locus": "CA",
                "allele_class": "reference_mapped_rare",
                "reference_mapped": True,
                "invalid_alleles": False,
                "normalized_count": 0.01,
                "sample_count": 1,
                "empirical_n_cellbins": 4,
                "empirical_cellbin_fraction": 0.2,
            },
            {
                "mosaic_allele": "TA_23_24insA",
                "locus": "TA",
                "allele_class": "unmapped_de_novo_candidate",
                "reference_mapped": False,
                "invalid_alleles": False,
                "normalized_count": float("nan"),
                "sample_count": float("nan"),
                "empirical_n_cellbins": 3,
                "empirical_cellbin_fraction": 0.12,
            },
            {
                "mosaic_allele": "RA_10_20dup",
                "locus": "RA",
                "allele_class": "reference_mapped_rare",
                "reference_mapped": True,
                "invalid_alleles": False,
                "normalized_count": 0.02,
                "sample_count": 1,
                "empirical_n_cellbins": 2,
                "empirical_cellbin_fraction": 0.08,
            },
        ]
    )


def _clone_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "joint_clone_id": "CA_23_24del@TA_23_24insA@RA_10_20dup",
                "n_cellbins": 3,
                "n_sections": 2,
                "section_distribution": "1:2;2:1",
                "median_joint_prob": 0.1,
                "joint_allele_num": 3,
                "BC_consistency": 1.0,
                "n_loci_present_median": 3.0,
            },
            {
                "joint_clone_id": "dummy_clone",
                "n_cellbins": 97,
                "n_sections": 2,
                "section_distribution": "1:60;2:37",
                "median_joint_prob": 0.1,
                "joint_allele_num": 1,
                "BC_consistency": 1.0,
                "n_loci_present_median": 1.0,
            },
        ]
    )


def _assignment() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "L126_Brain_s1",
                "slice_id": "L126_Brain_s1",
                "section_order": 1,
                "cellbin_id": "c1",
                "joint_clone_id": "CA_23_24del@TA_23_24insA@RA_10_20dup",
            },
            {
                "sample_id": "L126_Brain_s1",
                "slice_id": "L126_Brain_s1",
                "section_order": 1,
                "cellbin_id": "c2",
                "joint_clone_id": "CA_23_24del@TA_23_24insA@RA_10_20dup",
            },
            {
                "sample_id": "L126_Brain_s2",
                "slice_id": "L126_Brain_s2",
                "section_order": 2,
                "cellbin_id": "c3",
                "joint_clone_id": "CA_23_24del@TA_23_24insA@RA_10_20dup",
            },
        ]
    )


def _tile_map() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"sample_id": "L126_Brain_s1", "slice_id": "L126_Brain_s1", "section_order": 1, "tile_id": "tile_1", "tile_x_bin": 0, "tile_y_bin": 0, "cellbin_id": "c1", "x": 0.0, "y": 0.0},
            {"sample_id": "L126_Brain_s1", "slice_id": "L126_Brain_s1", "section_order": 1, "tile_id": "tile_1", "tile_x_bin": 0, "tile_y_bin": 0, "cellbin_id": "c2", "x": 1.0, "y": 0.0},
            {"sample_id": "L126_Brain_s2", "slice_id": "L126_Brain_s2", "section_order": 2, "tile_id": "tile_2", "tile_x_bin": 1, "tile_y_bin": 0, "cellbin_id": "c3", "x": 2.0, "y": 0.0},
        ]
    )
    frame["cell_key"] = frame["sample_id"] + "|" + frame["slice_id"] + "|" + frame["cellbin_id"]
    return frame


def test_joint_clone_summary_and_matrix_are_generic() -> None:
    clone_qc = build_validated_joint_clone_summary(_clone_summary(), _valid_alleles())
    target = clone_qc.loc[clone_qc["joint_clone_id"].eq("CA_23_24del@TA_23_24insA@RA_10_20dup")].iloc[0]
    assert target["reference_support_fraction"] == 2 / 3
    assert target["de_novo_allele_fraction"] == 1 / 3
    assert target["qc_status"] in {"pass", "warning"}

    assignment = build_joint_clone_assignment(_assignment(), clone_qc)
    matrix_root = Path("/tmp/nichefate_lineage_freeze_matrix")
    cell_summary, clone_index, payload = build_joint_clone_matrix(
        _tile_map(),
        assignment,
        clone_qc,
        matrix_root,
        overwrite=True,
    )
    assert payload["matrix_shape"] == [3, 1]
    assert sparse.load_npz(matrix_root / "cellbin_joint_clone_matrix.npz").shape == (3, 1)
    assert len(clone_index) == 1
    assert cell_summary["assignment_status"].isin({"assigned", "filtered_by_clone_qc"}).all()

    comp, summary = aggregate_joint_clone_to_units(
        _tile_map(),
        cell_summary,
        ["sample_id", "slice_id", "section_order", "tile_id", "tile_x_bin", "tile_y_bin"],
    )
    assert not comp.empty
    assert summary["clone_entropy"].ge(0).all()
    assert summary["simpson_clone_diversity"].ge(0).all()
