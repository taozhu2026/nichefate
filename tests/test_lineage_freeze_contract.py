from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nichefate.darlin import freeze_selected_joint_clone_policy
from nichefate.lineage import (
    EXPECTED_ASSAYS,
    LineageInputContract,
    draft_lineage_input_contract_payload,
    load_lineage_input_contract,
    validate_lineage_h5ad_schema,
)


def test_lineage_input_contract_parses_generic_join_key(tmp_path: Path) -> None:
    contract_path = tmp_path / "lineage_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "primary_join_key": ["sample_id", "slice_id", "cellbin_id"],
                "assay_list": ["CA", "TA", "RA"],
                "sample_list": ["L126_Brain_s1", "L126_Brain_s2", "L126_Brain_s3"],
                "excluded_samples": [],
                "expected_h5ad_files": [
                    "processed/h5ad/L126_Brain_s1.mRNA_processed.h5ad",
                ],
                "lineage_evidence_files": {
                    "primary": "processed/lineage_evidence/cellbin_lineage_evidence.tsv.gz",
                    "allele_annotation": "processed/lineage_evidence/feature_allele_annotation_long.tsv.gz",
                },
            }
        ),
        encoding="utf-8",
    )
    contract = load_lineage_input_contract(contract_path)
    assert isinstance(contract, LineageInputContract)
    assert contract.primary_join_key == ("sample_id", "slice_id", "cellbin_id")
    assert contract.assay_list == ("CA", "TA", "RA")
    assert set(contract.assay_list) == set(EXPECTED_ASSAYS)
    payload = draft_lineage_input_contract_payload(contract)
    assert payload["section_interpretation"] == "L126_Brain_s1/s2/s3 are serial sections, not timepoints"


def test_benchmark_config_loads_and_binds_selected_policy() -> None:
    config_path = Path("configs/datasets/l126_spatiodarlin.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["dataset_id"] == "L126_SPATIODARLIN"
    assert config["sample_ids"] == ["L126_Brain_s1", "L126_Brain_s2", "L126_Brain_s3"]
    assert freeze_selected_joint_clone_policy()["reference_bank_policy"] == "gr"


def test_h5ad_schema_accepts_gene_expression_layout(tmp_path: Path) -> None:
    anndata = pytest.importorskip("anndata")

    obs = pd.DataFrame(
        {
            "sample_id": ["L126_Brain_s1", "L126_Brain_s1"],
            "slice_id": ["L126_Brain_s1", "L126_Brain_s1"],
            "cellbin_id": ["c1", "c2"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        }
    )
    adata = anndata.AnnData(X=pd.DataFrame([[1, 0], [0, 1]]).to_numpy(), obs=obs)
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["spatial"] = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]]).to_numpy()
    adata.uns["modality"] = "Gene Expression"
    path = tmp_path / "gene_expression.h5ad"
    adata.write_h5ad(path)
    schema = validate_lineage_h5ad_schema(path)
    assert schema["schema_passed"] is True
    assert schema["has_counts_layer"] is True
    assert schema["has_spatial_obsm"] is True
