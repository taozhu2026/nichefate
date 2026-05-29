# L126 Legacy Script Mapping

Legacy L126-specific `planC_l126_*` scripts are retained for provenance only.
Public docs should use the generic module names and wrappers.

| Legacy script | Generic public wrapper |
|---|---|
| `planC_l126_barcode_adapter_round1.py` | `nichefate_lineage_00_validate_input_contract.py` |
| `planC_l126_barcode_adapter_round1.py` | `nichefate_lineage_01_build_evidence_adapter.py` |
| `planC_l126_full_barcode_niche_characterization.py` | `nichefate_lineage_02_characterize_barcode_niches.py` |
| `planC_l126_darlin_style_clone_calling_audit.py` | `nichefate_darlin_03_call_joint_clones.py` |
| `planC_l126_darlin_joint_clone_niche_v1.py` | `nichefate_darlin_03_call_joint_clones.py` |
| `planC_l126_darlin_joint_clone_niche_v1.py` | `nichefate_lineage_04_integrate_clones_to_niches.py` |
| `planC_l126_full_characterization_finalize.py` | `nichefate_lineage_05_finalize_reports.py` |

These wrappers map to E1 LineageEvidenceAdapter, E2 DARLINJointCloneCaller,
and E3 CloneNicheIntegrator in the concrete module registry.
