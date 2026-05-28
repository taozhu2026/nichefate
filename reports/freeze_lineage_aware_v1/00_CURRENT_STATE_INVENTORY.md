# Current State Inventory

## Generic Mainline Candidates

- `src/nichefate/lineage/*`
- `src/nichefate/darlin/*`
- `docs/modules/*`
- `docs/benchmarks/l126_spatiodarlin.md`
- `configs/lineage/*`
- `configs/darlin/*`
- `configs/datasets/l126_spatiodarlin.yaml`
- `scripts/nichefate_*`

## L126 Benchmark Specific

- `scripts/planC_l126_*.py`
- `reports/l126_*darlin*`
- `reports/l126_full_barcode_niche_characterization*`
- `reports/benchmarks/l126_spatiodarlin/*`
- `docs/legacy/l126_planC_script_mapping.md`

## Legacy Or Ablation

- `src/nichefate/planA_k/*`
- `src/nichefate/planA_l/*`
- `src/nichefate/planA_st_only/*`
- `src/nichefate/planB_nichebranchsbm/*`
- `reports/l126_plana_*`
- `docs/legacy/lineage_clone_ablation_history.md`

## Superseded Implementation Carriers

- `src/nichefate/darlin_joint_clone_niche_v1.py`
- `scripts/planC_l126_darlin_joint_clone_niche_v1.py`
- `scripts/planC_l126_clone_membership_rescue_round2_1.py`
- `scripts/planC_l126_darlin_style_clone_calling_audit.py`

## Exclude From GitHub

- raw FASTQ and other raw packet inputs
- large processed tables and matrices
- scratch outputs
- generated `.h5ad`, `.mtx`, `.npz`, and large `.tsv.gz` artifacts
