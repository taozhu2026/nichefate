# M0-M2.5 Validation

- Generated: 2026-05-16T02:06:33.015221+00:00
- Scope: M0-M2.5 backbone validation only
- Decision: `M0_M2_5_VALIDATION_PASSED`

## Commands

| command | status | details |
| --- | --- | --- |
| `python -m py_compile <selected M0/M1/M2/M2.5 modules and scripts>` | passed | returncode=0 |
| `conda run --no-capture-output -n omicverse python -m pytest -q tests/test_config.py tests/test_verify_raw_files.py tests/test_graph.py tests/test_spatial.py tests/test_metadata.py tests/test_niche.py tests/test_niche_qc.py tests/test_m2_representation.py tests/test_m2_full_runner.py tests/test_planA_k_metaniche_pilot.py tests/test_planA_k_metaniche_hardening.py tests/test_planA_k_full_m2_5_production.py tests/test_planA_k_production_preflight.py` | passed | returncode=0; tests_passed=68; warnings=1 |

## Skipped By Design
- Kmix_A tests
- GPCCA tests
- macrostate annotation tests
- absorption tests
- DARLIN tests
- full M2.5 production computation
- Kmix_A production
- GPCCA production

## Safety Checks

- Production data touched: `False`
- Raw data modified: `False`
- Scratch outputs modified: `False`
- /ssd output used: `False`
