# NicheFate Pipeline Module Index

This index maps the frozen lineage-aware baseline to the generic module names
used by the public documentation.

## Public Module Map

| NF module | Responsibility | Current implementation surface |
|---|---|---|
| NF-L0 | Lineage input contract | `src/nichefate/lineage/input_contract.py` |
| NF-L1 | Lineage evidence adapter | `src/nichefate/lineage/evidence.py` |
| NF-L2 | DARLIN-style joint clone calling | `src/nichefate/darlin/joint_clone_calling.py` |
| NF-L3 | Clone x niche integration | `src/nichefate/lineage/clone_niche.py` |
| NF-L4 | Lineage-aware spatial niche characterization | `src/nichefate/lineage/visualization.py` and `src/nichefate/lineage/clone_niche.py` |
| NF-L5 | Dynamics interface design | `src/nichefate/lineage/dynamics_interface.py` |

## Benchmark Positioning

L126 spatio-DARLIN is the first benchmark used to validate the lineage-aware
module surface. It contributes processed lineage evidence, matched ST data,
and spatial summaries for the clone and niche layers.

## Legacy Provenance

Legacy L126-specific scripts and reports remain available for provenance, but
they are not the public module names for the frozen baseline.
