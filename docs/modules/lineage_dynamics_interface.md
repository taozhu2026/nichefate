# Lineage Dynamics Interface

The lineage dynamics interface is a design-only contract for future
lineage-aware PlanA and PlanB modules.

## Interface Objects

- `C_cellbin_clone`
- `C_tile_clone`
- `C_niche_clone`

These matrices represent clone composition over the shared NicheFate spatial
units.

## Design Rules

- Clone overlap can support candidate dynamics terms when the dataset has
  time, perturbation, or another valid direction source.
- Clone composition can regularize niche coupling in future models.
- Clone entropy and clone diversity are niche lineage state variables.
- Direction still requires time, perturbation, or biological prior.
- Serial sections are not sufficient for temporal directionality claims.

This interface does not run PlanA or PlanB production.
