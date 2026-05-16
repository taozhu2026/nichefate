# M2.5 State Contract v2

        This contract supersedes the first M2.5 pilot contract for sparse-K preparation.
        Sparse-K should consume metaniche states only after coordinate rescue, purity QC,
        spatial compactness QC, and rare-state preservation checks are attached.

        | level | field_or_file | required | source | description |
| --- | --- | --- | --- | --- |
| anchor | anchor_id | True | derived | Stable `slice_id::anchor_index` identifier. |
| anchor | slice_id; anchor_index; anchor_cell_id | True | M2/M1 | Primary join key for M2 features and M1 coordinates. |
| anchor | x; y | True | M1 | Anchor-level spatial coordinates rescued from M1. |
| anchor | time; time_day; mouse_id | True | M2 | Temporal/sample metadata for directional sparse-K. |
| metaniche | anchor_to_metaniche.tsv | True | M2.5 | Anchor provenance and metaniche assignment. |
| metaniche | metaniche_table.tsv | True | M2.5 | State table with size, dominant metadata, and purity. |
| metaniche | metaniche_feature_centroids.csv | True | M2.5 | Feature centroid matrix for sparse-K candidate edges. |
| metaniche | metaniche_coordinates.preview.tsv | True | coordinate rescue | Metaniche x/y centroid and coordinate variance. |
| qc | spatial compactness QC | True | hardening | Radius distribution and diffuse-state flags. |
| qc | rare-state preservation audit | True | hardening | Rare-label collapse/enrichment warnings. |
| optional_annotation | metaniche_composition.tsv | False | M2.5 | Cell-type label composition for biological annotation. |
| production_blocker | full production M2.5 | True | future | Current outputs are sampled pilots and cannot support full GPCCA claims. |

        ## Required Files For Sparse-K

        - `anchor_to_metaniche.tsv`
        - `metaniche_table.tsv`
        - `metaniche_feature_centroids.csv`
        - `metaniche_coordinates.preview.tsv`
        - `05_rare_state_preservation_audit.tsv`
        - `04_spatial_compactness_qc.tsv`

        ## Blockers For Full Production

        - This is still a sampled pilot.
        - Production M2.5 must run across all intended slices with the same coordinate contract.
        - Rare states and diffuse metaniches require review before any GPCCA claim.
