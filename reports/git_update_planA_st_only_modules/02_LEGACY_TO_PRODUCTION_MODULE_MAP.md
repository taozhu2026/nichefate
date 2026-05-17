# Legacy To Production Module Map

|legacy_milestone|production_module|status_or_note|
|---|---|---|
|M0|SpatialDatasetAdapter|Stable ST input adapter and direct re-export.|
|M1|NicheBuilder|Stable anchor-centered niche construction re-export.|
|M2|NicheEncoder|Stable aligned niche representation re-export.|
|M2.5|MetanicheCoarsener / NicheStateCoarsener|Stable metaniche state coarsening re-export.|
|M3-v1|TransitionEvidence[pseudo_broad]|Legacy broad pseudo-transition evidence re-export.|
|M3-v2|TransitionEvidence[pseudo_sharpened]|Legacy sharpened pseudo-transition evidence re-export.|
|M4A|KernelAssembly|Frozen corrected feature-only Kmix_A outputs indexed; clean re-export pending.|
|K_gpcca|GPCCAMacrostateInference|Frozen corrected full GPCCA outputs indexed; clean re-export pending.|
|M4C / P_fate|EndpointMarkovInference / FateProbability|Frozen P_fate context plus Kmix_A absorption outputs indexed.|
|M4E|BiologicalAnnotation|Frozen annotation and role diagnostics indexed; clean re-export pending.|
|Visualization scripts|ResultVisualization|Frozen figure generation and QA outputs indexed; clean re-export pending.|
|Final result package|ResultPackage / FreezePackage|Frozen final ST-only result package indexed; clean re-export pending.|
|Future DARLIN adapter|BarcodeEvidenceAdapter|Future extension, excluded from this freeze.|
