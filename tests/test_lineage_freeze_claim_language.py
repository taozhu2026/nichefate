from __future__ import annotations

from pathlib import Path

from nichefate.darlin_clone_signature.reporting import positive_claim_hits


def test_frozen_docs_contain_no_positive_claim_language() -> None:
    roots = [
        Path("README.md"),
        Path("docs/index.md"),
        Path("docs/pipeline_module_index.md"),
        Path("docs/modules"),
        Path("docs/benchmarks/l126_spatiodarlin.md"),
        Path("docs/legacy"),
        Path("reports/benchmarks/l126_spatiodarlin"),
        Path("reports/freeze_lineage_aware_v1"),
    ]
    text_parts: list[str] = []
    for root in roots:
        if root.is_file():
            text_parts.append(root.read_text(encoding="utf-8"))
        elif root.is_dir():
            for path in sorted(root.glob("*.md")):
                text_parts.append(path.read_text(encoding="utf-8"))
    text = "\n".join(text_parts)
    assert positive_claim_hits(text) == []
