"""
utils/rag_context_builder.py — Build LLM-ready context from the RAG store.

Takes a BiomarkerRAGStore, fires multiple targeted queries,
deduplicates + prioritises results, and returns a formatted
context string ready for the extraction LLM.

Section priority: Results > Methods > Discussion > Body > others
"""

from __future__ import annotations

from loguru import logger
from utils.rag_store import BiomarkerRAGStore

# ── Query templates ───────────────────────────────────────────────────────────
# General evidence queries
_GENERAL_QUERIES = [
    "{gene} {cell} immunohistochemistry staining expression",
    "{gene} {cell} immunofluorescence protein localization",
    "{gene} {cell} flow cytometry FACS protein detection",
    "{gene} {cell} protein expression biomarker",
    "{gene} {cell} marker validation experimental evidence",
    "{gene} smFISH RNA localization single molecule",
]

# Disease-aware queries (added when disease is present)
_DISEASE_QUERIES = [
    "{gene} {cell} {disease} expression",
    "{gene} {disease} immunohistochemistry clinical",
    "{gene} {disease} protein biomarker patient",
]

# Section-targeted queries
_SECTION_QUERIES = [
    "{gene} {cell} methods immunohistochemistry antibody protocol",
    "{gene} {cell} results expression level statistical",
    "{gene} {disease} results significant association",
]

# Section priority weight (higher = shown first in context)
_SECTION_PRIORITY = {
    "Results":      4,
    "Methods":      3,
    "Discussion":   2,
    "Conclusion":   1,
    "Abstract":     1,
    "Body":         0,
}


def build_rag_context(
    store:       BiomarkerRAGStore,
    gene:        str,
    cell:        str,
    disease:     str = "",
    top_k_query: int = 5,
    max_chunks:  int = 15,
    pmid_cap:    int = 4,      # max chunks per PMID to ensure diversity
) -> str:
    """
    Build a numbered LLM context string from the RAG store.

    Args:
        store:       Populated BiomarkerRAGStore.
        gene:        Gene/protein name.
        cell:        Cell type.
        disease:     Disease context (empty = normal tissue).
        top_k_query: Chunks retrieved per query.
        max_chunks:  Max chunks in final context.
        pmid_cap:    Max chunks allowed per PMID (diversity cap).

    Returns:
        Formatted string with numbered [SECTION N] blocks,
        or empty string if store has no relevant content.
    """
    if store.size == 0:
        return ""

    # Build query list
    queries = [q.format(gene=gene, cell=cell, disease=disease)
               for q in _SECTION_QUERIES]
    queries += [q.format(gene=gene, cell=cell, disease=disease)
                for q in _GENERAL_QUERIES]
    if disease:
        queries += [q.format(gene=gene, cell=cell, disease=disease)
                    for q in _DISEASE_QUERIES]

    # Retrieve and deduplicate
    seen_texts: set[str] = set()
    all_chunks: list[dict] = []

    for query in queries:
        results = store.retrieve(query, top_k=top_k_query)
        for chunk in results:
            key = chunk["chunk_text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_chunks.append(chunk)

    if not all_chunks:
        logger.warning(f"[rag_context] No chunks retrieved for {gene} / {cell} / {disease}")
        return ""

    # Sort: high-value sections first, then RRF score
    all_chunks.sort(
        key=lambda c: (
            _SECTION_PRIORITY.get(c.get("section", "Body"), 0),
            c.get("rrf_score", 0),
        ),
        reverse=True,
    )

    # PMID diversity cap — prevent one paper dominating context
    pmid_count: dict[str, int] = {}
    selected: list[dict] = []
    for chunk in all_chunks:
        pmid = chunk.get("pmid", "unknown")
        if pmid_count.get(pmid, 0) >= pmid_cap:
            continue
        pmid_count[pmid] = pmid_count.get(pmid, 0) + 1
        selected.append(chunk)
        if len(selected) >= max_chunks:
            break

    logger.info(
        f"[rag_context] {gene}/{cell}/{disease or 'normal'}: "
        f"{len(all_chunks)} candidates → {len(selected)} chunks selected"
    )

    # Format as numbered sections
    lines = [
        "=== RETRIEVED EVIDENCE CONTEXT ===",
        f"Gene: {gene}  |  Cell Type: {cell}"
        + (f"  |  Disease: {disease}" if disease else "  |  Context: Normal tissue"),
        "",
    ]
    for i, chunk in enumerate(selected, start=1):
        lines.append(f"[SECTION {i}]")
        lines.append(
            f"Source: {chunk.get('source', '?')} | "
            f"PMID: {chunk.get('pmid', '?')} | "
            f"Section: {chunk.get('section', '?')}"
        )
        lines.append(chunk["chunk_text"])
        lines.append("")

    return "\n".join(lines)
