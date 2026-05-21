"""
tests/test_pipeline.py — Unit tests for the biomarker pipeline (no Graph-RAG).

Tests cover:
  - input_handler: multiple input formats
  - rag_store: ingest + retrieve + RRF
  - rag_context_builder: context string generation
  - validation_agent: PASS / FAIL / NA classification
  - scoring_agent: Evidence tier + score
  - output_agent: DataFrame structuring
  - planner: anchor queries and disease expansion

Run with:
    python -m pytest tests/test_pipeline.py -v
    # or
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ────────────────────────────────────────────────────────────────────────────
# 1. Input Handler
# ────────────────────────────────────────────────────────────────────────────

def test_input_handler_dict():
    from utils.input_handler import load_input_data
    df = load_input_data({"Gene": "SOX18", "Cell_Type": "Endothelial cells", "Disease": "breast cancer"})
    assert len(df) == 1
    assert df.iloc[0]["Gene"] == "SOX18"
    assert df.iloc[0]["Disease"] == "breast cancer"
    print("  [PASS] input_handler: dict input")


def test_input_handler_string():
    from utils.input_handler import load_input_data
    df = load_input_data("SOX18 + Endothelial cells + breast cancer")
    assert df.iloc[0]["Gene"] == "SOX18"
    assert df.iloc[0]["Disease"] == "breast cancer"
    print("  [PASS] input_handler: string input")


def test_input_handler_no_disease():
    from utils.input_handler import load_input_data
    df = load_input_data({"Gene": "SOX18", "Cell_Type": "Endothelial cells"})
    assert df.iloc[0]["Disease"] == ""
    print("  [PASS] input_handler: no disease")


# ────────────────────────────────────────────────────────────────────────────
# 2. RAG Store
# ────────────────────────────────────────────────────────────────────────────

_SAMPLE_PAPERS = [
    {
        "pmid":    "11111111",
        "source":  "PubMed",
        "sections": [
            {"section": "Abstract", "text": "SOX18 is expressed in endothelial cells detected by immunohistochemistry IHC in breast cancer tumor tissue."},
            {"section": "Results",  "text": "IHC staining confirmed SOX18 protein in CD31-positive endothelial cells of breast carcinoma patients with high confidence."},
        ],
    },
    {
        "pmid":    "22222222",
        "source":  "EuropePMC",
        "sections": [
            {"section": "Abstract", "text": "Flow cytometry FACS revealed SOX18 protein in endothelial cells from breast cancer xenograft models."},
        ],
    },
    {
        "pmid":    "33333333",
        "source":  "SemanticScholar",
        "sections": [
            {"section": "Abstract", "text": "RNA-seq transcriptomic profiling identified SOX18 gene upregulation in endothelial cell populations."},
        ],
    },
]


def test_rag_store_ingest():
    from utils.rag_store import BiomarkerRAGStore
    store = BiomarkerRAGStore()
    n = store.ingest(_SAMPLE_PAPERS, gene="SOX18", disease="breast cancer")
    assert n > 0, "Should produce at least 1 chunk"
    assert store.size == n
    print(f"  [PASS] rag_store: ingest → {n} chunks")


def test_rag_store_retrieve():
    from utils.rag_store import BiomarkerRAGStore
    store = BiomarkerRAGStore()
    store.ingest(_SAMPLE_PAPERS, gene="SOX18", disease="breast cancer")
    results = store.retrieve("SOX18 endothelial IHC breast cancer", top_k=5)
    assert len(results) > 0
    assert all("rrf_score" in r for r in results)
    assert all("chunk_text" in r for r in results)
    print(f"  [PASS] rag_store: retrieve → {len(results)} results with RRF scores")


def test_rag_store_clear():
    from utils.rag_store import BiomarkerRAGStore
    store = BiomarkerRAGStore()
    store.ingest(_SAMPLE_PAPERS, gene="SOX18", disease="breast cancer")
    store.clear()
    assert store.size == 0
    print("  [PASS] rag_store: clear")


# ────────────────────────────────────────────────────────────────────────────
# 3. RAG Context Builder
# ────────────────────────────────────────────────────────────────────────────

def test_rag_context_builder():
    from utils.rag_store import BiomarkerRAGStore
    from utils.rag_context_builder import build_rag_context
    store = BiomarkerRAGStore()
    store.ingest(_SAMPLE_PAPERS, gene="SOX18", disease="breast cancer")
    ctx = build_rag_context(store, gene="SOX18", cell="Endothelial cells", disease="breast cancer")
    assert "[SECTION" in ctx, "Context should contain numbered sections"
    assert "SOX18" in ctx
    print(f"  [PASS] rag_context_builder: {len(ctx)} chars, sections present")


# ────────────────────────────────────────────────────────────────────────────
# 4. Validation Agent
# ────────────────────────────────────────────────────────────────────────────

def test_validation_agent():
    from agents.validation_agent import validation_agent
    records = [
        {"assay": "IHC",     "evidence_level": "direct",      "confidence": "high",   "key_sentence": "expressed by IHC"},
        {"assay": "RNA-seq", "evidence_level": "indirect",     "confidence": "medium", "key_sentence": "mRNA detected"},
        {"assay": "",        "evidence_level": "no evidence",  "confidence": "low",    "key_sentence": ""},
        {"assay": "FACS",    "evidence_level": "direct",       "confidence": "high",   "key_sentence": "FACS protein level"},
    ]
    validated = validation_agent(records)
    statuses = [r["ManualReviewStatus"] for r in validated]
    assert statuses[0] == "PASS",  f"IHC should be PASS, got {statuses[0]}"
    assert statuses[1] == "FAIL",  f"RNA-seq only should be FAIL, got {statuses[1]}"
    assert statuses[2] == "NA",    f"No assay should be NA, got {statuses[2]}"
    assert statuses[3] == "PASS",  f"FACS should be PASS, got {statuses[3]}"
    print("  [PASS] validation_agent: PASS/FAIL/NA correctly assigned")


# ────────────────────────────────────────────────────────────────────────────
# 5. Scoring Agent
# ────────────────────────────────────────────────────────────────────────────

def test_scoring_agent():
    from agents.scoring_agent import scoring_agent
    records = [
        {"assay": "IHC",     "evidence_level": "direct",     "confidence": "high",   "key_sentence": "IHC protein", "source": "PubMed",   "ManualReviewStatus": "PASS"},
        {"assay": "review",  "evidence_level": "indirect",   "confidence": "medium", "key_sentence": "review",      "source": "PubMed",   "ManualReviewStatus": "NA"},
        {"assay": "",        "evidence_level": "no evidence", "confidence": "low",    "key_sentence": "",            "source": "CellMarker", "ManualReviewStatus": "NA"},
    ]
    scored = scoring_agent(records)
    scores = {r["Evidence"]: r["evidence_score"] for r in scored}
    assert "Primary-experimental" in scores
    assert scores["Primary-experimental"] >= 0.8
    assert scores.get("Tertiary-database", 0) <= 0.4
    print(f"  [PASS] scoring_agent: tiers = {list(scores.keys())}")


# ────────────────────────────────────────────────────────────────────────────
# 6. Output Agent
# ────────────────────────────────────────────────────────────────────────────

def test_output_agent():
    from agents.output_agent import output_agent
    records = [
        {
            "gene": "SOX18", "cell_type_query": "Endothelial cells",
            "disease": "breast cancer", "pmid": "11111111", "source": "PubMed",
            "assay": "IHC", "cell_type": "Endothelial", "tissue": "Breast",
            "localization": "nuclear", "evidence_level": "direct",
            "confidence": "high", "key_sentence": "SOX18 expressed by IHC",
            "rag_chunks_used": 3, "Evidence": "Primary-experimental",
            "evidence_score": 1.0, "ManualReviewStatus": "PASS",
        }
    ]
    df = output_agent(records)
    assert len(df) == 1
    assert "Marker Gene" in df.columns
    assert "Evidence Tier" in df.columns
    assert "Validation Status" in df.columns
    print(f"  [PASS] output_agent: {len(df)} rows, {len(df.columns)} columns")


# ────────────────────────────────────────────────────────────────────────────
# 7. Planner — LLM-based expansions (unit tests — no live LLM calls needed)
# ────────────────────────────────────────────────────────────────────────────

def test_planner_anchor_queries_include_cell_type():
    """
    Anchor queries for disease + general cell type (NK, endothelial, myeloid)
    MUST include the cell type term so retrieval finds cell-specific papers.
    Tests _build_anchor_queries() directly — pure logic, no LLM call needed.
    """
    from agents.planner_agent import _build_anchor_queries

    # Simulate: IFNGR1 + Natural killer + NSCLC (already expanded by LLM to full name)
    disease_expanded = "non-small cell lung cancer"
    synonyms = ["IFNGR1", "CD119", "IFNGR"]
    queries  = _build_anchor_queries("IFNGR1", "Natural killer", disease_expanded, synonyms)

    assert queries, "Anchor queries should not be empty"

    # At least one anchor must contain the cell type term
    ct_present = any("natural killer" in q.lower() for q in queries)
    assert ct_present, (
        "No anchor query contains 'natural killer' — cell type missing.\n"
        f"Queries: {queries}"
    )
    # Disease must appear in expanded form (not abbreviated)
    disease_present = any("non-small cell lung cancer" in q.lower() for q in queries)
    assert disease_present, (
        "No anchor query contains expanded disease name.\n"
        f"Queries: {queries}"
    )

    print(f"  [PASS] planner: anchor queries include cell type and expanded disease")
    for q in queries:
        print(f"         {q}")


def test_planner_anchor_queries_normal_context():
    """
    Anchor queries for normal tissue context use adjacent-normal IHC pattern.
    _build_anchor_queries — no LLM call.
    """
    from agents.planner_agent import _build_anchor_queries

    queries = _build_anchor_queries(
        "EDN3", "normal breast epithelium", "breast cancer",
        ["EDN3", "endothelin-3"]
    )
    assert queries, "Should produce anchor queries"
    # Normal context should include normal/adjacent-normal terms
    flat = " ".join(queries).lower()
    assert "normal" in flat or "adjacent" in flat, \
        f"Normal-context anchors missing 'normal/adjacent': {queries}"
    print(f"  [PASS] planner: normal-context anchor queries correct")


def test_planner_expand_disease_empty_string():
    """
    expand_disease('') returns ('', []) — no LLM call for normal tissue context.
    """
    from agents.planner_agent import expand_disease
    canonical, synonyms = expand_disease("")
    assert canonical == "", "Empty disease should stay empty"
    assert synonyms  == [], "Empty disease should have no synonyms"
    print("  [PASS] planner: expand_disease('') → ('', []) correctly")


# ────────────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧬 Biomarker Pipeline — Unit Tests (no Graph-RAG)\n")

    tests = [
        test_input_handler_dict,
        test_input_handler_string,
        test_input_handler_no_disease,
        test_rag_store_ingest,
        test_rag_store_retrieve,
        test_rag_store_clear,
        test_rag_context_builder,
        test_validation_agent,
        test_scoring_agent,
        test_output_agent,
        test_planner_anchor_queries_include_cell_type,
        test_planner_anchor_queries_normal_context,
        test_planner_expand_disease_empty_string,
    ]

    groups = {
        "Input Handler":  tests[0:3],
        "RAG Store":      tests[3:6],
        "RAG Context":    tests[6:7],
        "Validation":     tests[7:8],
        "Scoring":        tests[8:9],
        "Output":         tests[9:10],
        "Planner":        tests[10:],
    }

    passed = 0
    failed = 0
    for group, group_tests in groups.items():
        print(f"\n── {group} {'─' * (50 - len(group))}")
        for t in group_tests:
            try:
                t()
                passed += 1
            except Exception as exc:
                print(f"  [FAIL] {t.__name__}: {exc}")
                import traceback; traceback.print_exc()
                failed += 1

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed / {failed} failed / {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    sys.exit(0 if failed == 0 else 1)
