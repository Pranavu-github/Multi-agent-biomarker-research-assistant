"""
agents/extraction_agent.py — Agent 3: RAG-Powered Evidence Extraction.

Key design: ONE record per PMID (3–10 rows in the final table).

Pipeline per gene+cell+disease:
  1. Ingest all papers into BiomarkerRAGStore (FAISS + BM25)
  2. Build RAG context (top ranked chunks across papers)
  3. For each unique PMID with enough text:
       - Build PMID-specific mini-context (chunks from that paper only)
       - Call LLM to extract structured evidence
  4. Return list of extracted records

JSON output fields per record (aligned with notebook schema):
    gene, cell_type_query, disease, pmid, source,
    assay, cell_type, tissue, localization,
    evidence_level, confidence, key_sentence

Strict rules:
  - Extract from retrieved context ONLY (no hallucination)
  - Prefer protein-level evidence (IHC, IF, FACS)
  - If no evidence → evidence_level = "no evidence"
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from utils.llm_client import call_llm
from utils.rag_store import BiomarkerRAGStore
from utils.rag_context_builder import build_rag_context


# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are a biomedical evidence extraction expert specialising in biomarker validation.

Extract experimental evidence from the paper text below.

Gene/Protein      : {gene}
Cell Type (query) : {cell}
Cell subtypes     : {cell_subtypes}
Disease           : {disease}
PMID              : {pmid}

════════════════════════════════════════
PAPER TEXT (retrieved from {source}):
════════════════════════════════════════
{context}
════════════════════════════════════════

EXTRACTION RULES:
1. Extract ONLY from the text above — do NOT hallucinate.

2. Prefer protein-level assays: IHC, IF, immunofluorescence, FACS, flow cytometry.

3. {disease_context_rule}

4. CELL TYPE LINEAGE (critical — prevents missing subtype evidence):
   - The query cell type "{cell}" may be a broad lineage term.
   - Specific subtypes of this lineage: {cell_subtypes}
   - ACCEPT evidence from ANY of these subtypes — do NOT return "no evidence"
     just because the paper uses a subtype name instead of the broad term.
   - Examples of valid lineage matches:
       Query "myeloid cells"    → accept: dendritic cells, monocytes, macrophages,
                                          BDCA3+ DCs, CD141+ DCs, cDC1, pDC
       Query "T cells"          → accept: CD4+, CD8+, Treg, CTL, NKT cells
       Query "stromal cells"    → accept: fibroblasts, CAFs, endothelial cells
   - If evidence is found in a subtype, fill "cell_type" with the EXACT subtype
     name from the paper (e.g. "BDCA3+ dendritic cells"), not the broad query term.

5. If the gene has multiple aliases, accept any alias matching the gene.

6. Assay field: use the most specific assay name found (e.g. "FACS" not "flow").

   CRITICAL — TISSUE MICROARRAY (TMA) vs EXPRESSION MICROARRAY:
   These are two completely different technologies. Never confuse them.

   TISSUE MICROARRAY (TMA):
   - A histological FORMAT — multiple tissue cores on one glass slide
   - The actual ASSAY performed on a TMA is ALWAYS IHC or immunostaining
   - Paper language: "TMA sections were stained with antibody", "immunostained TMA"
   - Output: protein localisation ("luminal cells", "cytoplasmic staining")
   - Assay field: extract "IHC" (NOT "tissue microarray")

   GENE EXPRESSION MICROARRAY (mRNA / cDNA / Affymetrix array):
   - Measures mRNA transcript levels — this is RNA-level evidence only
   - Paper language: "gene expression profiling", "hybridised to microarray"
   - Output: fold change, expression values (NOT cell compartment localisation)
   - Assay field: extract "microarray" or "gene expression array"

   DIAGNOSTIC: If the key_sentence describes localisation in specific cell types
   (e.g. "luminal epithelial cells", "cytoplasmic staining") → the assay is IHC,
   regardless of whether the paper also used a TMA format. Extract assay = "IHC".

7. ABSENT / SILENCED EXPRESSION (critical for tumor suppressor genes and loss-of-expression biomarkers):
   - If the text shows that {gene} is ABSENT, LOST, SILENCED, DOWNREGULATED, or
     NOT EXPRESSED in the target cell type via a protein-level assay (IHC, IF,
     immunofluorescence, FACS, flow cytometry) — this IS valid Tier 1 evidence.
   - Set expression_status = "not expressed" and evidence_level = "direct".
   - Set the assay to the specific protein assay used (e.g. "IHC", "FACS").
   - Do NOT return "no evidence" just because the result is negative.
   - Examples:
       "IHC showed absence of SLC5A8 in breast tumor cells" → expression_status="not expressed", assay="IHC"
       "Flow cytometry revealed loss of protein in cancer cells" → expression_status="not expressed", assay="FACS"
       "methylation-PCR showed promoter silencing" → NOT Tier 1 (RT-PCR/methylation-PCR are supplementary only)
       "RT-PCR showed downregulation" → NOT Tier 1 (RNA-level, not protein-level)
   - Supplementary-only assays (NOT eligible for expression_status Tier 1):
       RT-PCR, methylation-PCR, bisulfite sequencing, RNA-seq, qPCR, microarray

8. ADJACENT NORMAL TISSUE (critical for normal cell type queries in cancer context):
   - If the cell type queried is NORMAL (contains "normal", "adjacent", "luminal",
     "basal", "non-tumour", "benign"), AND the disease is provided:
   - ACCEPT papers that study normal tissue as an internal control arm within a
     cancer/disease study. This is the standard IHC study design in oncology.
     Example: "EDN3 is abundant in normal breast epithelium but lost in invasive carcinoma"
     → The paper studies BOTH normal and tumour tissue from the same cancer patients.
   - Extract the evidence specifically for the NORMAL TISSUE arm.
   - Fill cell_type with the exact normal cell type name from the paper
     (e.g. "normal breast epithelium", "luminal epithelial cells").
   - Do NOT reject these papers just because they also report tumour findings.
   - Do NOT report the tumour arm as the main finding — report the NORMAL tissue arm.
   - If the paper describes expression in normal tissue AND loss in tumour, fill:
       expression_status = "expressed" (for the normal tissue arm)
       key_sentence      = the sentence about normal tissue expression
   - The fact that the "normal" tissue comes from cancer patients does NOT disqualify it —
     adjacent-normal biopsies from cancer patients are the accepted source for
     normal tissue IHC reference data in oncology.

Return ONLY valid JSON — no prose, no markdown fences, no explanation:

{{
  "assay":             "IHC / IF / FACS / smFISH / RNA-seq / review / unknown",
  "cell_type":         "exact cell type or subtype name from the text",
  "tissue":            "tissue or organ described",
  "localization":      "nuclear / cytoplasmic / membrane / extracellular / N/A",
  "evidence_level":    "direct / indirect / no evidence",
  "expression_status": "expressed / not expressed / silenced / unknown",
  "confidence":        "high / medium / low",
  "key_sentence":      "most relevant sentence (≤40 words)"
}}

If NO relevant evidence exists, return exactly:
{{
  "assay": "", "cell_type": "", "tissue": "",
  "localization": "", "evidence_level": "no evidence",
  "expression_status": "unknown",
  "confidence": "low", "key_sentence": ""
}}
"""


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> dict | None:
    """Extract first valid JSON object from potentially prose-wrapped LLM output."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())

    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Bracket-depth counting fallback (handles prose wrapping)
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j     = i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i: j + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
            j += 1
        i += 1
    return None


_NO_EVIDENCE_RECORD: dict = {
    "assay": "", "cell_type": "", "tissue": "",
    "localization": "", "evidence_level": "no evidence",
    "expression_status": "unknown",
    "confidence": "low", "key_sentence": "",
}


# ── Per-PMID extraction ───────────────────────────────────────────────────────

def _extract_one_pmid(
    pmid:          str,
    source:        str,
    chunks:        list[dict],
    gene:          str,
    cell:          str,
    disease:       str,
    cell_subtypes: list[str] | None = None,
) -> dict:
    """
    Build mini-context for one PMID and call LLM for extraction.
    Returns raw extracted dict (without gene/pmid metadata — added by caller).
    """
    # Build mini context from this PMID's chunks
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({chunk.get('section', 'Body')}) {chunk['chunk_text']}")
    context = "\n\n".join(lines)

    if not context.strip():
        return _NO_EVIDENCE_RECORD.copy()

    # Format subtypes for the prompt — clear string if none available
    subtypes_str = (
        ", ".join(cell_subtypes)
        if cell_subtypes
        else "none (cell type is already specific — accept exact matches only)"
    )

    # ── Build the disease context rule ────────────────────────────────────────
    # When disease IS provided: accept disease-specific + multi-cancer papers,
    #   reject completely unrelated tissues.
    # When disease is NOT provided: STRICT normal-tissue-only enforcement —
    #   cancer/tumour papers are rejected regardless of gene mention.
    disease_label = disease or "None (normal tissue only)"
    if disease.strip():
        disease_context_rule = (
            f"DISEASE CONTEXT — strict disease-aware extraction:\n"
            f"   - Target disease: {disease}\n"
            f"   - ACCEPT: papers that study {disease} specifically OR multi-cancer papers\n"
            f"     where {disease} is one of several diseases studied.\n"
            f"   - REJECT: papers about a completely unrelated disease/tissue that\n"
            f"     make zero mention of {disease} or closely related terms.\n"
            f"   - Papers purely about NORMAL/healthy tissue with no disease context → no-evidence."
        )
    else:
        disease_context_rule = (
            "TISSUE CONTEXT — NORMAL TISSUE ONLY (no disease specified):\n"
            "   - The user has NOT specified a disease. This query is about NORMAL/healthy tissue.\n"
            "   - STRICT RULE: ONLY extract evidence from normal, healthy, non-diseased tissue.\n"
            "   - REJECT any evidence that comes exclusively from cancer, tumour, or disease\n"
            "     experiments. If the paper only reports findings in cancer/disease conditions\n"
            "     → return the no-evidence JSON.\n"
            "   - ACCEPT: papers studying normal physiology, healthy donors, or normal\n"
            "     tissue used as a control in a disease study (extract the NORMAL arm)."
        )

    prompt = _EXTRACT_PROMPT.format(
        gene=gene, cell=cell,
        cell_subtypes=subtypes_str,
        disease=disease_label,
        disease_context_rule=disease_context_rule,
        pmid=pmid, source=source,
        context=context[:3500],
    )

    try:
        raw = call_llm(prompt, max_tokens=512)
        result = _extract_json_object(raw)
        if result is None:
            logger.warning(f"[extraction] JSON parse failed for PMID {pmid} — using no-evidence")
            return _NO_EVIDENCE_RECORD.copy()
        return result
    except Exception as exc:
        logger.warning(f"[extraction] LLM call failed for PMID {pmid}: {exc}")
        return _NO_EVIDENCE_RECORD.copy()


# ── Public API ────────────────────────────────────────────────────────────────

def extraction_agent(
    papers:        list[dict],
    gene:          str,
    cell:          str,
    disease:       str = "",
    store:         BiomarkerRAGStore | None = None,
    cell_subtypes: list[str] | None = None,
) -> list[dict]:
    """
    Agent 3 — RAG-Powered Evidence Extraction.

    Returns one record per PMID (3–10 rows in the final table).

    Args:
        papers:        List of paper dicts from retrieval_agent.
        gene:          Gene/protein symbol.
        cell:          Cell type (broad or specific).
        disease:       Disease context (empty = normal tissue).
        store:         Pre-built BiomarkerRAGStore (optional).
        cell_subtypes: Specific subtypes of a broad cell type (from planner).
                       Passed to the LLM so it accepts subtype evidence
                       (e.g. "dendritic cells" when querying "myeloid cells").

    Returns:
        List of extracted record dicts with keys:
            gene, cell_type_query, disease, pmid, source,
            assay, cell_type, tissue, localization,
            evidence_level, confidence, key_sentence, rag_chunks_used
    """
    logger.info(
        f"[extraction] Starting for {gene} | {cell} | {disease or 'normal'} | "
        f"{len(papers)} papers | subtypes={len(cell_subtypes or [])} defined"
    )

    if not papers:
        return []

    # ── Step 1: Ingest into RAG store ────────────────────────────────────────
    if store is None:
        from utils.rag_store import get_store, reset_store
        reset_store()
        store = get_store()

    n_chunks = store.ingest(papers, gene=gene, disease=disease)
    logger.info(f"[extraction] Ingested {n_chunks} chunks")

    if n_chunks == 0:
        logger.warning("[extraction] No chunks produced — returning empty")
        return []

    # ── Step 2: Collect unique PMIDs and their chunks ────────────────────────
    # Group chunks by PMID
    pmid_chunks: dict[str, list[dict]] = {}
    pmid_source: dict[str, str]        = {}

    # Retrieve broadly to collect per-PMID chunks
    broad_query = f"{gene} {cell} {disease} expression biomarker IHC IF FACS".strip()
    all_chunks  = store.retrieve(broad_query, top_k=min(n_chunks, 60))

    for chunk in all_chunks:
        pmid = chunk.get("pmid", "N/A")
        if pmid not in pmid_chunks:
            pmid_chunks[pmid] = []
            pmid_source[pmid] = chunk.get("source", "unknown")
        pmid_chunks[pmid].append(chunk)

    # Also include database-type papers (HPA, CellMarker, UniProt)
    db_papers = [p for p in papers if p.get("evidence_type") == "database"]
    for p in db_papers:
        pmid = p.get("pmid", "N/A")
        if pmid not in pmid_source:
            pmid_source[pmid] = p.get("source", "unknown")
        if pmid not in pmid_chunks or not pmid_chunks[pmid]:
            # Build a pseudo-chunk from abstract
            text = p.get("abstract", "") or ""
            if text:
                pmid_chunks[pmid] = [{"chunk_text": text, "section": "Abstract", "pmid": pmid}]

    logger.info(f"[extraction] Extracting from {len(pmid_chunks)} unique PMIDs")

    # ── Step 3: Extract one record per PMID (parallel LLM calls) ────────────
    def _extract_and_wrap(pmid: str) -> dict:
        chunks = pmid_chunks.get(pmid, [])
        source = pmid_source.get(pmid, "unknown")
        pmid_specific_chunks = chunks[:6]  # cap context window
        extracted = _extract_one_pmid(
            pmid=pmid, source=source,
            chunks=pmid_specific_chunks,
            gene=gene, cell=cell, disease=disease,
            cell_subtypes=cell_subtypes,   # pass lineage subtypes through
        )
        return {
            "gene":            gene,
            "cell_type_query": cell,
            "disease":         disease,
            "pmid":            pmid,
            "source":          source,
            "rag_chunks_used": len(pmid_specific_chunks),
            **extracted,
        }

    valid_pmids = [pmid for pmid, chunks in pmid_chunks.items() if chunks]
    records: list[dict] = []

    # max_workers=2: keeps concurrent Groq token usage under the free-tier
    # 6,000 TPM limit.  Each extraction call is ~1,500–2,000 tokens; 2 workers
    # = max ~4,000 tokens in flight at once, leaving headroom for retries.
    # With the retry-backoff in llm_client.py this is robust even under bursts.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="extraction") as pool:
        futures = {pool.submit(_extract_and_wrap, pmid): pmid for pmid in valid_pmids}
        for future in as_completed(futures):
            pmid = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                logger.warning(f"[extraction] PMID {pmid} failed: {exc}")

    logger.info(f"[extraction] Done: {len(records)} records extracted")
    return records
