"""
agents/planner_agent.py — Agent 1: Query Planner + Intelligent Expansion.

Responsibilities:
  1. Expand gene    → synonyms/aliases              (LLM)
  2. Expand cell type → subtypes/synonyms           (LLM)
  3. Expand disease   → full name + synonyms        (LLM)
  4. Build anchor queries (deterministic floor)
  5. Generate additional LLM search queries
  6. Return a plan dict for downstream agents

All three expansions use the same LLM-call pattern — no hardcoded lookup
tables.  This makes the planner work correctly for ANY gene, cell type, or
disease the user queries, including rare or novel entities.

The three LLM calls (gene, cell, disease) are fired concurrently via
ThreadPoolExecutor so total latency is max(individual call) not their sum.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from utils.llm_client import call_llm

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "planner_prompt.txt")


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _safe_json_load(text: str) -> dict:
    """Parse LLM JSON, stripping markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text.strip())


# ── Sub-agent 1a: Gene synonym expansion ─────────────────────────────────────

def expand_gene_synonyms(gene: str) -> list[str]:
    """
    LLM: expand a gene symbol to all known aliases and protein names.

    Returns list of unique terms including the canonical gene symbol.
    Falls back to [gene] if the LLM call fails.
    """
    prompt = f"""You are a biomedical nomenclature expert.

For the gene: {gene}

Return ALL known synonyms, aliases, and protein names including:
- Official HGNC gene symbol
- Previous gene symbols
- Protein name(s)
- Common abbreviations used in literature
- UniProt entry names (if well-known)

IMPORTANT — gene family disambiguation:
- Only include aliases that refer to EXACTLY this gene.
- Do NOT include aliases for related paralogs or family members.
  Example: if gene is KLRC2 (encodes NKG2C), do NOT include NKG2A or KLRC1 —
  they are different genes in the same family.
- Apply this rule to any gene: list only its own aliases, not its relatives.

Return ONLY valid JSON — no preamble, no markdown:

{{
  "gene": "{gene}",
  "synonyms": ["alias1", "alias2", "protein_name"]
}}
"""
    try:
        text = call_llm(prompt, max_tokens=400, temperature=0.1, use_fast=False)
        data = _safe_json_load(text)
        synonyms = data.get("synonyms", [])
        all_terms = [gene] + [s for s in synonyms if s.lower() != gene.lower()]
        seen, unique = set(), []
        for t in all_terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        logger.info(f"[planner] Gene synonyms for {gene}: {unique}")
        return unique
    except Exception as exc:
        logger.warning(f"[planner] Synonym expansion failed for {gene}: {exc}")
        return [gene]


# ── Sub-agent 1b: Cell type subtype expansion ─────────────────────────────────

def expand_cell_subtypes(celltype: str) -> list[str]:
    """
    LLM: expand a cell type term to specific subtypes and synonyms.

    Works for any cell type — broad lineages (myeloid, T cells) or
    specific subtypes (NK cells, plasmablasts, Kupffer cells).

    Returns [] if the cell type is already maximally specific with no
    known subtypes (e.g. a very rare or novel cell population).
    Falls back to [] on LLM failure (caller handles gracefully).

    Examples:
        "myeloid cells"   → ["dendritic cells", "monocytes", "macrophages", ...]
        "natural killer"  → ["NK cells", "CD56bright NK cells", "CD56dim NK", ...]
        "T cells"         → ["CD4+ T cells", "CD8+ T cells", "Treg", "CTL", ...]
        "Kupffer cells"   → ["liver macrophages", "hepatic macrophages", "KCs"]
        "SOX18"           → []  (not a cell type — LLM returns empty list)
    """
    prompt = f"""You are an expert cell biologist with deep knowledge of human cell biology,
immunology, and cancer biology.

Cell type query: "{celltype}"

Your task:
1. Determine if this is a broad lineage term OR a specific cell population.
2. If broad → return all relevant specific subtypes and synonyms used in biomedical literature.
3. If specific → return synonyms and common abbreviations (e.g. "NK cells" → ["natural killer cells", "CD56+ cells"]).
4. If this is NOT a cell type (e.g. it is a gene name like "SOX18") → return an empty subtypes list.

Include:
- Official names and abbreviations (e.g. "NK cells" and "natural killer cells")
- Surface marker-based names (e.g. "CD56+ cells", "CD8+ T cells")
- Functional names (e.g. "cytotoxic T lymphocytes", "tumor-infiltrating lymphocytes")
- Tissue-specific variants relevant to cancer/immunology (e.g. "tumor-infiltrating NK cells")
- Common research terms found in PubMed abstracts

Do NOT include:
- Cell types from completely different lineages
- Made-up or speculative names not used in real papers
- More than 10 subtypes (keep it focused and high-quality)

Return ONLY valid JSON — no preamble, no markdown:

{{
  "cell_type": "{celltype}",
  "is_broad_lineage": true,
  "subtypes": ["subtype1", "synonym1", "abbreviation1"]
}}

If no relevant subtypes or synonyms exist, return:
{{
  "cell_type": "{celltype}",
  "is_broad_lineage": false,
  "subtypes": []
}}
"""
    try:
        text = call_llm(prompt, max_tokens=500, temperature=0.1, use_fast=False)
        data = _safe_json_load(text)
        subtypes = data.get("subtypes", [])
        # Deduplicate, preserve order, remove the query term itself if present
        seen, unique = set(), []
        for s in subtypes:
            if s.lower() not in seen and s.lower() != celltype.lower():
                seen.add(s.lower())
                unique.append(s)
        if unique:
            logger.info(
                f"[planner] Cell type '{celltype}' "
                f"({'broad' if data.get('is_broad_lineage') else 'specific'}) "
                f"→ {len(unique)} subtypes/synonyms"
            )
        else:
            logger.info(f"[planner] Cell type '{celltype}' → no subtypes (already specific or not a cell type)")
        return unique
    except Exception as exc:
        logger.warning(f"[planner] Cell subtype expansion failed for '{celltype}': {exc}")
        return []


# ── Sub-agent 1c: Disease expansion ──────────────────────────────────────────

def expand_disease(disease: str) -> tuple[str, list[str]]:
    """
    LLM: expand a disease term to its full canonical name and synonyms.

    Returns (canonical_name, [synonym1, synonym2, ...])

    Handles:
    - Abbreviations: "NSCLC" → "non-small cell lung cancer"
    - Shorthand:     "breast cancer" → keeps as-is, returns MeSH/oncology synonyms
    - Empty string:  "" → ("", []) — normal tissue context, no expansion

    Falls back to (disease, []) on LLM failure.
    """
    if not disease.strip():
        return "", []

    prompt = f"""You are a clinical oncology expert with knowledge of disease nomenclature,
ICD codes, MeSH terms, and cancer biology terminology.

Disease query: "{disease}"

Your task:
1. Identify the canonical full name of this disease.
2. Expand abbreviations (e.g. NSCLC → non-small cell lung cancer).
3. Return synonyms and alternative names used in biomedical literature and PubMed searches.

Include:
- Full official name (canonical)
- Common abbreviations (e.g. "NSCLC", "HCC")
- MeSH terms used by PubMed
- Histological subtypes if the input is a broad term (e.g. "lung cancer" → NSCLC, SCLC, adenocarcinoma)
- Related terms that papers use when studying this disease
  (e.g. "non-small cell lung cancer" papers also use "lung carcinoma", "NSCLC patients")

Do NOT include:
- Unrelated diseases
- More than 8 synonyms (keep it focused)

Return ONLY valid JSON — no preamble, no markdown:

{{
  "disease": "{disease}",
  "canonical_name": "full official name here",
  "synonyms": ["synonym1", "abbreviation1", "alternate_name1"]
}}
"""
    try:
        text = call_llm(prompt, max_tokens=400, temperature=0.1, use_fast=False)
        data = _safe_json_load(text)
        canonical = data.get("canonical_name", disease).strip()
        # Guard: if LLM returns empty canonical (rate-limited response, partial
        # JSON, or hallucination), fall back to the original user-supplied disease
        # string so it is NEVER silently dropped from downstream pipeline steps.
        if not canonical:
            logger.warning(
                f"[planner] Disease expansion returned empty canonical for '{disease}' "
                "— using original input as canonical name"
            )
            canonical = disease
        synonyms  = data.get("synonyms", [])
        # Deduplicate, keep canonical separate
        seen: set[str] = {canonical.lower(), disease.lower()}
        unique_syns: list[str] = []
        for s in synonyms:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique_syns.append(s)
        if canonical.lower() != disease.lower():
            logger.info(f"[planner] Disease expanded: '{disease}' → '{canonical}'")
        logger.info(f"[planner] Disease synonyms: {[canonical] + unique_syns}")
        return canonical, unique_syns
    except Exception as exc:
        logger.warning(f"[planner] Disease expansion failed for '{disease}': {exc}")
        return disease, []


# ── Sub-agent 1d: Query generation ───────────────────────────────────────────

def _load_prompt_template() -> str:
    try:
        path = os.path.normpath(_PROMPT_PATH)
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return (
            "You are a biomedical search strategist.\n"
            "Gene: {gene}\nCell Type: {celltype}\nDisease: {disease}\n"
            "Synonyms: {synonyms}\nCell Subtypes: {cell_subtypes}\n"
            "Return ONLY JSON: {{\"queries\": [...]}}"
        )


def _generate_queries_llm(
    gene:          str,
    celltype:      str,
    disease:       str,
    synonyms:      list[str],
    cell_subtypes: list[str],
) -> list[str]:
    """Ask LLM to generate search queries using broad + subtype terms."""
    template = _load_prompt_template()
    subtypes_str = ", ".join(cell_subtypes) if cell_subtypes else "none (cell type is already specific)"
    prompt = (
        template
        .replace("{gene}",          gene)
        .replace("{celltype}",      celltype)
        .replace("{disease}",       disease or "None (use normal tissue context)")
        .replace("{synonyms}",      ", ".join(synonyms))
        .replace("{cell_subtypes}", subtypes_str)
    )
    text    = call_llm(prompt, max_tokens=700, temperature=0.1, use_fast=False)
    data    = _safe_json_load(text)
    queries = data.get("queries", [])
    if not isinstance(queries, list) or not queries:
        raise ValueError("LLM returned empty or invalid queries list")
    return queries


def _fallback_queries(
    gene:          str,
    celltype:      str,
    disease:       str,
    synonyms:      list[str],
    cell_subtypes: list[str],
) -> list[str]:
    """Deterministic fallback if LLM query generation fails."""
    queries    = []
    base_terms = synonyms[:2]
    cell_terms = [celltype] + cell_subtypes[:2]

    for term in base_terms:
        for ct in cell_terms:
            if disease:
                queries.extend([
                    f'"{term}" "{ct}" immunohistochemistry "{disease}"',
                    f'"{term}" "{ct}" flow cytometry "{disease}"',
                    f'"{term}" "{disease}" protein expression',
                ])
            else:
                queries.extend([
                    f'"{term}" "{ct}" immunohistochemistry',
                    f'"{term}" "{ct}" flow cytometry',
                    f'"{term}" "{ct}" protein expression',
                ])

    return list(dict.fromkeys(queries))


# ── Anchor queries ────────────────────────────────────────────────────────────
#
# Deterministic queries included in EVERY run regardless of LLM output.
# They guarantee a retrieval floor so the system never returns 0 papers
# if PubMed has relevant literature.
#
# Design rules:
#   - Use explicit AND syntax (required by both PubMed and EuropePMC)
#   - Multi-word phrases wrapped in double quotes for exact phrase matching
#   - Cell type ALWAYS included for disease+general cell queries (prevents
#     retrieving irrelevant papers with no cell-type evidence)
#   - 3-4 anchors max (leaves room for LLM queries within the 5-query cap)
#
# Context detection uses lightweight keyword rules — just enough to pick
# the right anchor template. LLM does the full semantic reasoning.

_NORMAL_CELL_MARKERS = {"normal", "adjacent", "luminal", "basal",
                         "non-tumour", "benign", "non-neoplastic"}
_CANCER_CELL_MARKERS = {"cancer", "tumor", "carcinoma", "malignant", "neoplastic"}


def _quote(term: str) -> str:
    """Wrap multi-word terms in double quotes for exact phrase matching."""
    return f'"{term}"' if " " in term else term


def _build_anchor_queries(
    gene:          str,
    celltype:      str,
    disease:       str,
    synonyms:      list[str],
) -> list[str]:
    """
    Build 3-4 deterministic anchor queries that are guaranteed in every run.

    The disease string passed here should already be the canonical/expanded form
    (e.g. "non-small cell lung cancer" not "NSCLC") so EuropePMC matches correctly.
    """
    ct_lower = celltype.lower()

    is_normal_context = (
        any(m in ct_lower for m in _NORMAL_CELL_MARKERS)
        and not any(m in ct_lower for m in _CANCER_CELL_MARKERS)
    )
    is_cancer_context = any(m in ct_lower for m in _CANCER_CELL_MARKERS)

    alt_gene = synonyms[1] if len(synonyms) > 1 else gene
    ct_q     = _quote(ct_lower)

    if disease:
        d   = disease.lower()
        d_q = _quote(d)

        if is_normal_context:
            anchors = [
                f'{gene} AND {d_q} AND "normal tissue" AND immunohistochemistry',
                f'{gene} AND {d_q} AND "adjacent normal" AND IHC',
                f'{gene} AND {d_q} AND "loss of expression" AND immunohistochemistry',
                f'{alt_gene} AND {d_q} AND normal AND protein',
            ]

        elif is_cancer_context:
            anchors = [
                f'{gene} AND {d_q} AND tumor AND immunohistochemistry',
                f'{gene} AND {d_q} AND carcinoma AND IHC AND immunostaining',
                f'{gene} AND {d_q} AND "protein expression" AND immunostaining',
                f'{alt_gene} AND {d_q} AND immunohistochemistry',
            ]

        else:
            # General cell type + disease (NK cells, endothelial, myeloid, etc.)
            # ALWAYS include cell type so we retrieve cell-specific papers.
            anchors = [
                f'{gene} AND {ct_q} AND {d_q} AND immunohistochemistry',
                f'{gene} AND {ct_q} AND {d_q} AND "protein expression"',
                f'{gene} AND {d_q} AND IHC AND immunofluorescence',
                f'{alt_gene} AND {ct_q} AND {d_q} AND "protein expression"',
            ]

    else:
        if is_normal_context:
            anchors = [
                f'{gene} AND "normal tissue" AND immunohistochemistry',
                f'{gene} AND "adjacent normal" AND immunohistochemistry',
                f'{gene} AND normal AND tumor AND immunohistochemistry',
            ]

        elif is_cancer_context:
            anchors = [
                f'{gene} AND tumor AND immunohistochemistry AND "protein expression"',
                f'{gene} AND carcinoma AND IHC AND immunostaining',
                f'{gene} AND cancer AND "protein expression"',
            ]

        else:
            anchors = [
                f'{gene} AND {ct_q} AND immunohistochemistry',
                f'{gene} AND {ct_q} AND immunofluorescence',
                f'{gene} AND "protein expression" AND IHC',
            ]

    return list(dict.fromkeys(anchors))


# ── Public API ────────────────────────────────────────────────────────────────

def planner_agent(
    gene:     str,
    celltype: str,
    disease:  str = "",
) -> dict:
    """
    Agent 1 — Query Planner.

    Runs three LLM expansions in parallel (gene synonyms, cell subtypes,
    disease full name), then builds deterministic anchor queries + LLM queries.

    Args:
        gene:     Gene/protein symbol (e.g. "IFNGR1").
        celltype: Cell type (e.g. "natural killer").
        disease:  Disease context (e.g. "NSCLC") or "" for normal tissue.

    Returns:
        {
            "gene":             str,
            "celltype":         str,
            "disease":          str,   ← canonical/expanded (e.g. "non-small cell lung cancer")
            "disease_original": str,   ← as typed by user (e.g. "NSCLC")
            "synonyms":         [str, ...],
            "cell_subtypes":    [str, ...],
            "disease_synonyms": [str, ...],
            "queries":          [str, ...],
            "anchor_queries":   [str, ...],
            "llm_queries":      [str, ...],
        }
    """
    logger.info(
        f"[planner] Starting | gene={gene} | cell={celltype} | "
        f"disease={disease or 'normal'}"
    )

    # ── Step 1: Three parallel LLM expansions ────────────────────────────────
    # Gene synonyms + cell subtypes + disease expansion all fire concurrently.
    # Total latency = max(3 LLM calls) instead of their sum.
    synonyms:         list[str] = [gene]
    cell_subtypes:    list[str] = []
    disease_canonical: str      = disease
    disease_synonyms:  list[str] = []

    def _expand_gene():
        return expand_gene_synonyms(gene)

    def _expand_cell():
        return expand_cell_subtypes(celltype)

    def _expand_disease():
        return expand_disease(disease)

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="planner_expand") as pool:
        fut_gene    = pool.submit(_expand_gene)
        fut_cell    = pool.submit(_expand_cell)
        fut_disease = pool.submit(_expand_disease)

        try:
            synonyms = fut_gene.result()
        except Exception as exc:
            logger.warning(f"[planner] Gene expansion future failed: {exc}")

        try:
            cell_subtypes = fut_cell.result()
        except Exception as exc:
            logger.warning(f"[planner] Cell expansion future failed: {exc}")

        try:
            disease_canonical, disease_synonyms = fut_disease.result()
        except Exception as exc:
            logger.warning(f"[planner] Disease expansion future failed: {exc}")

    logger.info(
        f"[planner] Expansions done | "
        f"{len(synonyms)} gene synonyms | "
        f"{len(cell_subtypes)} cell subtypes | "
        f"disease='{disease_canonical}' ({len(disease_synonyms)} synonyms)"
    )

    # ── Step 2: Build anchor queries ─────────────────────────────────────────
    # Use expanded disease so EuropePMC gets the full name.
    anchor_queries = _build_anchor_queries(gene, celltype, disease_canonical, synonyms)
    logger.info(f"[planner] Anchor queries: {len(anchor_queries)}")

    # ── Step 3: LLM query generation ─────────────────────────────────────────
    try:
        llm_queries = _generate_queries_llm(
            gene, celltype, disease_canonical, synonyms, cell_subtypes
        )
        logger.info(f"[planner] LLM generated {len(llm_queries)} queries")
    except Exception as exc:
        logger.warning(f"[planner] LLM query generation failed ({exc}) — using fallback")
        llm_queries = _fallback_queries(
            gene, celltype, disease_canonical, synonyms, cell_subtypes
        )
        logger.info(f"[planner] Fallback generated {len(llm_queries)} queries")

    # ── Step 4: Merge and deduplicate (anchors first) ────────────────────────
    seen_queries: set[str] = set()
    queries:      list[str] = []
    for q in anchor_queries + llm_queries:
        q_norm = q.lower().strip()
        if q_norm not in seen_queries:
            seen_queries.add(q_norm)
            queries.append(q)

    # ── Step 5: Quote hyphenated/dotted gene symbols in every query ──────────
    # Gene names like NKX2-8, NKX2.8, IFNG-R1 contain hyphens/dots that search
    # engines tokenise as separate terms without quoting.
    def _quote_term_in_queries(qs: list[str], term: str) -> list[str]:
        quoted = f'"{term}"'
        return [
            q.replace(term, quoted) if term in q and quoted not in q else q
            for q in qs
        ]

    queries = _quote_term_in_queries(queries, gene)
    for syn in synonyms:
        if syn != gene and any(c in syn for c in "-."):
            queries = _quote_term_in_queries(queries, syn)

    logger.info(
        f"[planner] Final: {len(anchor_queries)} anchors + "
        f"{len(llm_queries)} LLM = {len(queries)} unique queries"
    )

    return {
        "gene":             gene,
        "celltype":         celltype,
        "disease":          disease_canonical,
        "disease_original": disease,
        "synonyms":         synonyms,
        "cell_subtypes":    cell_subtypes,
        "disease_synonyms": disease_synonyms,
        "queries":          queries,
        "anchor_queries":   anchor_queries,
        "llm_queries":      llm_queries,
    }
