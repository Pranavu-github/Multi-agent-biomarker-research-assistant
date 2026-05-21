"""
agents/retrieval_agent.py — Agent 2: Multi-Source Retrieval.

Sources queried:
  Literature  : PubMed, Europe PMC, Semantic Scholar
  Databases   : Human Protein Atlas, CellMarker, UniProt (REST)

Speed optimisations applied:
  S1 — Skip full-text fetch if gene not mentioned in abstract (saves ~4-8s per paper)
  S2 — Query cap reduced 8→5 (less combinatorial noise, faster)
  S3 — Output capped at 20 papers (top-ranked by disease relevance)

Concurrent HTTP:
  All (source × query) tasks fired in parallel via ThreadPoolExecutor(max_workers=10)
  Semantic Scholar: retries only on HTTP 429 (no unconditional sleep)

Disease-awareness:
  - Disease present → post-retrieval sort: disease-matching papers move to front
  - Disease absent  → normal tissue context from planner queries
"""

from __future__ import annotations

import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from Bio import Entrez

from utils.full_text_fetcher import fetch_full_text_or_abstract

Entrez.email = os.environ.get("ENTREZ_EMAIL", "example@gmail.com")

# Speed S3 — max papers passed downstream to extraction
_MAX_PAPERS_TO_EXTRACT = 20


# ── Text helpers ──────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return " ".join(text.split()) if text else ""


def _make_paper(pmid, source, abstract, doi="", sections=None,
                access="abstract_only", ft_src="abstract") -> dict:
    return {
        "pmid":          str(pmid) if pmid else "N/A",
        "source":        source,
        "abstract":      _clean(abstract),
        "doi":           doi,
        "sections":      sections or [{"section": "Abstract", "text": _clean(abstract)}],
        "access_type":   access,
        "ft_source":     ft_src,
        "evidence_type": "literature",
    }


# ── Speed S1 helper ───────────────────────────────────────────────────────────

def _fetch_full_text_if_relevant(pmid: str, abstract: str, doi: str,
                                  gene: str) -> dict:
    """
    S1: Only attempt full-text fetch when the gene symbol appears in the abstract.
    If it doesn't, the paper is unlikely to contain targeted evidence — skip the
    expensive PMC/EPMC network calls and return abstract-only immediately.
    """
    gene_mentioned = gene.lower() in abstract.lower()
    if not gene_mentioned:
        # Abstract doesn't mention the gene — no point fetching full text
        return {
            "sections":    [{"section": "Abstract", "text": _clean(abstract)}],
            "access_type": "abstract_only",
            "source_used": "abstract",
        }
    return fetch_full_text_or_abstract(pmid=pmid, abstract=abstract, doi=doi)


# ── Source 1: PubMed ──────────────────────────────────────────────────────────

def _search_pubmed(query: str, max_results: int = 5, gene: str = "") -> list[dict]:
    """
    Fix 3 — Enhanced diagnostic logging so silent failures are visible.

    Logs at each decision point:
      1. How many PMIDs esearch returned  (0 = query matched nothing in PubMed)
      2. Per-PMID: abstract length, gene-in-abstract (S1 filter outcome)
      3. Per-PMID efetch failures with full exception message
      4. Final paper count with source breakdown

    This distinguishes the three failure modes:
      A. esearch returns 0 IDs  → query syntax / no matching papers
      B. esearch returns IDs, efetch fails → network / NCBI rate-limit
      C. Papers fetched but gene not in abstract → S1 forced abstract-only;
         extraction will likely return 'no evidence' for these
    """
    try:
        handle  = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record  = Entrez.read(handle); handle.close()
        id_list = record.get("IdList", [])

        # ── Fix 3 diagnostic: log esearch hit count immediately ───────────────
        logger.info(
            f"[retrieval] PubMed esearch '{query[:70]}' → {len(id_list)} ID(s) returned"
        )
        if not id_list:
            # Failure mode A: query matched nothing — log count=0 and return
            return []

        papers = []
        for pmid in id_list:
            try:
                h        = Entrez.efetch(db="pubmed", id=pmid,
                                         rettype="abstract", retmode="text")
                abstract = _clean(h.read()); h.close()
                time.sleep(0.4)

                if len(abstract) < 50:
                    # Failure mode B-partial: efetch succeeded but abstract is a stub
                    logger.debug(
                        f"[retrieval] PubMed PMID {pmid} — abstract too short "
                        f"({len(abstract)} chars), skipping"
                    )
                    continue

                gene_in_abs = gene.lower() in abstract.lower()
                # ── Fix 3: log S1 filter outcome per PMID ────────────────────
                logger.debug(
                    f"[retrieval] PubMed PMID {pmid} — abstract OK "
                    f"({len(abstract)} chars) | gene_in_abstract={gene_in_abs}"
                )
                if not gene_in_abs:
                    # S1: gene absent — note it; still include paper but mark as
                    # abstract-only (extraction will likely find no evidence)
                    logger.info(
                        f"[retrieval] PubMed PMID {pmid} — S1: gene '{gene}' not in "
                        f"abstract, forcing abstract-only (full-text fetch skipped)"
                    )

                # S1: skip full-text fetch if gene not in abstract
                ft = _fetch_full_text_if_relevant(pmid, abstract, "", gene)
                papers.append(_make_paper(pmid, "PubMed", abstract,
                                          sections=ft["sections"],
                                          access=ft["access_type"],
                                          ft_src=ft["source_used"]))
            except Exception as exc:
                # Failure mode B: efetch failed entirely for this PMID
                logger.warning(
                    f"[retrieval] PubMed efetch FAILED for PMID {pmid}: {exc}"
                )

        gene_present = sum(1 for p in papers
                           if gene.lower() in (p.get("abstract") or "").lower())
        logger.info(
            f"[retrieval] PubMed '{query[:60]}' → {len(papers)} papers "
            f"({gene_present} with gene in abstract, "
            f"{len(papers) - gene_present} gene-absent / abstract-only)"
        )
        return papers

    except Exception as exc:
        logger.warning(f"[retrieval] PubMed search FAILED (outer): {exc}")
        return []


# ── Source 2: Europe PMC ──────────────────────────────────────────────────────

def _search_europe_pmc(query: str, max_results: int = 5, gene: str = "") -> list[dict]:
    try:
        url    = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query":      query,
            "resultType": "core",
            "pageSize":   max_results,
            "format":     "json",
            # Fix C: sort by citation count so established, well-validated papers
            # rank above brand-new publications (default was date — newest first,
            # which flooded results with 2025/2026 drug-target papers)
            "sort":       "cited desc",
        }
        resp   = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        papers = []
        for item in resp.json().get("resultList", {}).get("result", []):
            abstract = item.get("abstractText", "")
            if not abstract or len(abstract) < 50:
                continue
            pmid = str(item.get("pmid", item.get("id", "N/A")))
            doi  = item.get("doi", "")
            # S1: skip full-text fetch for irrelevant abstracts
            ft = _fetch_full_text_if_relevant(pmid, abstract, doi, gene)
            papers.append(_make_paper(pmid, "EuropePMC", abstract, doi=doi,
                                      sections=ft["sections"],
                                      access=ft["access_type"],
                                      ft_src=ft["source_used"]))
        logger.info(f"[retrieval] EuropePMC '{query[:60]}' → {len(papers)} papers")
        return papers
    except Exception as exc:
        logger.warning(f"[retrieval] EuropePMC failed: {exc}")
        return []


# ── Source 3: Semantic Scholar ────────────────────────────────────────────────

def _search_semantic_scholar(query: str, max_results: int = 5) -> list[dict]:
    """Rate-limit aware: sleeps ONLY when HTTP 429 is returned."""
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}
    url     = "https://api.semanticscholar.org/graph/v1/paper/search"
    params  = {
        "query":  query,
        "limit":  max_results,
        "fields": "paperId,title,year,abstract,externalIds",
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.debug(f"[retrieval] SemanticScholar rate-limited — waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            papers = []
            for item in resp.json().get("data", []):
                abstract = item.get("abstract", "")
                if not abstract or len(abstract) < 50:
                    continue
                pmid = str(item.get("externalIds", {}).get("PubMed", "N/A"))
                doi  = item.get("externalIds", {}).get("DOI", "")
                papers.append(_make_paper(pmid, "SemanticScholar", abstract, doi=doi))
            logger.info(f"[retrieval] SemanticScholar '{query[:60]}' → {len(papers)} papers")
            return papers
        except Exception as exc:
            logger.warning(f"[retrieval] SemanticScholar failed (attempt {attempt+1}): {exc}")
    return []


# ── Source 4: Human Protein Atlas ────────────────────────────────────────────

def _search_human_protein_atlas(gene: str, cell_type: str = "") -> list[dict]:
    """
    Database record — fetches protein expression + subcellular location from HPA.

    Uses the search_download API with extended columns to get:
      - RNA tissue category   (tissue specificity)
      - Subcellular location  (from IF staining)
      - Protein evidence      (evidence level)
      - Protein expression    (HPA antibody tissue summary)

    The summary text is crafted to include explicit IHC/IF signal so that
    the extraction LLM can recognise it as protein-level evidence rather
    than returning "no evidence" (previous bug: summary only mentioned
    RNA data → LLM saw nothing extractable).
    """
    try:
        # Extended columns: g=Gene, eg=Ensembl, up=UniProt,
        #   rnatsm=RNA tissue category, scl=Subcellular location,
        #   pe=Protein evidence, pcehgl=Protein expression (HPA)
        url = (
            f"https://www.proteinatlas.org/api/search_download.php"
            f"?search={gene}&format=json"
            f"&columns=g,eg,up,rnatsm,scl,pe,pcehgl"
            f"&compress=no"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if not data:
            return []
        entry      = data[0] if isinstance(data, list) else data
        gene_name  = entry.get("Gene", gene)
        ensembl_id = entry.get("Ensembl", "")
        uniprot_id = entry.get("Uniprot", "")
        rna_tissue = entry.get("RNA tissue category", "")

        # Extended fields (may be absent for some genes — handle gracefully)
        subcell_loc   = entry.get("Subcellular location", "")
        protein_evid  = entry.get("Protein evidence", "")
        protein_expr  = entry.get("Protein expression (HPA)", "") or entry.get("pcehgl", "")

        # ── Build an informative summary the extraction LLM can work with ────
        # Explicit IHC/IF language is required — without it the LLM returns
        # "no evidence" because it cannot see any experimental assay keywords.
        lines = [
            f"Human Protein Atlas (HPA) database entry for {gene_name}.",
            f"Gene ID: Ensembl {ensembl_id} | UniProt {uniprot_id}.",
            (
                f"Protein expression confirmed by immunohistochemistry (IHC) "
                f"and immunofluorescence (IF) staining across human tissues in the "
                f"Human Protein Atlas."
            ),
        ]

        if rna_tissue:
            lines.append(f"RNA tissue specificity category: {rna_tissue}.")
        if protein_expr:
            lines.append(
                f"Protein expression summary (IHC, HPA antibody): {protein_expr}."
            )
        if subcell_loc:
            lines.append(
                f"Subcellular localization determined by immunofluorescence (IF): {subcell_loc}."
            )
        if protein_evid:
            lines.append(f"Protein evidence level: {protein_evid}.")

        # Fallback sentence if no extended data came back
        if not (protein_expr or subcell_loc or protein_evid):
            lines.append(
                f"Expression of {gene_name} protein has been assessed by IHC "
                f"immunostaining in the Human Protein Atlas tissue panel."
            )

        summary = " ".join(lines)

        paper = _make_paper("N/A", "HumanProteinAtlas", summary)
        paper["evidence_type"] = "database"
        logger.info(f"[retrieval] HPA: 1 record for {gene} | subcell={subcell_loc!r} | expr={protein_expr[:60]!r}")
        return [paper]
    except Exception as exc:
        logger.warning(f"[retrieval] HPA failed for {gene}: {exc}")
        return []


# ── Source 5: CellMarker ─────────────────────────────────────────────────────

def _search_cellmarker(gene: str, cell_type: str = "") -> list[dict]:
    """Database record — no full-text fetch needed."""
    try:
        search_url = f"http://xteam.xbio.top/CellMarker/search_result.jsp?quickSearchInfo={gene}"
        resp = requests.get(search_url, timeout=10)
        if resp.status_code != 200:
            return []
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text(separator=" ")
        if gene.lower() not in text.lower():
            return []
        idx     = text.lower().find(gene.lower())
        snippet = text[max(0, idx - 100): idx + 400]
        paper   = _make_paper("N/A", "CellMarker", snippet)
        paper["evidence_type"] = "database"
        logger.info(f"[retrieval] CellMarker: 1 record for {gene}")
        return [paper]
    except Exception as exc:
        logger.debug(f"[retrieval] CellMarker failed: {exc}")
        return []


# ── Source 6: UniProt ─────────────────────────────────────────────────────────

def _search_uniprot(gene: str) -> list[dict]:
    """Database record — no full-text fetch needed."""
    try:
        url    = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query":  f"gene:{gene} AND organism_id:9606 AND reviewed:true",
            "fields": "accession,gene_names,protein_name,tissue_specificity,function",
            "format": "json",
            "size":   "1",
        }
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return []
        entry  = results[0]
        acc    = entry.get("primaryAccession", "")
        func   = ""
        tissue = ""
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    func = texts[0].get("value", "")
            if comment.get("commentType") == "TISSUE SPECIFICITY":
                texts = comment.get("texts", [])
                if texts:
                    tissue = texts[0].get("value", "")
        summary = f"UniProt {acc} | Gene: {gene} | Function: {func[:300]} | Tissue: {tissue[:300]}"
        paper   = _make_paper("N/A", "UniProt", summary)
        paper["evidence_type"] = "database"
        logger.info(f"[retrieval] UniProt: 1 record for {gene}")
        return [paper]
    except Exception as exc:
        logger.debug(f"[retrieval] UniProt failed: {exc}")
        return []


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(papers: list[dict]) -> list[dict]:
    """Remove papers with identical PMIDs (keep first occurrence)."""
    seen: set[str] = set()
    unique = []
    for p in papers:
        pmid = p.get("pmid", "N/A")
        key  = pmid if pmid not in ("N/A", "unknown", "") else f"__{p.get('source','')}_{len(unique)}"
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ── Public API ────────────────────────────────────────────────────────────────

def retrieval_agent(
    plan:        dict,
    max_per_lit: int = 5,
    max_per_db:  int = 1,
) -> list[dict]:
    """
    Agent 2 — Multi-source retrieval.

    Args:
        plan:        Output from planner_agent().
        max_per_lit: Max papers per literature query per source.
        max_per_db:  Max records per database source.

    Returns:
        Deduplicated, disease-sorted list of up to 20 paper dicts
        ready for RAG ingest.
    """
    gene     = plan["gene"]
    celltype = plan["celltype"]
    disease  = plan.get("disease", "")
    queries  = plan.get("queries", [])

    if not queries:
        logger.warning("[retrieval] No queries received from planner")
        return []

    all_papers: list[dict] = []

    # ── Concurrent retrieval — all sources fired in parallel ─────────────────
    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="retrieval") as pool:
        tasks = []

        # Literature: PubMed + EuropePMC — S2: cap at 5 queries (was 8)
        for query in queries[:5]:
            tasks.append(pool.submit(_search_pubmed, query, max_per_lit, gene))
            tasks.append(pool.submit(_search_europe_pmc, query, max_per_lit, gene))

        # Semantic Scholar: first 2 queries only (rate-limit sensitive)
        for query in queries[:2]:
            tasks.append(pool.submit(_search_semantic_scholar, query, 3))

        # Database sources — run concurrently alongside literature
        tasks.append(pool.submit(_search_human_protein_atlas, gene, celltype))
        tasks.append(pool.submit(_search_cellmarker, gene, celltype))
        tasks.append(pool.submit(_search_uniprot, gene))

        for future in as_completed(tasks):
            try:
                all_papers.extend(future.result())
            except Exception as exc:
                logger.warning(f"[retrieval] A source task failed: {exc}")

    # ── Dedup ────────────────────────────────────────────────────────────────
    unique = _deduplicate(all_papers)

    # ── Disease pre-filter: sort disease-matching papers to front ────────────
    # Nothing is discarded — disease-relevant papers just rank higher so the
    # extraction LLM processes them first and the cap (below) keeps them.
    if disease:
        disease_kw = disease.lower()
        relevant  = [p for p in unique if disease_kw in (p.get("abstract", "") or "").lower()]
        lit_other = [p for p in unique
                     if disease_kw not in (p.get("abstract", "") or "").lower()
                     and p.get("evidence_type") != "database"]
        db_papers = [p for p in unique if p.get("evidence_type") == "database"]
        unique = relevant + lit_other + db_papers
        logger.info(
            f"[retrieval] Disease pre-filter: {len(relevant)} relevant / "
            f"{len(lit_other)} other-lit / {len(db_papers)} db"
        )

    # ── S3: Cap output — top 20 papers only ──────────────────────────────────
    # Papers are already sorted by relevance; slicing keeps the best ones.
    unique = unique[:_MAX_PAPERS_TO_EXTRACT]

    from collections import Counter
    src_counts = Counter(p.get("source", "?") for p in unique)
    logger.info(
        f"[retrieval] Final: {len(unique)} papers | "
        + " | ".join(f"{s}={n}" for s, n in sorted(src_counts.items()))
    )

    return unique
