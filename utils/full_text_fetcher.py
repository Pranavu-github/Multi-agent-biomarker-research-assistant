"""
utils/full_text_fetcher.py — Full-text fetcher with abstract fallback.

Access priority per paper:
  1. PubMed Central (PMC) — free full XML via Entrez
  2. Europe PMC         — open-access full text REST
  3. Unpaywall          — legal open-access HTML link
  4. bioRxiv / medRxiv  — preprint full text
  5. FALLBACK           — abstract only

Copyright fully respected. No paywall bypassing.
"""

from __future__ import annotations

import os
import re
import time
import requests
from loguru import logger
from Bio import Entrez

# ── Entrez config (reads from .env) ──────────────────────────────────────────
Entrez.email = os.environ.get("ENTREZ_EMAIL", "example@gmail.com")

# ── Section detection patterns ────────────────────────────────────────────────
_SECTION_PATTERNS = {
    "Abstract":     r"\b(abstract)\b",
    "Introduction": r"\b(introduction|background)\b",
    "Methods":      r"\b(methods?|materials?\s+and\s+methods?|methodology|"
                    r"experimental\s+procedures?|patients?\s+and\s+methods?)\b",
    "Results":      r"\b(results?|findings?|observations?)\b",
    "Discussion":   r"\b(discussion|interpretation)\b",
    "Conclusion":   r"\b(conclusion|summary|concluding\s+remarks?)\b",
    "Supplementary":r"\b(supplementary|supplemental|supporting\s+information)\b",
}


# ── Text utilities ────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return " ".join(text.split())


def _detect_section(block: str) -> str:
    first = block[:120].lower()
    for label, pattern in _SECTION_PATTERNS.items():
        if re.search(pattern, first, re.IGNORECASE):
            return label
    return "Body"


def _split_sections(full_text: str) -> list[dict]:
    """Split full text into labelled section dicts."""
    blocks = re.split(r"\n(?=[A-Z][A-Z\s]{2,40}\n)", full_text)
    if len(blocks) < 2:
        return [{"section": "Body", "text": _clean(full_text)}]
    sections = []
    for block in blocks:
        if len(block.strip()) < 50:
            continue
        sections.append({"section": _detect_section(block), "text": _clean(block)})
    return sections or [{"section": "Body", "text": _clean(full_text)}]


def _abstract_only(abstract: str) -> dict:
    return {
        "sections":    [{"section": "Abstract", "text": _clean(abstract)}],
        "access_type": "abstract_only",
        "source_used": "abstract",
    }


# ── Source 1: PubMed Central ─────────────────────────────────────────────────

def _fetch_pmc(pmid: str) -> list[dict] | None:
    """Return sections from PMC full-text XML, or None if unavailable."""
    try:
        link_handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        link_record = Entrez.read(link_handle)
        link_handle.close()

        pmc_ids = [
            link["Id"]
            for linkset in link_record
            for linksetdb in linkset.get("LinkSetDb", [])
            if linksetdb.get("LinkName") == "pubmed_pmc"
            for link in linksetdb.get("Link", [])
        ]

        if not pmc_ids:
            return None

        time.sleep(0.4)
        ft_handle = Entrez.efetch(
            db="pmc", id=pmc_ids[0], rettype="full", retmode="xml"
        )
        xml_text = ft_handle.read()
        ft_handle.close()

        # Extract text from XML (simple approach — strip tags)
        clean_text = re.sub(r"<[^>]+>", " ", xml_text if isinstance(xml_text, str) else xml_text.decode("utf-8", errors="ignore"))
        clean_text = _clean(clean_text)

        if len(clean_text) < 200:
            return None

        return _split_sections(clean_text)

    except Exception as exc:
        logger.debug(f"[full_text_fetcher] PMC fetch failed for PMID {pmid}: {exc}")
        return None


# ── Source 2: Europe PMC full text ────────────────────────────────────────────

def _fetch_epmc(pmid: str) -> list[dict] | None:
    """Try Europe PMC open-access full text."""
    try:
        url  = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmid}/fullTextXML"
        resp = requests.get(url, timeout=6)  # reduced: 12s → 6s
        if resp.status_code != 200:
            return None
        xml_text  = resp.text
        clean_text = _clean(re.sub(r"<[^>]+>", " ", xml_text))
        if len(clean_text) < 200:
            return None
        return _split_sections(clean_text)
    except Exception as exc:
        logger.debug(f"[full_text_fetcher] EPMC full-text failed for {pmid}: {exc}")
        return None


# ── Source 3: Unpaywall ───────────────────────────────────────────────────────

def _fetch_unpaywall(doi: str) -> list[dict] | None:
    """Try Unpaywall to get a legal open-access HTML URL, then scrape."""
    if not doi:
        return None
    email = os.environ.get("UNPAYWALL_EMAIL", "example@gmail.com")
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}?email={email}", timeout=10
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        oa_url = data.get("best_oa_location", {}) or {}
        url    = oa_url.get("url_for_landing_page") or oa_url.get("url")
        if not url:
            return None

        time.sleep(0.5)
        page = requests.get(url, timeout=8)  # reduced: 15s → 8s
        if page.status_code != 200:
            return None

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(page.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = _clean(soup.get_text(separator=" "))
        if len(text) < 300:
            return None
        return _split_sections(text)

    except Exception as exc:
        logger.debug(f"[full_text_fetcher] Unpaywall failed for DOI {doi}: {exc}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_full_text_or_abstract(
    pmid:     str = "",
    abstract: str = "",
    doi:      str = "",
) -> dict:
    """
    Attempt full-text fetch (PMC → EPMC → Unpaywall) for a paper.
    Falls back to abstract-only.

    Returns:
        {
            "sections":    [{"section": str, "text": str}, ...],
            "access_type": "full_text" | "abstract_only",
            "source_used": str,
        }
    """
    # 1. PMC
    if pmid and pmid not in ("N/A", "unknown"):
        sections = _fetch_pmc(pmid)
        if sections:
            return {"sections": sections, "access_type": "full_text", "source_used": "PMC"}

    # 2. EPMC
    if pmid and pmid not in ("N/A", "unknown"):
        sections = _fetch_epmc(pmid)
        if sections:
            return {"sections": sections, "access_type": "full_text", "source_used": "EuropePMC"}

    # 3. Unpaywall
    if doi:
        sections = _fetch_unpaywall(doi)
        if sections:
            return {"sections": sections, "access_type": "full_text", "source_used": "Unpaywall"}

    # Fallback: abstract only
    return _abstract_only(abstract or "No text available.")
