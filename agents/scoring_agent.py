"""
agents/scoring_agent.py — Agent 5: Evidence Scoring.

Classifies each record into one of three evidence tiers
and assigns a numeric confidence score (0.0–1.0).

Evidence tiers (aligned with project spec):
  Tier 1 (Primary-experimental):     IHC, IF, FACS, smFISH — score 0.8–1.0
  Tier 1 (Primary-not-expressed):    IHC/IF/FACS confirms ABSENCE — score 0.8–1.0
                                      Valid for tumor suppressors silenced in cancer.
  Tier 2 (Secondary-review):         Review papers, clinical studies — score 0.5–0.7
  Tier 3 (Tertiary-database):        HPA, CellMarker, UniProt — score 0.2–0.4

Score penalties:
  - RNA-only evidence        : -0.3
  - Low confidence from LLM  : -0.1
  - "no evidence" label      : score = 0.0
"""

from __future__ import annotations

import re
from loguru import logger

# Tier keyword sets
_PRIMARY_ASSAY_KW = {
    "ihc", "immunohistochemistry",
    "if", "ifs", "immunofluorescence",
    "flow cytometry", "facs",
    "smfish", "sm-fish", "single-molecule fish",
    "western blot", "elisa", "co-localiz",
    # TMA is a histological format for IHC — treat as protein-level evidence
    "tissue microarray", "tma", "immunostaining", "immunostained",
    "immunolabeling", "immunolabelling", "antibody stain",
}

_REVIEW_KW = {
    "review", "meta-analysis", "systematic review",
    "literature review", "survey", "clinical study",
    "cohort", "patient", "clinical trial",
}

_DATABASE_SOURCES = {
    "humanproteinatlas", "cellmarker", "uniprot", "hpa",
    "proteinatlas", "gepia",
}

_RNA_KW = {
    "rna-seq", "rnaseq", "transcript", "mrna", "qpcr",
    "scrna", "bulk rna",
    # ── IMPORTANT: bare "microarray" removed ──────────────────────────────────
    # "tissue microarray" (TMA) is a protein assay platform (IHC on cores).
    # Using "microarray" as a substring caused TMA papers to score as RNA-only.
    # Use specific terms for RNA-level microarrays instead:
    "gene expression microarray", "cdna microarray", "mrna microarray",
    "affymetrix", "expression array", "gene chip",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s\-]", " ", str(text or "").lower().strip())


def scoring_agent(validated: list[dict]) -> list[dict]:
    """
    Agent 5 — Evidence Strength Classification.

    Adds to each record:
        Evidence       : str (tier label)
        evidence_score : float (0.0–1.0)

    Args:
        validated: Records from validation_agent.

    Returns:
        Same list with Evidence + evidence_score added.
    """
    for record in validated:
        assay      = _normalize(record.get("assay", ""))
        source     = _normalize(record.get("source", ""))
        ev_level   = _normalize(record.get("evidence_level", ""))
        confidence = _normalize(record.get("confidence", ""))
        key_sent   = _normalize(record.get("key_sentence", ""))
        full_text  = " ".join([assay, source, ev_level, key_sent])

        # ── No evidence → score 0 ─────────────────────────────────────────
        if "no evidence" in ev_level or ev_level == "no evidence":
            record["Evidence"]       = "No evidence"
            record["evidence_score"] = 0.0
            continue

        # ── Tertiary: database sources always classified here ─────────────
        source_clean = source.replace(" ", "").replace("-", "")
        if any(db in source_clean for db in _DATABASE_SOURCES):
            record["Evidence"]       = "Tertiary-database"
            record["evidence_score"] = 0.3
            continue

        # ── Fix 2: NOT_EXPRESSED via protein assay → Tier 1 score ────────
        # When validation_agent assigned NOT_EXPRESSED (IHC/IF/FACS confirmed
        # absence), this is as strong as positive Tier 1 evidence.
        # Tumor suppressor silencing is a biologically meaningful finding.
        validation_status = str(record.get("ManualReviewStatus", "")).upper()
        expr_status_raw   = str(record.get("expression_status", "")).lower().strip()
        if validation_status == "NOT_EXPRESSED" or expr_status_raw in ("not expressed", "silenced"):
            # Check protein assay is present (belt-and-braces)
            if any(kw in full_text for kw in _PRIMARY_ASSAY_KW):
                base_score = 0.9  # slightly below perfect 1.0 (absence vs confirmed presence)
                if confidence == "low":
                    base_score -= 0.15
                elif confidence == "medium":
                    base_score -= 0.05
                record["Evidence"]       = "Primary-not-expressed"
                record["evidence_score"] = round(max(base_score, 0.5), 2)
                continue

        # ── Primary: protein-level experimental ───────────────────────────
        if any(kw in full_text for kw in _PRIMARY_ASSAY_KW):
            base_score = 1.0
            # Penalty for RNA mention alongside protein
            if any(kw in full_text for kw in _RNA_KW):
                base_score -= 0.15
            # Confidence penalty
            if confidence == "low":
                base_score -= 0.15
            elif confidence == "medium":
                base_score -= 0.05
            record["Evidence"]       = "Primary-experimental"
            record["evidence_score"] = round(max(base_score, 0.5), 2)
            continue

        # ── RNA-only ──────────────────────────────────────────────────────
        # MUST come BEFORE Secondary-review:
        # RNA papers can mention "patient" / "cohort" in key sentences,
        # which would falsely trigger Secondary-review (0.6) before reaching here.
        # ManualReviewStatus == "FAIL" is the authoritative signal set by
        # validation_agent — always score FAIL records as RNA-only (0.2).
        if (
            validation_status == "FAIL"
            or any(kw in full_text for kw in _RNA_KW)
        ):
            record["Evidence"]       = "RNA-only"
            record["evidence_score"] = 0.2
            continue

        # ── Secondary: review / clinical / indirect ───────────────────────
        if any(kw in full_text for kw in _REVIEW_KW) or ev_level == "indirect":
            base_score = 0.6
            if confidence == "low":
                base_score -= 0.1
            record["Evidence"]       = "Secondary-review"
            record["evidence_score"] = round(max(base_score, 0.3), 2)
            continue

        # ── Fallback ──────────────────────────────────────────────────────
        record["Evidence"]       = "Unknown"
        record["evidence_score"] = 0.1

    # Sort by score descending
    validated.sort(key=lambda r: r.get("evidence_score", 0), reverse=True)

    tier_summary = {}
    for r in validated:
        ev = r.get("Evidence", "?")
        tier_summary[ev] = tier_summary.get(ev, 0) + 1
    logger.info(f"[scoring] Evidence tiers: {tier_summary}")

    return validated
