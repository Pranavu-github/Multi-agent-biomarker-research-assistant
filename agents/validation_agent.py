"""
agents/validation_agent.py — Agent 4: Biological Validation.

Classifies each extracted record by ManualReviewStatus:
  PASS         → protein-level assay confirms EXPRESSION (IHC, IF, FACS, smFISH)
  NOT_EXPRESSED → protein-level assay confirms ABSENCE/LOSS (IHC, IF, FACS, Flow)
                  — Tier 1 evidence for tumor suppressor silencing
  FAIL         → RNA-only evidence (RNA-seq, transcriptomics)
  NA           → unclear, ambiguous, no evidence, OR negated/unsupported assay

Fixes applied:
  Fix A — Word boundaries on negation words (\bno\b, \bnot\b, etc.) so that
           "normal", "notable", "noted" etc. do NOT falsely trigger negation.
           Root cause: previously "no" matched inside "normal" → PMID 41800236
           was incorrectly labeled NA despite clear IF evidence.

  Fix B — evidence_level="no evidence" check runs FIRST, before assay keywords.
           Prevents a record with assay="IHC" but evidence_level="no evidence"
           from getting a false PASS (was happening on PMID 41969468).

  Fix 2 — NOT_EXPRESSED status added for protein-level assays (IHC/IF/FACS/Flow)
           that confirm ABSENCE of the gene. This is valid Tier 1 evidence —
           critical for tumor suppressor genes silenced in cancer cells
           (e.g. SLC5A8, MLH1, BRCA1 loss-of-expression in tumor tissue).
           RT-PCR / methylation-PCR results are NOT eligible for NOT_EXPRESSED
           (those remain FAIL or NA — RNA/epigenetic level only).
"""

from __future__ import annotations

import re
from loguru import logger

# ── Assay keyword sets ────────────────────────────────────────────────────────

_PROTEIN_ASSAYS = {
    "ihc", "immunohistochemistry",
    "if", "ifs", "immunofluorescence",
    "flow cytometry", "facs",
    "smfish", "sm-fish", "single-molecule fish",
    "protein expression", "western blot", "elisa",
    "co-localiz", "colocaliz",
    # TMA is a histological FORMAT — the assay on a TMA is always IHC/immunostaining
    # Must be here so "tissue microarray" is not wrongly classified as RNA-only
    "tissue microarray", "tma", "immunostaining", "immunostained",
    "protein localiz", "antibody stain", "immunolabeling", "immunolabelling",
}

_RNA_ASSAYS = {
    "rna-seq", "rnaseq", "rna seq", "transcriptomics",
    "bulk rna", "scrna", "single-cell rna",
    "mrna", "transcript", "qpcr", "rt-pcr",
    # ── IMPORTANT: "microarray" alone is NOT listed here ──────────────────────
    # Reason: "tissue microarray" (TMA) contains the substring "microarray" but
    # is a PROTEIN assay platform (IHC on tissue cores), NOT an RNA assay.
    # Using bare "microarray" caused tissue microarray papers to be wrongly
    # classified as RNA-only (FAIL). Use specific RNA-array terms instead:
    "gene expression microarray", "cdna microarray", "mrna microarray",
    "affymetrix", "expression array", "gene chip",
    "in situ hybridiz",
}

# ── Negation detection (Fix A — word boundaries added) ───────────────────────
#
# WHY word boundaries matter:
#   Without \b: "no" matches the first 2 chars of "normal"
#               → "normal fibroblast ... immunofluorescence" triggers false negation
#   With \b:    "no" only matches the standalone word "no"
#               → "normal" is skipped correctly

# Pattern A: negation word BEFORE the assay keyword
# Catches: "not confirmed by IHC", "no IHC staining", "absent immunofluorescence"
_NEGATION_BEFORE = re.compile(
    r"(\bnot\b|\bno\b|\bnever\b|\babsent\b|\bundetected\b|\bnegative\b|"
    r"\bnot\s+confirmed\b|\bnot\s+expressed\b|\bnot\s+detected\b|"
    r"\bnot\s+found\b|\black\s+of\b|\bfailed\s+to\b|\bcould\s+not\b)"
    r".{0,50}"   # up to 50 chars between negation and assay keyword
    r"(ihc|immunohistochemistry|immunofluorescence|\bifs?\b|facs|"
    r"flow\s*cytometry|smfish|western\s*blot)",
    re.IGNORECASE | re.DOTALL,
)

# Pattern B: assay keyword BEFORE the negation word
# Catches: "IHC has not been confirmed", "FACS staining was negative"
_NEGATION_AFTER = re.compile(
    r"(ihc|immunohistochemistry|immunofluorescence|\bifs?\b|facs|"
    r"flow\s*cytometry|smfish|western\s*blot)"
    r".{0,50}"
    r"(\bnot\b|\bno\b|\bnever\b|\babsent\b|\bundetected\b|\bnegative\b|"
    r"\bnot\s+confirmed\b|\bnot\s+expressed\b|\bnot\s+detected\b|"
    r"\bnot\s+found\b)",
    re.IGNORECASE | re.DOTALL,
)


def _is_negated(text: str) -> bool:
    """Return True if an assay keyword appears in a negated context."""
    return bool(_NEGATION_BEFORE.search(text) or _NEGATION_AFTER.search(text))


def _normalize(text: str) -> str:
    """Lowercase + remove non-alphanumeric (except space/hyphen)."""
    text = str(text or "").lower().strip()
    return re.sub(r"[^a-z0-9\s\-]", " ", text)


# ── Absence / silencing keywords (Fix 2) ─────────────────────────────────────
#
# These phrase patterns (combined with a protein assay keyword) indicate
# that the gene is ABSENT or LOST rather than just mentioned in a negative sentence.
# Distinct from _is_negated() which detects "not confirmed" / "staining was negative".

_ABSENCE_KW = re.compile(
    r"(\bloss\s+of\s+expression\b|\bexpression\s+loss\b|\bsilenced\b|\bsilencing\b|"
    r"\bepigenetic\s+silenc\b|\bdownregulated?\b|\bdownregulation\b|\bsuppressed?\b|"
    r"\bnot\s+expressed\b|\bno\s+expression\b|\babsent\b|\babsence\s+of\b|"
    r"\blost\s+expression\b|\black\s+of\s+expression\b|\bexpression\s+was\s+lost\b|"
    r"\bexpression\s+is\s+lost\b|\bnegative\s+for\b|\bundetectable\b|\bundetected\b)",
    re.IGNORECASE,
)

# Only Tier-1 protein assays count for NOT_EXPRESSED (not RT-PCR, methylation-PCR, etc.)
# Tissue microarray (TMA) is a histological FORMAT for IHC — included as Tier-1
# Immunostaining is a synonym for IHC — included as Tier-1
_TIER1_PROTEIN_ASSAYS = {
    "ihc", "immunohistochemistry",
    "if", "ifs", "immunofluorescence",
    "flow cytometry", "facs",
    "smfish", "sm-fish", "single-molecule fish",
    "tissue microarray", "tma",
    "immunostaining", "immunostained",
    "immunolabeling", "immunolabelling",
}

# Supplementary-only assays: NOT eligible for NOT_EXPRESSED Tier 1
_SUPPLEMENTARY_ONLY_KW = {
    "rt-pcr", "methylation-pcr", "bisulfite", "qpcr", "mrna", "rna-seq",
    "rnaseq", "transcript", "microarray",
}


def _is_tier1_absence(assay: str, full_text: str) -> bool:
    """
    Return True when:
      1. A Tier-1 protein assay (IHC/IF/FACS/Flow) is present in assay/full_text
      2. An absence/silencing keyword is present
      3. The absence is NOT driven solely by supplementary assays (RT-PCR, methylation-PCR)
    """
    has_tier1    = any(kw in assay or kw in full_text for kw in _TIER1_PROTEIN_ASSAYS)
    has_absence  = bool(_ABSENCE_KW.search(full_text))
    # Check if absence evidence comes from protein assay or only from RNA/methylation
    supp_only    = (
        not has_tier1
        and any(kw in full_text for kw in _SUPPLEMENTARY_ONLY_KW)
    )
    return has_tier1 and has_absence and not supp_only


def validation_agent(
    extracted:     list[dict],
) -> list[dict]:
    """
    Agent 4 — Biological Validation.

    Adds 'ManualReviewStatus' to each record: PASS / NOT_EXPRESSED / FAIL / NA

    Args:
        extracted: List of records from extraction_agent.

    Returns:
        Same list with ManualReviewStatus added in-place.
    """
    for record in extracted:
        assay       = _normalize(record.get("assay", ""))
        ev_level    = _normalize(record.get("evidence_level", ""))
        key_sent    = _normalize(record.get("key_sentence", ""))
        expr_status = _normalize(record.get("expression_status", ""))
        full_text   = assay + " " + ev_level + " " + key_sent + " " + expr_status

        # ── Fix B: if LLM said "no evidence" or key sentence is empty,
        #           go straight to NA — don't let assay field override this.
        #           Prevents: assay="IHC" + evidence_level="no evidence" → PASS
        if "no evidence" in ev_level or not record.get("key_sentence", "").strip():
            record["ManualReviewStatus"] = "NA"
            record["assay_normalized"]   = assay
            continue

        # ── Fix 2: NOT_EXPRESSED — protein assay confirms absence (Tier 1) ─
        # Check expression_status field set by LLM first (most reliable signal),
        # then fall back to keyword detection in full text.
        expr_status_raw = str(record.get("expression_status", "")).lower().strip()
        if expr_status_raw in ("not expressed", "silenced"):
            # Confirm that a Tier-1 protein assay was actually used
            if any(kw in assay or kw in full_text for kw in _TIER1_PROTEIN_ASSAYS):
                record["ManualReviewStatus"] = "NOT_EXPRESSED"
                record["assay_normalized"]   = assay
                continue
        # Fallback: keyword detection in full text (handles cases where LLM
        # set expression_status="unknown" but the key sentence shows loss/silencing)
        #
        # GUARD: skip this fallback when LLM explicitly said "expressed".
        # Rationale: the extraction LLM is given the query's disease context and
        # already applies Rule 8 (adjacent-normal) to pick the correct tissue arm.
        # If it says "expressed", that IS the correct answer for this query context.
        # The keyword scanner reads the raw sentence — for comparison papers like
        # "downregulated in HCC vs adjacent non-cancerous", it sees "downregulated"
        # and wrongly overrides the LLM's normal-tissue reading.
        # Without the guard:  normal-context query → extraction says "expressed"
        #                     → keyword fires → NOT_EXPRESSED  (wrong)
        # With the guard:     normal-context query → extraction says "expressed"
        #                     → keyword skipped  → PASS         (correct)
        # Cancer-context:     extraction says "not expressed"
        #                     → Fix 2a above catches it → NOT_EXPRESSED (correct)
        if expr_status_raw != "expressed" and _is_tier1_absence(assay, full_text):
            record["ManualReviewStatus"] = "NOT_EXPRESSED"
            record["assay_normalized"]   = assay
            continue

        # ── FAIL: RNA-only (no protein assay present) ─────────────────────
        if any(kw in full_text for kw in _RNA_ASSAYS):
            has_protein = any(kw in full_text for kw in _PROTEIN_ASSAYS)
            if not has_protein:
                record["ManualReviewStatus"] = "FAIL"
                record["assay_normalized"]   = assay
                continue

        # ── PASS: protein-level evidence — only if NOT negated (Fix A) ────
        if any(kw in full_text for kw in _PROTEIN_ASSAYS):
            if _is_negated(key_sent):
                # Assay is mentioned but in a negative context → NA
                record["ManualReviewStatus"] = "NA"
                record["validation_note"]    = "Negated assay evidence"
            else:
                record["ManualReviewStatus"] = "PASS"
            record["assay_normalized"] = assay
            continue

        # ── NA: no evidence / ambiguous ───────────────────────────────────
        record["ManualReviewStatus"] = "NA"
        record["assay_normalized"]   = assay

    pass_n  = sum(1 for r in extracted if r.get("ManualReviewStatus") == "PASS")
    notex_n = sum(1 for r in extracted if r.get("ManualReviewStatus") == "NOT_EXPRESSED")
    fail_n  = sum(1 for r in extracted if r.get("ManualReviewStatus") == "FAIL")
    na_n    = sum(1 for r in extracted if r.get("ManualReviewStatus") == "NA")
    logger.info(
        f"[validation] PASS={pass_n} | NOT_EXPRESSED={notex_n} | FAIL={fail_n} | NA={na_n}"
    )

    return extracted
