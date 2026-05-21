"""
agents/output_agent.py — Agent 6: Output Structuring.

Converts the scored records list into a clean pandas DataFrame,
renames columns for readability, reorders them,
and optionally saves to CSV.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

# Column rename map (internal key → display name)
_RENAME = {
    "gene":              "Marker Gene",
    "cell_type_query":   "Cell Type (Query)",
    "disease":           "Disease Context",
    "pmid":              "Reference (PMID)",
    "source":            "Data Source",
    "assay":             "Assay",
    "cell_type":         "Detected Cell Type",
    "tissue":            "Tissue Specificity",
    "localization":      "Localization",
    "evidence_level":    "Evidence Level",
    "expression_status": "Expression Status",   # Fix 2 — new field
    "confidence":        "LLM Confidence",
    "key_sentence":      "Key Evidence Sentence",
    "rag_chunks_used":   "RAG Chunks Used",
    "Evidence":          "Evidence Tier",
    "evidence_score":    "Evidence Score",
    "ManualReviewStatus": "Validation Status",
}

# Preferred column order for the display table
_COLUMN_ORDER = [
    "Marker Gene",
    "Cell Type (Query)",
    "Disease Context",
    "Detected Cell Type",
    "Assay",
    "Expression Status",        # Fix 2 — new column (expressed / not expressed / silenced)
    "Tissue Specificity",
    "Localization",
    "Key Evidence Sentence",
    "Evidence Level",
    "LLM Confidence",
    "Evidence Tier",
    "Evidence Score",
    "Validation Status",
    "Reference (PMID)",
    "Data Source",
    "RAG Chunks Used",
]


def output_agent(
    results:  list[dict],
    out_path: str | None = None,
) -> pd.DataFrame:
    """
    Agent 6 — Output Structuring.

    Args:
        results:  List of scored records from scoring_agent.
        out_path: If provided, save CSV to this path.

    Returns:
        Formatted pd.DataFrame.
    """
    if not results:
        logger.warning("[output] No results to structure")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Ensure all source columns exist before rename (fill missing with "")
    for col in _RENAME:
        if col not in df.columns:
            df[col] = ""

    df = df.rename(columns=_RENAME)

    # Reorder columns (only those present in the DataFrame)
    ordered = [c for c in _COLUMN_ORDER if c in df.columns]
    extra   = [c for c in df.columns if c not in ordered]
    df      = df[ordered + extra]

    # Sort by Evidence Score descending
    if "Evidence Score" in df.columns:
        df = df.sort_values("Evidence Score", ascending=False).reset_index(drop=True)

    # Save CSV
    if out_path:
        try:
            df.to_csv(out_path, index=False, encoding="utf-8")
            logger.info(f"[output] Saved {len(df)} rows → {out_path}")
        except Exception as exc:
            logger.error(f"[output] Failed to save CSV: {exc}")

    logger.info(f"[output] Structured {len(df)} evidence records")
    return df
