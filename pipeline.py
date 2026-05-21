"""
pipeline.py — Biomarker Validation Pipeline Orchestrator.

Runs the 6-agent pipeline end-to-end for one gene + cell type + disease.

Agent flow:
  [1] Planner     → Generate disease-aware queries + gene synonyms
  [2] Retrieval   → Fetch papers from PubMed, EPMC, SS, databases
  [3] Extraction  → FAISS+BM25 RAG → LLM extracts one record per PMID
  [4] Validation  → Classify PASS / FAIL / NA
  [5] Scoring     → Tier + numeric score
  [6] Output      → Clean DataFrame

Usage:
    from pipeline import run_pipeline
    df = run_pipeline("SOX18", "Endothelial cells", "breast cancer")
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make sure local packages are importable
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
import pandas as pd

from agents.planner_agent    import planner_agent
from agents.retrieval_agent  import retrieval_agent
from agents.extraction_agent import extraction_agent
from agents.validation_agent import validation_agent
from agents.scoring_agent    import scoring_agent
from agents.output_agent     import output_agent


def run_pipeline(
    gene:        str,
    cell:        str,
    disease:     str = "",
    out_path:    str | None = None,
    max_per_lit: int = 5,
) -> pd.DataFrame:
    """
    Run the full 6-agent biomarker validation pipeline.

    Args:
        gene:        Gene/protein symbol (e.g. "SOX18").
        cell:        Cell type (e.g. "Endothelial cells").
        disease:     Disease (e.g. "breast cancer") or "" for normal tissue.
        out_path:    Optional CSV output path.
        max_per_lit: Max papers per query per literature source.

    Returns:
        pd.DataFrame with one row per PMID evidence record.
    """
    gene    = gene.strip()
    cell    = cell.strip()
    disease = disease.strip()

    logger.info("=" * 60)
    logger.info(f"PIPELINE START | Gene: {gene} | Cell: {cell} | Disease: {disease or 'normal'}")
    logger.info("=" * 60)

    t0 = time.time()

    # Store original disease label for display (may get expanded by planner e.g. NSCLC)
    disease_display = disease

    # ── Agent 1: Planner ─────────────────────────────────────────────────────
    logger.info("[1/6] Planner agent...")
    plan = planner_agent(gene=gene, celltype=cell, disease=disease)
    # Use the disease as expanded by the planner (e.g. NSCLC → non-small cell lung cancer)
    # so retrieval, extraction, validation and scoring are all consistent.
    disease = plan["disease"]
    logger.info(f"      Synonyms: {plan['synonyms']}")
    logger.info(f"      Subtypes: {plan.get('cell_subtypes', [])}")
    logger.info(f"      Disease : {disease}")
    logger.info(f"      Queries : {len(plan['queries'])}")

    if not plan.get("queries"):
        logger.error("[pipeline] Planner returned no queries — aborting")
        return pd.DataFrame()

    # ── Agent 2: Retrieval ───────────────────────────────────────────────────
    logger.info("[2/6] Retrieval agent...")
    papers = retrieval_agent(plan, max_per_lit=max_per_lit)
    logger.info(f"      Papers retrieved: {len(papers)}")

    if not papers:
        logger.warning("[pipeline] No papers retrieved — returning empty DataFrame")
        return pd.DataFrame()

    # ── Agent 3: Extraction ──────────────────────────────────────────────────
    logger.info("[3/6] Extraction agent (RAG + LLM)...")
    extracted = extraction_agent(
        papers=papers, gene=gene, cell=cell, disease=disease,
        cell_subtypes=plan.get("cell_subtypes"),   # NK cell synonyms, T cell subtypes, etc.
    )
    logger.info(f"      Records extracted: {len(extracted)}")

    if not extracted:
        logger.warning("[pipeline] Extraction returned nothing — returning empty DataFrame")
        return pd.DataFrame()

    # ── Agent 4: Validation ──────────────────────────────────────────────────
    logger.info("[4/6] Validation agent...")
    validated = validation_agent(extracted)

    # ── Agent 5: Scoring ─────────────────────────────────────────────────────
    logger.info("[5/6] Scoring agent...")
    scored = scoring_agent(validated)

    # ── Agent 6: Output ──────────────────────────────────────────────────────
    logger.info("[6/6] Output agent...")
    df = output_agent(scored, out_path=out_path)

    elapsed = round(time.time() - t0, 1)
    logger.info("=" * 60)
    logger.info(f"PIPELINE DONE | {len(df)} records | {elapsed}s elapsed")
    logger.info("=" * 60)

    return df


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    input_data,
    out_path:    str | None = None,
    max_per_lit: int = 5,
) -> pd.DataFrame:
    """
    Run the pipeline for multiple gene + cell type pairs.

    Args:
        input_data: Any format supported by utils.input_handler.load_input_data().
        out_path:   Optional CSV output path for combined results.
        max_per_lit: Max papers per query per literature source.

    Returns:
        Combined pd.DataFrame with results for all inputs.
    """
    from utils.input_handler import load_input_data
    df_input = load_input_data(input_data)
    logger.info(f"[batch] Processing {len(df_input)} gene(s)")

    all_results: list[pd.DataFrame] = []

    for idx, row in enumerate(df_input.itertuples(), start=1):
        gene    = str(row.Gene).strip()
        cell    = str(row.Cell_Type).strip()
        disease = str(row.Disease).strip() if hasattr(row, "Disease") else ""

        logger.info(f"\n[batch] {idx}/{len(df_input)}: {gene} | {cell} | {disease or 'normal'}")
        try:
            df_result = run_pipeline(gene, cell, disease, max_per_lit=max_per_lit)
            all_results.append(df_result)
        except Exception as exc:
            logger.error(f"[batch] Error for {gene} / {cell}: {exc}")
            continue

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    if out_path:
        combined.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"[batch] Saved {len(combined)} total rows → {out_path}")

    return combined


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Biomarker Validation Pipeline")
    parser.add_argument("--gene",    required=True,  help="Gene/protein symbol")
    parser.add_argument("--cell",    required=True,  help="Cell type")
    parser.add_argument("--disease", default="",     help="Disease context (optional)")
    parser.add_argument("--out",     default="biomarker_results.csv", help="Output CSV path")
    args = parser.parse_args()

    result_df = run_pipeline(
        gene=args.gene, cell=args.cell,
        disease=args.disease, out_path=args.out,
    )
    print(result_df.to_string(index=False))
