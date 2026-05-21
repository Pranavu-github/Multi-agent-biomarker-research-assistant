"""
mcp_server/server.py — FastMCP Biomarker Research Server.

Exposes a single tool: run_pipeline(gene, cell, disease)
which runs the full 6-agent biomarker validation pipeline
and returns a JSON-serialisable evidence table.

Start with:
    python mcp_server/server.py

Or in your Claude / MCP client config:
    {
      "mcpServers": {
        "biomarker": {
          "command": "python",
          "args": ["/path/to/biomarker-simplified-workflow/mcp_server/server.py"]
        }
      }
    }
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

# Add project root to path so pipeline.py imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastmcp import FastMCP
from loguru import logger

# ── Create MCP server ─────────────────────────────────────────────────────────
mcp = FastMCP(
    name="biomarker-research",
    instructions=(
        "Biomarker Validation Research Assistant. "
        "Use run_pipeline to validate whether a gene is expressed "
        "in a specific cell type, with optional disease context."
    ),
)


# ── Tool: run_pipeline ────────────────────────────────────────────────────────

@mcp.tool()
def run_pipeline(
    gene:    str,
    cell:    str,
    disease: str = "",
) -> dict:
    """
    Run the full biomarker validation pipeline for a gene + cell type.

    Queries PubMed, Europe PMC, Semantic Scholar, Human Protein Atlas,
    CellMarker, and UniProt. Extracts protein-level experimental evidence
    (IHC, IF, FACS) per paper using FAISS + BM25 RAG and LLM extraction.

    Args:
        gene:    Gene or protein symbol (e.g. "SOX18", "DARC", "IFNGR1").
        cell:    Cell type (e.g. "Endothelial cells", "NK cells").
        disease: Disease context (e.g. "breast cancer", "NSCLC").
                 Leave empty for normal tissue context.

    Returns:
        {
            "status":    "success" | "error",
            "gene":      str,
            "cell":      str,
            "disease":   str,
            "n_records": int,
            "records":   [
                {
                    "Marker Gene":         str,
                    "Cell Type (Query)":   str,
                    "Disease Context":     str,
                    "Assay":               str,
                    "Detected Cell Type":  str,
                    "Tissue Specificity":  str,
                    "Evidence Level":      str,
                    "Evidence Tier":       str,
                    "Evidence Score":      float,
                    "Validation Status":   str,   # PASS / FAIL / NA
                    "Key Evidence Sentence": str,
                    "Reference (PMID)":    str,
                    "Data Source":         str,
                },
                ...
            ],
            "error": str  (only present on error)
        }
    """
    logger.info(f"[MCP] run_pipeline called | gene={gene} | cell={cell} | disease={disease}")

    try:
        from pipeline import run_pipeline as _run
        df = _run(gene=gene, cell=cell, disease=disease)

        if df.empty:
            return {
                "status":    "success",
                "gene":      gene,
                "cell":      cell,
                "disease":   disease,
                "n_records": 0,
                "records":   [],
                "message":   "No evidence records found for the given inputs.",
            }

        # Serialise DataFrame rows as list of dicts
        records = json.loads(df.to_json(orient="records"))

        return {
            "status":    "success",
            "gene":      gene,
            "cell":      cell,
            "disease":   disease,
            "n_records": len(records),
            "records":   records,
        }

    except Exception as exc:
        logger.error(f"[MCP] Pipeline error: {exc}")
        return {
            "status":  "error",
            "gene":    gene,
            "cell":    cell,
            "disease": disease,
            "error":   str(exc),
        }


# ── Health-check resource ─────────────────────────────────────────────────────

@mcp.resource("biomarker://info")
def server_info() -> str:
    """Return server capabilities and version info."""
    return json.dumps({
        "name":    "Biomarker Research Server",
        "version": "1.0.0",
        "tools":   ["run_pipeline"],
        "sources": [
            "PubMed", "EuropePMC", "SemanticScholar",
            "HumanProteinAtlas", "CellMarker", "UniProt",
            "bioRxiv/medRxiv",
        ],
        "rag":     "FAISS + BM25 hybrid (no cross-encoder)",
    }, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Biomarker MCP Server...")
    mcp.run()
