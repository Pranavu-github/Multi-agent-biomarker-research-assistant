"""
backend.py — FastAPI server for the Biomarker Validation Pipeline.

Start with:
    uvicorn backend:app --host 127.0.0.1 --port 8001 --reload

Endpoints:
    GET  /health                    → liveness check
    POST /pipeline/run              → submit job, returns job_id
    GET  /pipeline/status/{job_id}  → poll stage + results
    DELETE /pipeline/job/{job_id}   → clean up completed job
"""

from __future__ import annotations

import sys
import json
import time
import uuid
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Make local packages importable
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Biomarker Pipeline API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Job store ─────────────────────────────────────────────────────────────────
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_executor  = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")

# Agent stage labels (used for live progress in the UI)
STAGES = [
    "Queued",
    "Agent 1 — Planner (synonyms + queries)",
    "Agent 2 — Retrieval (PubMed · EPMC · SS · databases)",
    "Agent 3 — Extraction (FAISS+BM25 RAG + LLM)",
    "Agent 4 — Validation (PASS / FAIL / NA)",
    "Agent 5 — Scoring (evidence tiers)",
    "Agent 6 — Output (structuring table)",
    "Done",
]


# ── Pipeline worker ───────────────────────────────────────────────────────────

def _set_job(job_id: str, stage: str, pct: int, **kw):
    """Thread-safe job state update."""
    with _jobs_lock:
        _jobs[job_id].update({"stage": stage, "pct": pct, **kw})


def _run_pipeline_job(
    job_id:      str,
    gene:        str,
    cell:        str,
    disease:     str,
    max_per_lit: int,
):
    """Runs the 6-agent pipeline in a background thread, updating job state at each stage."""
    from collections import Counter

    _set_job(job_id, STAGES[0], 0)
    t0   = time.time()
    diag: dict[str, Any] = {
        "query": {
            "gene":    gene,
            "cell":    cell,
            "disease": disease or "(normal tissue — no disease specified)",
        }
    }

    try:
        # ── Agent 1 — Planner ────────────────────────────────────────────────
        _set_job(job_id, STAGES[1], 10)
        from agents.planner_agent import planner_agent
        plan = planner_agent(gene=gene, celltype=cell, disease=disease)
        if not plan.get("queries"):
            raise RuntimeError("Planner returned no queries — check LLM API key")

        # Use the disease as expanded by the planner (e.g. NSCLC → non-small cell
        # lung cancer) so all downstream agents are consistent.
        disease = plan["disease"]

        # Update diag query block with expanded disease
        diag["query"]["disease_expanded"] = disease

        diag["planner"] = {
            "synonyms":              plan.get("synonyms", []),
            "cell_subtypes":         plan.get("cell_subtypes", []),
            "disease_original":      plan.get("disease_original", ""),
            "disease_expanded":      disease,
            "disease_synonyms":      plan.get("disease_synonyms", []),
            "anchor_queries":        plan.get("anchor_queries", []),
            "llm_queries":           plan.get("llm_queries", []),
            "total_unique_queries":  len(plan.get("queries", [])),
            "queries_to_retrieval":  plan.get("queries", [])[:5],
        }

        # ── Agent 2 — Retrieval ──────────────────────────────────────────────
        _set_job(job_id, STAGES[2], 25)
        from agents.retrieval_agent import retrieval_agent
        papers = retrieval_agent(plan, max_per_lit=max_per_lit)

        diag["retrieval"] = {
            "papers_retrieved": len(papers),
            "by_source":        dict(Counter(p.get("source", "?") for p in papers)),
            "papers": [
                {
                    "pmid":              p.get("pmid", "N/A"),
                    "source":            p.get("source", "?"),
                    "gene_in_abstract":  gene.lower() in (p.get("abstract", "") or "").lower(),
                    "access_type":       p.get("access_type", "abstract_only"),
                    "snippet":           (p.get("abstract", "") or "")[:200],
                }
                for p in papers
            ],
        }

        if not papers:
            _set_job(job_id, "Done", 100,
                     status="done", records=[], n=0,
                     elapsed=round(time.time() - t0, 1),
                     warning="No papers retrieved — try different inputs",
                     diag=diag)
            return

        # ── Agent 3 — Extraction ─────────────────────────────────────────────
        # Pass cell_subtypes from the plan so the LLM knows to accept
        # subtype evidence (e.g. "dendritic cells" for query "myeloid cells").
        _set_job(job_id, STAGES[3], 50)
        from agents.extraction_agent import extraction_agent
        extracted = extraction_agent(
            papers=papers, gene=gene, cell=cell, disease=disease,
            cell_subtypes=plan.get("cell_subtypes"),
        )

        diag["extraction"] = {
            "records_extracted": len(extracted),
            "no_evidence_count": sum(
                1 for r in extracted
                if r.get("evidence_level", "").lower() == "no evidence"
            ),
            "evidence_found_count": sum(
                1 for r in extracted
                if r.get("evidence_level", "").lower() not in ("no evidence", "")
            ),
            "per_pmid": {
                r.get("pmid", "?"): {
                    "evidence_level":    r.get("evidence_level", ""),
                    "assay":             r.get("assay", ""),
                    "expression_status": r.get("expression_status", ""),
                    "key_sentence":      (r.get("key_sentence") or "")[:200],
                    "rag_chunks_used":   r.get("rag_chunks_used", 0),
                    "source":            r.get("source", ""),
                }
                for r in extracted
            },
        }

        # ── Agent 4 — Validation ─────────────────────────────────────────────
        _set_job(job_id, STAGES[4], 70)
        from agents.validation_agent import validation_agent
        validated = validation_agent(extracted)

        diag["validation"] = dict(
            Counter(r.get("ManualReviewStatus", "?") for r in validated)
        )

        # ── Agent 5 — Scoring ────────────────────────────────────────────────
        _set_job(job_id, STAGES[5], 85)
        from agents.scoring_agent import scoring_agent
        scored = scoring_agent(validated)

        diag["scoring"] = {
            "tier_summary": dict(Counter(r.get("Evidence", "?") for r in scored)),
            "top_score":    max((r.get("evidence_score", 0) for r in scored), default=0),
        }

        # ── Agent 6 — Output ─────────────────────────────────────────────────
        _set_job(job_id, STAGES[6], 95)
        from agents.output_agent import output_agent
        df = output_agent(scored)

        elapsed = round(time.time() - t0, 1)
        records = json.loads(df.to_json(orient="records")) if not df.empty else []

        _set_job(job_id, "Done", 100,
                 status="done", records=records, n=len(records),
                 elapsed=elapsed, diag=diag)

    except Exception as exc:
        _set_job(job_id, "Error", 100,
                 status="error",
                 error=str(exc),
                 trace=traceback.format_exc(),
                 elapsed=round(time.time() - t0, 1),
                 diag=diag)


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    gene:        str
    cell:        str
    disease:     str = ""
    max_per_lit: int = 5


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — also shows active job count."""
    return {"status": "ok", "active_jobs": len(_jobs)}


@app.post("/pipeline/run")
def run_pipeline(req: RunRequest):
    """
    Submit a new pipeline job.

    Returns:
        {"job_id": str, "status": "running"}
    """
    if not req.gene.strip():
        raise HTTPException(status_code=400, detail="gene is required")
    if not req.cell.strip():
        raise HTTPException(status_code=400, detail="cell is required")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status":  "running",
            "stage":   STAGES[0],
            "pct":     0,
            "records": [],
            "n":       0,
            "elapsed": 0,
            "gene":    req.gene,
            "cell":    req.cell,
            "disease": req.disease,
        }

    _executor.submit(
        _run_pipeline_job,
        job_id,
        req.gene.strip(),
        req.cell.strip(),
        req.disease.strip(),
        req.max_per_lit,
    )
    return {"job_id": job_id, "status": "running"}


@app.get("/pipeline/status/{job_id}")
def get_status(job_id: str):
    """
    Poll job status.

    Returns job dict with fields:
        status  : "running" | "done" | "error"
        stage   : current agent name
        pct     : 0–100
        records : list of evidence dicts (populated when done)
        elapsed : seconds
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.delete("/pipeline/job/{job_id}")
def delete_job(job_id: str):
    """Remove a completed job from the store."""
    with _jobs_lock:
        _jobs.pop(job_id, None)
    return {"deleted": job_id}


@app.get("/pipeline/stages")
def get_stages():
    """Return the ordered list of agent stage labels (used by UI progress bar)."""
    return {"stages": STAGES}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8001, reload=True)
