# Biomarker Research Assistant — Internal Reference (Detailed)

> **Audience:** Developer / researcher reference only. Contains implementation internals, bug history, scoring formulas, regex patterns, known limitations, and design rationale. Do not share externally.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Architecture Overview](#2-architecture-overview)
3. [Complete File Reference](#3-complete-file-reference)
4. [Setup and Environment](#4-setup-and-environment)
5. [Running the Application](#5-running-the-application)
6. [Agent Pipeline — Detailed Walkthrough](#6-agent-pipeline--detailed-walkthrough)
   - [Agent 1 — Planner](#agent-1--planner-planner_agentpy)
   - [Agent 2 — Retrieval](#agent-2--retrieval-retrieval_agentpy)
   - [Agent 3 — Extraction](#agent-3--extraction-extraction_agentpy)
   - [Agent 4 — Validation](#agent-4--validation-validation_agentpy)
   - [Agent 5 — Scoring](#agent-5--scoring-scoring_agentpy)
   - [Agent 6 — Output](#agent-6--output-output_agentpy)
7. [Utility Modules](#7-utility-modules)
   - [RAG Store (FAISS + BM25)](#rag-store-rag_storepy)
   - [RAG Context Builder](#rag-context-builder-rag_context_builderpy)
   - [LLM Client](#llm-client-llm_clientpy)
   - [Full Text Fetcher](#full-text-fetcher-full_text_fetcherpy)
8. [Evidence Hierarchy and Rules](#8-evidence-hierarchy-and-rules)
9. [Disease Context Enforcement](#9-disease-context-enforcement)
10. [Cell Lineage Hierarchy Map](#10-cell-lineage-hierarchy-map)
11. [Validation Logic — Internal Detail](#11-validation-logic--internal-detail)
12. [Scoring Logic — Internal Detail](#12-scoring-logic--internal-detail)
13. [Output Column Schema](#13-output-column-schema)
14. [Backend API (FastAPI)](#14-backend-api-fastapi)
15. [MCP Server](#15-mcp-server)
16. [History of Bugs and Fixes](#16-history-of-bugs-and-fixes)
17. [Speed Optimisations Applied](#17-speed-optimisations-applied)
18. [Known Limitations and Edge Cases](#18-known-limitations-and-edge-cases)
19. [Testing](#19-testing)
20. [Design Decisions and Rationale](#20-design-decisions-and-rationale)

---

## 1. Project Purpose

This pipeline validates whether a **gene/protein** is expressed in a **specific cell type**, optionally within a **disease context** (e.g. breast cancer). It was built to replace manual literature review with a semi-automated, evidence-tiered system that returns structured, reproducible results comparable to what a bench researcher would compile.

The core scientific question the system answers:

> *"Is gene X expressed (or absent) in cell type Y, in disease Z, based on protein-level experimental evidence?"*

**Why protein-level matters:** RNA expression data (RNA-seq, qPCR) does not reliably predict protein presence or localisation. The pipeline strictly prioritises IHC, IF/immunofluorescence, and FACS/Flow cytometry as Tier 1 evidence. This mirrors the standard in academic biomarker validation papers.

---

## 2. Architecture Overview

```
User Input (gene + cell + disease)
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │               FastAPI Backend (backend.py)               │
  │  Job queue: ThreadPoolExecutor (4 workers)               │
  │  Endpoints: /pipeline/run  /pipeline/status  /health     │
  └────────────────────────┬────────────────────────────────┘
                           │ submits background job
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │                  6-Agent Pipeline                        │
  │                                                         │
  │  Agent 1: Planner ──────────────────────────────────────┤
  │    • LLM gene synonym expansion                         │
  │    • Cell lineage hierarchy expansion (_CELL_LINEAGE_MAP)│
  │    • LLM query generation (6–8 queries)                  │
  │                                                         │
  │  Agent 2: Retrieval ────────────────────────────────────┤
  │    • PubMed (Biopython Entrez)                          │
  │    • Europe PMC (REST, sort=cited desc)                 │
  │    • Semantic Scholar (REST, rate-limit aware)          │
  │    • Human Protein Atlas (REST)                         │
  │    • CellMarker (HTML scrape)                           │
  │    • UniProt (REST)                                     │
  │    • All sources fired concurrently (10 workers)        │
  │    • Disease pre-filter, dedup, cap at 20 papers        │
  │                                                         │
  │  Agent 3: Extraction ───────────────────────────────────┤
  │    • Ingest papers → FAISS + BM25 RAG store             │
  │    • One record per PMID (3–10 rows final)              │
  │    • LLM extraction per PMID (5 parallel workers)       │
  │    • Extracts: assay, cell_type, tissue, localization,  │
  │      evidence_level, expression_status, key_sentence    │
  │                                                         │
  │  Agent 4: Validation ───────────────────────────────────┤
  │    • PASS / NOT_EXPRESSED / FAIL / NA                   │
  │    • Word-boundary negation detection (regex)           │
  │    • RT-PCR/methylation-PCR → FAIL (not Tier 1)        │
  │                                                         │
  │  Agent 5: Scoring ──────────────────────────────────────┤
  │    • Evidence Tier: Primary / Secondary / Tertiary      │
  │    • Numeric score 0.0–1.0 with penalties               │
  │                                                         │
  │  Agent 6: Output ───────────────────────────────────────┤
  │    • Pandas DataFrame, column rename + reorder          │
  │    • Sort by evidence_score descending                  │
  └─────────────────────────────────────────────────────────┘
                           │
                           ▼
              Streamlit Frontend (app.py)
              Live progress bar + evidence table
```

---

## 3. Complete File Reference

```
biomarker-simplified-workflow/
│
├── agents/
│   ├── __init__.py
│   ├── planner_agent.py       # Agent 1 — gene synonyms, lineage expansion, queries
│   ├── retrieval_agent.py     # Agent 2 — 6 sources, concurrent, disease-sorted
│   ├── extraction_agent.py    # Agent 3 — FAISS+BM25 RAG, LLM per PMID
│   ├── validation_agent.py    # Agent 4 — PASS/NOT_EXPRESSED/FAIL/NA rules
│   ├── scoring_agent.py       # Agent 5 — evidence tiers + scores
│   └── output_agent.py        # Agent 6 — DataFrame structuring, column rename
│
├── utils/
│   ├── __init__.py
│   ├── rag_store.py           # BiomarkerRAGStore — FAISS + BM25 + RRF fusion
│   ├── rag_context_builder.py # build_rag_context() — multi-query context assembly
│   ├── llm_client.py          # call_llm() — Anthropic primary, Groq fallback
│   ├── full_text_fetcher.py   # PMC → EPMC → Unpaywall → abstract fallback
│   └── input_handler.py       # load_input_data() — normalises dict/string/CSV input
│
├── prompts/
│   └── planner_prompt.txt     # Prompt template for LLM query generation
│
├── mcp_server/
│   ├── __init__.py
│   └── server.py              # FastMCP server — run_pipeline() tool exposed
│
├── tests/
│   └── test_pipeline.py       # Unit tests (no LLM calls): RAG, validation, scoring
│
├── app.py                     # Streamlit frontend — job polling, progress bar, table
├── backend.py                 # FastAPI backend — job queue, 4 endpoints
├── pipeline.py                # Standalone runner — returns DataFrame (no UI)
├── requirements.txt           # All dependencies
├── .env.example               # API key template
└── README.md                  # External-facing readme
```

---

## 4. Setup and Environment

### Python Version

Python 3.10+ required. Tested on 3.10.

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages and why they exist:

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | ≥0.25 | Primary LLM (Claude claude-sonnet-4-6) |
| `groq` | ≥0.5 | Fallback LLM (llama3-70b-8192) |
| `faiss-cpu` | ≥1.7.4 | Dense vector index for RAG (cosine via IndexFlatIP) |
| `sentence-transformers` | ≥2.5 | Embedding model: `all-MiniLM-L6-v2` (384 dim) |
| `rank-bm25` | ≥0.2.2 | BM25Okapi sparse retrieval |
| `langchain-text-splitters` | ≥0.2 | RecursiveCharacterTextSplitter for chunking |
| `biopython` | ≥1.83 | Entrez API (PubMed, PMC) |
| `beautifulsoup4` / `lxml` | latest | HTML scraping (CellMarker, Unpaywall) |
| `fastapi` + `uvicorn` | latest | Backend REST API |
| `fastmcp` | ≥0.4 | MCP server framework |
| `streamlit` | ≥1.32 | Frontend web UI |
| `loguru` | ≥0.7 | Structured logging across all agents |

### Environment Variables

```bash
cp .env.example .env
# then edit .env with real keys
```

| Variable | Required? | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Required** | Claude claude-sonnet-4-6 — all LLM calls route here first |
| `ENTREZ_EMAIL` | **Required** | NCBI policy — must be a real email for PubMed/PMC access |
| `GROQ_API_KEY` | Optional | LLM fallback. Model: `llama3-70b-8192`. Activates automatically if Anthropic fails |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Without key: 100 req/5min limit. With key: 1 req/sec |
| `UNPAYWALL_EMAIL` | Optional | Enables Unpaywall full-text. Without it, Unpaywall is skipped |

---

## 5. Running the Application

Two terminals required — backend and frontend must run simultaneously.

### Terminal 1 — FastAPI Backend

```bash
cd biomarker-simplified-workflow
uvicorn backend:app --host 127.0.0.1 --port 8001 --reload
```

`--reload` watches for file changes (useful during development). The `--host 127.0.0.1` restricts access to localhost — do not expose to 0.0.0.0 in production without authentication.

### Terminal 2 — Streamlit Frontend

```bash
streamlit run app.py
```

Frontend opens at `http://localhost:8501`.

The Streamlit app polls `/pipeline/status/{job_id}` every second and updates a live progress bar as each agent stage completes.

### Standalone (No UI)

```bash
python pipeline.py
```

Or import directly:

```python
from pipeline import run_pipeline
df = run_pipeline(gene="SOX18", cell="endothelial cells", disease="breast cancer")
print(df.to_string())
```

---

## 6. Agent Pipeline — Detailed Walkthrough

### Agent 1 — Planner (`planner_agent.py`)

**Purpose:** Maximise recall by generating the best possible search queries before hitting any database.

#### Step 1a: Gene Synonym Expansion

Calls the LLM to return all known aliases, previous symbols, protein names, and common abbreviations for the input gene:

```python
expand_gene_synonyms("KLRC2")
# Returns: ["KLRC2", "NKG2C", "CD159c", "killer cell lectin-like receptor C2"]
```

**Critical disambiguation rule (added after KLRC2/NKG2A bug):** The LLM prompt explicitly instructs: *"Only include aliases for this exact gene. Do NOT include aliases for related paralogs or family members."* — This prevents, for example, KLRC1/NKG2A being included as an alias for KLRC2/NKG2C (they are different genes in the same family but with different expression patterns).

Fallback: if LLM call fails, returns `[gene]` (just the input symbol, no expansion).

#### Step 1b: Cell Type Lineage Expansion

`expand_cell_subtypes(celltype)` looks up the input cell type in `_CELL_LINEAGE_MAP`. If a mapping exists, it returns specific subtypes. If not (input is already specific), it returns an empty list and the query generator uses only the exact term.

The full lineage map is:

```python
_CELL_LINEAGE_MAP = {
    # Myeloid
    "myeloid cells":           ["dendritic cells", "monocytes", "macrophages",
                                 "neutrophils", "myeloid dendritic cells",
                                 "plasmacytoid dendritic cells", "cDC1", "BDCA3+ dendritic cells"],
    "myeloid":                 ["dendritic cells", "monocytes", "macrophages", "neutrophils"],
    "monocytes":               ["classical monocytes", "non-classical monocytes",
                                "CD14+ monocytes", "CD16+ monocytes"],
    "macrophages":             ["tumor-associated macrophages", "TAM", "M1 macrophages",
                                "M2 macrophages", "tissue-resident macrophages"],
    "dendritic cells":         ["plasmacytoid dendritic cells", "pDC", "myeloid dendritic cells",
                                "conventional dendritic cells", "cDC1", "cDC2",
                                "BDCA3+ dendritic cells", "CD141+ dendritic cells"],

    # Lymphoid
    "lymphoid cells":          ["T cells", "B cells", "NK cells", "innate lymphoid cells"],
    "t cells":                 ["CD4+ T cells", "CD8+ T cells", "regulatory T cells",
                                "Treg", "cytotoxic T lymphocytes", "CTL", "NKT cells"],
    "cd4+ t cells":            ["helper T cells", "Th1", "Th2", "Th17", "Treg"],
    "cd8+ t cells":            ["cytotoxic T lymphocytes", "CTL"],
    "nk cells":                ["natural killer cells", "CD56+ cells",
                                "CD56bright NK", "CD56dim NK"],
    "b cells":                 ["plasma cells", "memory B cells", "naive B cells",
                                "germinal center B cells"],
    "innate lymphoid cells":   ["ILC1", "ILC2", "ILC3"],

    # Stromal / structural
    "stromal cells":           ["fibroblasts", "cancer-associated fibroblasts", "CAF",
                                "endothelial cells", "pericytes", "myofibroblasts"],
    "fibroblasts":             ["cancer-associated fibroblasts", "CAF",
                                "activated fibroblasts", "myofibroblasts"],
    "endothelial cells":       ["vascular endothelial cells", "lymphatic endothelial cells",
                                "tumor endothelial cells", "HUVEC"],

    # Epithelial — normal
    "epithelial cells":        ["tumor cells", "cancer cells", "carcinoma cells"],

    # Cancer epithelial — Fix 1 (added for tumor suppressor queries like SLC5A8)
    "cancer epithelial":       ["breast cancer cells", "tumor cells", "carcinoma cells",
                                "cancer cells", "epithelial tumor cells",
                                "luminal epithelial cells", "malignant epithelial cells",
                                "colorectal cancer cells", "lung cancer cells",
                                "gastric cancer cells", "thyroid cancer cells"],
    "cancer epithelial cells": ["breast cancer cells", "tumor cells", "carcinoma cells",
                                "cancer cells", "malignant epithelial cells",
                                "luminal epithelial cells", "epithelial tumor cells"],
    "tumor cells":             ["cancer cells", "carcinoma cells", "malignant cells",
                                "breast cancer cells", "tumor epithelial cells"],
    "cancer cells":            ["tumor cells", "carcinoma cells", "malignant cells"],
    "carcinoma cells":         ["adenocarcinoma cells", "squamous cell carcinoma cells",
                                "tumor cells", "cancer cells"],

    # Broad immune
    "immune cells":            ["T cells", "B cells", "NK cells", "dendritic cells",
                                "macrophages", "monocytes", "neutrophils"],
    "tumor-infiltrating cells":["tumor-infiltrating lymphocytes", "TIL",
                                "tumor-infiltrating macrophages", "tumor-infiltrating NK cells"],
}
```

**Why this matters:** Papers rarely say *"myeloid cells expressed CLEC9A"* — they say *"BDCA3+ dendritic cells expressed CLEC9A."* Without subtype expansion, these papers are never retrieved.

#### Step 1c: LLM Query Generation

Uses `prompts/planner_prompt.txt` as a template, filling in:
- `{gene}` — canonical gene symbol
- `{celltype}` — user's cell type input (broad term)
- `{disease}` — disease or "None (use normal tissue context)"
- `{synonyms}` — comma-separated synonym list from step 1a
- `{cell_subtypes}` — comma-separated subtype list from step 1b

The prompt instructs the LLM to generate **6–8 queries** across these types:
1. Specific subtype + protein assay + disease (most important — 2–3 queries)
2. Broad term + protein assay + disease
3. Clinical validation (patient tissue + protein expression)
4. Expression profiling (no assay restriction)
5–6. Discovery queries (gene signature, immune panel, TIL studies)

Also instructs abbreviation expansion: `CAF → "cancer-associated fibroblast"`, `NK → "natural killer cell"`, `DC → "dendritic cell"`, etc.

Fallback if LLM fails: `_fallback_queries()` generates deterministic queries from synonyms × cell terms × assay keywords.

#### Planner Output Dict

```python
{
    "gene":          "CLEC9A",
    "celltype":      "myeloid cells",
    "disease":       "breast cancer",
    "synonyms":      ["CLEC9A", "DNGR-1", "dendritic cell NK lectin group receptor-1"],
    "cell_subtypes": ["dendritic cells", "monocytes", "macrophages", ...],
    "queries":       ["CLEC9A BDCA3+ dendritic cells IHC breast cancer", ...],
}
```

---

### Agent 2 — Retrieval (`retrieval_agent.py`)

**Purpose:** Fetch raw evidence from 6 sources, deduplicate, disease-sort, cap at 20.

#### Sources

| Source | API / Method | Max Results | Notes |
|---|---|---|---|
| PubMed | Biopython Entrez.esearch + efetch | `max_per_lit` per query (default 5) | 0.4s sleep between IDs (NCBI rate limit) |
| Europe PMC | REST `/search` | `max_per_lit` per query | `sort=cited desc` (not default date-sort) — critical to avoid brand-new 2025/2026 papers dominating |
| Semantic Scholar | REST `/paper/search` | 3 per query (first 2 queries only) | Rate-limit aware: retries only on HTTP 429, no unconditional sleep |
| Human Protein Atlas | REST search API | 1 record per gene | Returns gene-level RNA tissue category summary |
| CellMarker | HTML scrape | 1 record per gene | Uses BeautifulSoup + lxml |
| UniProt | REST search | 1 record per gene | Reviewed=true, organism_id=9606 (human only) |

#### S1 — Abstract Relevance Filter (Speed Optimisation)

Before attempting an expensive PMC/EPMC full-text fetch (which can add 4–8 seconds per paper), the system checks if the gene symbol appears anywhere in the abstract:

```python
def _fetch_full_text_if_relevant(pmid, abstract, doi, gene):
    if gene.lower() not in abstract.lower():
        return {"sections": [...abstract only...], "access_type": "abstract_only"}
    return fetch_full_text_or_abstract(pmid, abstract, doi)
```

If the gene isn't mentioned in the abstract, the paper is extremely unlikely to contain targeted protein-level evidence for that gene — full-text fetch is skipped entirely.

#### Concurrent Retrieval

All tasks are submitted to `ThreadPoolExecutor(max_workers=10)` simultaneously. PubMed, EuropePMC (up to 5 queries each), Semantic Scholar (2 queries), HPA, CellMarker, UniProt all run in parallel. Total retrieval time is bounded by the slowest single source rather than the sum of all sources.

#### Deduplication

`_deduplicate()` removes papers with duplicate PMIDs (keeps first occurrence). Database records with PMID="N/A" get a unique synthetic key so they are never accidentally merged.

#### Disease Pre-filter and Sort

When disease is provided:
```python
relevant  = [p for p in unique if disease_kw in abstract.lower()]  # disease-matched
lit_other = [p for p in unique if disease not in abstract and not database]
db_papers = [p for p in unique if evidence_type == "database"]
unique = relevant + lit_other + db_papers
```

Nothing is discarded — papers are just reordered so disease-relevant ones hit the extraction LLM first and survive the cap.

#### Output Cap (S3)

Final list sliced to `[:20]` papers. With the disease sort above, the top 20 will always favour disease-matching literature before database records.

---

### Agent 3 — Extraction (`extraction_agent.py`)

**Purpose:** Extract structured evidence from each paper's text using a hybrid RAG + LLM approach.

#### Step 3a: Ingest into RAG Store

All retrieved papers are chunked and indexed into `BiomarkerRAGStore` (FAISS + BM25). See [RAG Store section](#rag-store-rag_storepy) for full internals.

#### Step 3b: Broad Retrieve to Group Chunks by PMID

```python
broad_query = f"{gene} {cell} {disease} expression biomarker IHC IF FACS"
all_chunks = store.retrieve(broad_query, top_k=min(n_chunks, 60))
```

Chunks are grouped into `pmid_chunks` dict. Database papers (HPA, CellMarker, UniProt) with no FAISS chunks are given a pseudo-chunk from their abstract.

#### Step 3c: Per-PMID LLM Extraction (Parallel)

For each unique PMID, takes up to 6 chunks (capped to keep prompt within token limit) and calls the LLM with `_EXTRACT_PROMPT`.

The extraction prompt instructs the LLM to return a JSON object with these fields:

```json
{
  "assay":             "IHC / IF / FACS / smFISH / RNA-seq / review / unknown",
  "cell_type":         "exact cell type or subtype name from text",
  "tissue":            "tissue or organ described",
  "localization":      "nuclear / cytoplasmic / membrane / extracellular / N/A",
  "evidence_level":    "direct / indirect / no evidence",
  "expression_status": "expressed / not expressed / silenced / unknown",
  "confidence":        "high / medium / low",
  "key_sentence":      "most relevant sentence (≤40 words)"
}
```

**Key rules in the extraction prompt:**

- **Rule 3 (Disease context — relaxed in Fix 3):** Multi-cancer/pan-cancer papers where the target disease is one of several studied cancers are now accepted. Only papers with zero mention of the target disease are rejected. Previously this was strict: any paper studying a different disease was rejected, which caused SLC5A8 evidence in colorectal+breast multi-cancer papers to be lost.

- **Rule 4 (Cell type lineage):** The LLM receives the `cell_subtypes` list and is instructed to accept evidence from any subtype, not just the exact broad term. If evidence is found in a subtype (e.g. "BDCA3+ dendritic cells"), the `cell_type` field should contain the exact subtype name from the paper.

- **Rule 7 (Absent/silenced expression — Fix 2):** If a Tier-1 protein assay (IHC/IF/FACS/Flow) shows ABSENCE, LOSS, SILENCING, or DOWNREGULATION of the gene, this is valid Tier-1 evidence. `expression_status` should be set to `"not expressed"` or `"silenced"` and `evidence_level` should be `"direct"`. RT-PCR, methylation-PCR, bisulfite sequencing are explicitly listed as supplementary-only (not eligible for Tier-1 status).

**JSON Parsing:** `_extract_json_object()` tries direct `json.loads()` first, then falls back to bracket-depth scanning to extract the JSON object even if the LLM adds surrounding prose or markdown fences.

**Parallelism:** All per-PMID extractions run concurrently via `ThreadPoolExecutor(max_workers=5)`.

#### Extraction Output

Each record returned by Agent 3:

```python
{
    "gene":            "SLC5A8",
    "cell_type_query": "cancer epithelial",
    "disease":         "breast cancer",
    "pmid":            "12345678",
    "source":          "PubMed",
    "rag_chunks_used": 4,
    # LLM-extracted:
    "assay":             "IHC",
    "cell_type":         "breast cancer cells",
    "tissue":            "breast tumor",
    "localization":      "membrane",
    "evidence_level":    "direct",
    "expression_status": "not expressed",
    "confidence":        "high",
    "key_sentence":      "IHC revealed loss of SLC5A8 protein in breast carcinoma tissue",
}
```

---

### Agent 4 — Validation (`validation_agent.py`)

**Purpose:** Apply rule-based biological validation to each extracted record. Adds `ManualReviewStatus` field.

#### Validation Statuses

| Status | Condition |
|---|---|
| `PASS` | Protein-level assay (IHC/IF/FACS/smFISH) confirms expression AND not in a negated context |
| `NOT_EXPRESSED` | Protein-level assay confirms absence/loss of expression (Tier-1 absence evidence) |
| `FAIL` | RNA-only evidence with no protein assay present |
| `NA` | evidence_level="no evidence", empty key_sentence, OR assay mentioned in negated context |

#### Decision Order (Important)

The logic runs in this strict order to prevent one check overriding another:

1. **Fix B guard (first):** If `evidence_level` contains "no evidence" OR `key_sentence` is empty → immediately `NA`. This prevents a record with `assay="IHC"` but `evidence_level="no evidence"` from getting a false PASS.

2. **Fix 2 — NOT_EXPRESSED check:** If `expression_status` (from LLM) is `"not expressed"` or `"silenced"`, AND a Tier-1 protein assay is present → `NOT_EXPRESSED`.

3. **Keyword fallback for NOT_EXPRESSED:** If `_is_tier1_absence()` returns True (detects absence keywords + protein assay in full text) → `NOT_EXPRESSED`. This catches cases where the LLM set `expression_status="unknown"` but the key sentence clearly describes loss/silencing.

4. **FAIL check:** If RNA keywords present AND no protein assay → `FAIL`.

5. **PASS check:** If protein assay present AND `_is_negated(key_sentence)` is False → `PASS`. If negated → `NA` with `validation_note="Negated assay evidence"`.

6. **Default:** `NA`.

#### Negation Detection (Fix A — Word Boundaries)

Two patterns catch negated assay contexts:

```python
# Pattern A: negation BEFORE assay keyword
# "not confirmed by IHC", "no FACS staining", "absent immunofluorescence"
_NEGATION_BEFORE = re.compile(
    r"(\bnot\b|\bno\b|\bnever\b|\babsent\b|\bundetected\b|\bnegative\b|"
    r"\bnot\s+confirmed\b|\bnot\s+expressed\b|\bnot\s+detected\b|"
    r"\bnot\s+found\b|\black\s+of\b|\bfailed\s+to\b|\bcould\s+not\b)"
    r".{0,50}"
    r"(ihc|immunohistochemistry|immunofluorescence|\bifs?\b|facs|"
    r"flow\s*cytometry|smfish|western\s*blot)",
    re.IGNORECASE | re.DOTALL,
)

# Pattern B: assay keyword BEFORE negation
# "IHC staining was negative", "FACS has not been confirmed"
_NEGATION_AFTER = re.compile(
    r"(ihc|immunohistochemistry|immunofluorescence|\bifs?\b|facs|"
    r"flow\s*cytometry|smfish|western\s*blot)"
    r".{0,50}"
    r"(\bnot\b|\bno\b|\bnever\b|\babsent\b|\bundetected\b|\bnegative\b|"
    r"\bnot\s+confirmed\b|\bnot\s+expressed\b|\bnot\s+detected\b|"
    r"\bnot\s+found\b)",
    re.IGNORECASE | re.DOTALL,
)
```

**Why `\b` word boundaries are critical:** Without them, `"no"` matches inside `"normal"`. A sentence like *"FAP expression was observed in normal fibroblast tissue by immunofluorescence"* would falsely trigger negation because `"no"` matches the start of `"normal"`. With `\b`, the pattern only matches standalone `"no"`.

#### Absence Keyword Detection (Fix 2)

```python
_ABSENCE_KW = re.compile(
    r"(\bloss\s+of\s+expression\b|\bexpression\s+loss\b|\bsilenced\b|\bsilencing\b|"
    r"\bepigenetic\s+silenc\b|\bdownregulated?\b|\bdownregulation\b|\bsuppressed?\b|"
    r"\bnot\s+expressed\b|\bno\s+expression\b|\babsent\b|\babsence\s+of\b|"
    r"\blost\s+expression\b|\black\s+of\s+expression\b|\bexpression\s+was\s+lost\b|"
    r"\bexpression\s+is\s+lost\b|\bnegative\s+for\b|\bundetectable\b|\bundetected\b)",
    re.IGNORECASE,
)
```

`_is_tier1_absence()` requires all three conditions:
1. A Tier-1 protein assay present in `assay` or `full_text`
2. An absence keyword present in `full_text`
3. NOT solely supplementary assay (RT-PCR / methylation-PCR with no protein assay)

RT-PCR-only absence still routes to FAIL, not NOT_EXPRESSED.

---

### Agent 5 — Scoring (`scoring_agent.py`)

**Purpose:** Assign an evidence tier label and numeric score (0.0–1.0) to each record.

#### Scoring Logic (in order of precedence)

1. **No evidence** (`evidence_level` contains "no evidence") → `Evidence = "No evidence"`, `score = 0.0`

2. **Tertiary-database** (source is HPA / CellMarker / UniProt) → `Evidence = "Tertiary-database"`, `score = 0.3`

3. **Primary-not-expressed** (Fix 2 — `ManualReviewStatus == "NOT_EXPRESSED"` OR `expression_status` in ("not expressed", "silenced")) AND Tier-1 protein assay present:
   - `base_score = 0.9` (slightly below 1.0 — absence is strong but below confirmed positive expression)
   - Penalty: `confidence == "low"` → `-0.15`
   - Penalty: `confidence == "medium"` → `-0.05`
   - Floor: 0.5
   - `Evidence = "Primary-not-expressed"`

4. **Primary-experimental** (Tier-1 protein assay present — IHC/IF/FACS/smFISH):
   - `base_score = 1.0`
   - Penalty: RNA keywords also present → `-0.15` (mixed evidence)
   - Penalty: `confidence == "low"` → `-0.15`
   - Penalty: `confidence == "medium"` → `-0.05`
   - Floor: 0.5
   - `Evidence = "Primary-experimental"`

5. **Secondary-review** (review/meta-analysis/clinical/cohort/patient keywords, or `evidence_level == "indirect"`):
   - `base_score = 0.6`
   - Penalty: `confidence == "low"` → `-0.1`
   - Floor: 0.3
   - `Evidence = "Secondary-review"`

6. **RNA-only** (RNA-seq/transcript/mRNA/qPCR keywords, no protein assay) → `score = 0.2`

7. **Fallback** → `score = 0.1`

Records are sorted by `evidence_score` descending after scoring.

---

### Agent 6 — Output (`output_agent.py`)

Converts the scored list to a Pandas DataFrame with:
- Column rename: internal keys → display names
- Column reorder: fixed `_COLUMN_ORDER` list
- Missing columns filled with `""` before rename to avoid KeyError
- CSV save if `out_path` provided
- Sorted by `evidence_score` descending

**Column rename map:**

| Internal Key | Display Name |
|---|---|
| `gene` | Marker Gene |
| `cell_type_query` | Cell Type (Query) |
| `disease` | Disease Context |
| `pmid` | Reference (PMID) |
| `source` | Data Source |
| `assay` | Assay |
| `cell_type` | Detected Cell Type |
| `tissue` | Tissue Specificity |
| `localization` | Localization |
| `evidence_level` | Evidence Level |
| `expression_status` | Expression Status *(Fix 2 — new)* |
| `confidence` | LLM Confidence |
| `key_sentence` | Key Evidence Sentence |
| `rag_chunks_used` | RAG Chunks Used |
| `Evidence` | Evidence Tier |
| `evidence_score` | Evidence Score |
| `ManualReviewStatus` | Validation Status |

---

## 7. Utility Modules

### RAG Store (`rag_store.py`)

**`BiomarkerRAGStore`** — in-memory hybrid FAISS + BM25 store with RRF fusion. One store instance per pipeline run; cleared and rebuilt for each new query.

#### Embedding Model

Primary: `all-MiniLM-L6-v2` (SentenceTransformer, 384-dimensional embeddings)

Fallback if `sentence-transformers` or `torch` not installed: `_HashEmbedder` — a simple hash-based 128-dim vector. Much lower quality but keeps the BM25 side functional. Always install torch + sentence-transformers for production.

#### Chunking Parameters

```python
CHUNK_SIZE    = 512   # characters
CHUNK_OVERLAP = 80    # character overlap between adjacent chunks
separators    = ["\n\n", "\n", ". ", " ", ""]
```

Chunks shorter than 30 characters are discarded.

#### Section Labelling

Each chunk is labelled with its paper section (Abstract, Introduction, Methods, Results, Discussion, Conclusion, Body). High-value sections: Results, Methods, Discussion. These are given priority in `rag_context_builder.py`.

#### FAISS Index

```python
index = faiss.IndexFlatIP(d)  # Inner Product on L2-normalised vectors = cosine similarity
index.add(embeddings)
```

Vectors are L2-normalised before indexing so Inner Product equals cosine similarity.

#### BM25 Index

```python
from rank_bm25 import BM25Okapi
tokenised = [chunk.text.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenised)
```

#### RRF Fusion

```python
RRF_K = 60  # standard RRF constant

for idx in all_candidate_indices:
    score = 0.0
    if idx in dense_rank:   score += 1.0 / (RRF_K + dense_rank[idx] + 1)
    if idx in sparse_rank:  score += 1.0 / (RRF_K + sparse_rank[idx] + 1)
    rrf_scores[idx] = score
```

No cross-encoder reranking — RRF is the final fusion step. This was a deliberate choice to keep latency low while still benefiting from both dense (semantic) and sparse (keyword exact-match) retrieval.

**Why RRF specifically:** RRF does not require calibrated scores from both retrievers. Dense cosine scores and BM25 scores are on incompatible scales; using raw scores from both would be misleading. Rank-based fusion with RRF sidesteps this by only using the rank position, not the raw score values.

#### Module-Level Singleton

```python
_active_store: Optional[BiomarkerRAGStore] = None

def get_store() -> BiomarkerRAGStore: ...
def reset_store() -> None: ...
```

`reset_store()` is called at the start of each `extraction_agent()` call to clear any state from a previous run.

---

### RAG Context Builder (`rag_context_builder.py`)

`build_rag_context()` fires multiple targeted queries against the RAG store and assembles a formatted numbered context string for the LLM.

**Query templates used:**

- Section-targeted: Methods (antibody protocol), Results (expression level, statistical), Disease-specific Results
- General: IHC, IF, FACS, protein expression, smFISH
- Disease-aware (added when disease is provided): disease + expression, disease + IHC clinical, disease + protein biomarker

**Ranking of chunks:**

```python
all_chunks.sort(key=lambda c: (
    _SECTION_PRIORITY.get(c["section"], 0),  # Results=4 > Methods=3 > Discussion=2 > rest
    c["rrf_score"],
), reverse=True)
```

**PMID diversity cap:** max 4 chunks per PMID. Prevents one paper with many short sections from completely dominating the context window.

**Output format:**
```
=== RETRIEVED EVIDENCE CONTEXT ===
Gene: SOX18  |  Cell Type: Endothelial cells  |  Disease: breast cancer

[SECTION 1]
Source: PubMed | PMID: 12345678 | Section: Results
IHC staining confirmed SOX18 protein expression in CD31-positive endothelial cells...

[SECTION 2]
...
```

---

### LLM Client (`llm_client.py`)

`call_llm(prompt, model, max_tokens, system)` routes to Anthropic first, falls back to Groq.

- **Primary:** `claude-sonnet-4-6` (Anthropic) — all agents use this
- **Fallback:** `llama3-70b-8192` (Groq) — activated only if Anthropic key is missing or the API call throws an exception
- Raises `RuntimeError` if both are unavailable

Token budgets per agent call:
- Gene synonym expansion: `max_tokens=400`
- Query generation: `max_tokens=700`
- Evidence extraction per PMID: `max_tokens=512`

Context passed to extraction is capped at 3500 characters to stay within the token budget.

---

### Full Text Fetcher (`full_text_fetcher.py`)

`fetch_full_text_or_abstract()` tries in priority order:

1. **PMC** via Biopython Entrez elink + efetch: parses full XML, strips tags with regex, splits into sections
2. **Europe PMC** REST `/fullTextXML`: same XML processing, 6s timeout
3. **Unpaywall**: resolves DOI → open-access landing page URL → scrapes HTML via BeautifulSoup (removes script/style/nav/footer/header)
4. **Fallback**: abstract text only

All sources respect copyright — no paywall bypassing. Access type is recorded in `access_type` field ("full_text" vs "abstract_only") and `ft_source` field in paper dict.

Section detection splits full text on uppercase headings and labels each block via `_SECTION_PATTERNS` regex.

---

## 8. Evidence Hierarchy and Rules

### Tier 1 — Primary Experimental (Score: 0.8–1.0)

Valid assays:
- IHC (Immunohistochemistry)
- IF / IFS (Immunofluorescence staining)
- FACS / Flow cytometry
- smFISH (single-molecule FISH) — **only valid assay for non-coding RNA / lncRNA**

Both positive (expressed) and negative (not expressed) results from these assays qualify as Tier 1. Confirmed absence via IHC/FACS is `Primary-not-expressed` at score 0.85–0.9.

### Tier 2 — Secondary Review/Clinical (Score: 0.5–0.7)

- Review papers, meta-analyses, systematic reviews
- Clinical/patient tissue studies without direct protein-level validation
- Indirect evidence (gene signature studies, cohort analysis)

### Tier 3 — Tertiary Database (Score: 0.2–0.4)

- Human Protein Atlas records
- CellMarker database entries
- UniProt function/tissue annotations

### Not Eligible for Tier 1 (Supplementary Only)

- RNA-seq / transcriptomics
- RT-PCR / qPCR / mRNA quantification
- Methylation-PCR / bisulfite sequencing (epigenetic silencing only — not protein evidence)
- Microarray expression profiling
- In situ hybridisation (ISH) — note: smFISH is accepted for lncRNA only

---

## 9. Disease Context Enforcement

### Rules (enforced across all agents)

**Disease provided:**
- Agent 1 (Planner): queries include disease term in all query types
- Agent 2 (Retrieval): papers with disease keyword in abstract sorted to front of the cap-20 list
- Agent 3 (Extraction Rule 3 — relaxed Fix 3):
  - PREFERRED: papers specifically about the target disease
  - ALSO ACCEPT: multi-cancer / pan-cancer papers where target disease is one of several
  - REJECT ONLY: papers with zero mention of the target disease anywhere in text
  - REJECT: papers about purely normal/healthy tissue when disease is specified

**No disease provided:**
- All queries use normal tissue context
- Extraction rejects cancer/disease-specific papers

### Why Rule 3 Was Relaxed (Fix 3)

Original strict rule: *"If this paper studies a DIFFERENT disease or cancer type → return no-evidence JSON."*

Problem discovered during SLC5A8 testing: Many important SLC5A8 papers study silencing across multiple cancer types simultaneously (colorectal cancer, thyroid cancer, breast cancer in the same paper). These were previously rejected because the paper also discussed colorectal cancer, which triggered the strict filter even when breast cancer evidence was present in the same abstract.

Fix: The rule now accepts multi-cancer papers. The extraction LLM is instructed to extract only the portion of evidence relevant to the target disease.

---

## 10. Cell Lineage Hierarchy Map

See the complete `_CELL_LINEAGE_MAP` in the [Agent 1 section](#step-1b-cell-type-lineage-expansion).

The map is keyed by **lowercase** cell type strings. Matching is case-insensitive via `.lower().strip()`.

If the user input does not match any key (e.g. "endothelial cells" has no subtypes listed as it is already specific), the function returns `[]` and the planner uses only the exact term provided.

**Why the map is intentionally not exhaustive:** Listing every possible subtype would generate dozens of queries and overwhelm the retrieval budget. The map covers the most biologically meaningful expansions where papers commonly use subtypes instead of broad terms.

---

## 11. Validation Logic — Internal Detail

See [Agent 4 section](#agent-4--validation-validation_agentpy) for the complete decision flow.

### Assay Keyword Sets

```python
_PROTEIN_ASSAYS = {
    "ihc", "immunohistochemistry",
    "if", "ifs", "immunofluorescence",
    "flow cytometry", "facs",
    "smfish", "sm-fish", "single-molecule fish",
    "protein expression", "western blot", "elisa",
    "co-localiz", "colocaliz",
}

_RNA_ASSAYS = {
    "rna-seq", "rnaseq", "rna seq", "transcriptomics",
    "gene expression", "bulk rna", "scrna", "single-cell rna",
    "mrna", "transcript", "qpcr", "rt-pcr", "pcr",
    "microarray", "in situ hybridiz",
}

_TIER1_PROTEIN_ASSAYS = {
    "ihc", "immunohistochemistry",
    "if", "ifs", "immunofluorescence",
    "flow cytometry", "facs",
    "smfish", "sm-fish", "single-molecule fish",
}

_SUPPLEMENTARY_ONLY_KW = {
    "rt-pcr", "methylation-pcr", "bisulfite", "qpcr", "mrna", "rna-seq",
    "rnaseq", "transcript", "microarray",
}
```

Note: `western blot` and `elisa` are in `_PROTEIN_ASSAYS` (used for PASS detection) but NOT in `_TIER1_PROTEIN_ASSAYS` (not used for NOT_EXPRESSED Tier-1 scoring). This is deliberate — Western blot and ELISA are valid protein assays but are not the standard histology/cytometry assays for cell-type-specific expression localisation.

---

## 12. Scoring Logic — Internal Detail

See [Agent 5 section](#agent-5--scoring-scoring_agentpy) for the full decision tree.

### Score Summary Table

| Condition | Evidence Label | Base Score | Penalties | Floor |
|---|---|---|---|---|
| evidence_level = "no evidence" | No evidence | 0.0 | — | — |
| source is database (HPA/CellMarker/UniProt) | Tertiary-database | 0.3 | — | — |
| NOT_EXPRESSED + Tier-1 protein assay | Primary-not-expressed | 0.9 | low: -0.15, med: -0.05 | 0.5 |
| Tier-1 protein assay (PASS) | Primary-experimental | 1.0 | RNA present: -0.15, low: -0.15, med: -0.05 | 0.5 |
| review/clinical/indirect | Secondary-review | 0.6 | low: -0.1 | 0.3 |
| RNA-only | RNA-only | 0.2 | — | — |
| None of above | Unknown | 0.1 | — | — |

---

## 13. Output Column Schema

Final DataFrame column order (left to right in the UI table):

```
Marker Gene → Cell Type (Query) → Disease Context → Detected Cell Type
→ Assay → Expression Status → Tissue Specificity → Localization
→ Key Evidence Sentence → Evidence Level → LLM Confidence
→ Evidence Tier → Evidence Score → Validation Status
→ Reference (PMID) → Data Source → RAG Chunks Used
```

`Expression Status` was added in Fix 2. It surfaces the `expression_status` field extracted by the LLM: `expressed / not expressed / silenced / unknown`. This is the key field for tumor suppressor gene queries where absence is the biologically meaningful finding.

---

## 14. Backend API (FastAPI)

`backend.py` — runs on port 8001.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check, returns active job count |
| `POST` | `/pipeline/run` | Submit job; returns `{job_id, status: "running"}` |
| `GET` | `/pipeline/status/{job_id}` | Poll job — returns full job state dict |
| `DELETE` | `/pipeline/job/{job_id}` | Clean up completed job |
| `GET` | `/pipeline/stages` | Returns ordered list of stage labels for progress bar |

### Job State Dict

```python
{
    "status":  "running" | "done" | "error",
    "stage":   "Agent 3 — Extraction (FAISS+BM25 RAG + LLM)",  # current stage label
    "pct":     50,           # 0–100 (shown in progress bar)
    "records": [...],        # populated when done
    "n":       12,           # record count
    "elapsed": 143.2,        # seconds
    "gene":    "SLC5A8",
    "cell":    "cancer epithelial",
    "disease": "breast cancer",
    "warning": "...",        # optional, e.g. "No papers retrieved"
    "error":   "...",        # only on status=error
    "trace":   "...",        # traceback, only on status=error
}
```

### Stage Labels and Progress Percentages

```python
STAGES = [
    "Queued",           # 0%
    "Agent 1 — Planner (synonyms + queries)",          # 10%
    "Agent 2 — Retrieval (PubMed · EPMC · SS · databases)",  # 25%
    "Agent 3 — Extraction (FAISS+BM25 RAG + LLM)",    # 50%
    "Agent 4 — Validation (PASS / FAIL / NA)",         # 70%
    "Agent 5 — Scoring (evidence tiers)",              # 85%
    "Agent 6 — Output (structuring table)",            # 95%
    "Done",             # 100%
]
```

### Threading Model

- `ThreadPoolExecutor(max_workers=4)` in the backend — up to 4 simultaneous pipeline jobs
- Each job runs all 6 agents sequentially in its own thread
- Within Agent 2 (retrieval), `ThreadPoolExecutor(max_workers=10)` parallelises across sources
- Within Agent 3 (extraction), `ThreadPoolExecutor(max_workers=5)` parallelises across PMIDs
- Job state stored in `_jobs` dict (thread-safe via `threading.Lock()`)

---

## 15. MCP Server

`mcp_server/server.py` — exposes the pipeline as an MCP tool via FastMCP.

### Starting the Server

```bash
python mcp_server/server.py
```

### Claude Desktop / MCP Client Config

```json
{
  "mcpServers": {
    "biomarker": {
      "command": "python",
      "args": ["/absolute/path/to/biomarker-simplified-workflow/mcp_server/server.py"]
    }
  }
}
```

### Tool: `run_pipeline`

```python
run_pipeline(gene: str, cell: str, disease: str = "") -> dict
```

Calls `pipeline.py:run_pipeline()` which runs the full 6-agent pipeline synchronously and returns a JSON-serialisable dict with `status`, `gene`, `cell`, `disease`, `n_records`, and `records` list.

### Resource: `biomarker://info`

Returns server capabilities as JSON (name, version, tools, sources, RAG method).

---

## 16. History of Bugs and Fixes

This section documents every bug encountered during real-world testing, with root cause and fix applied.

### Bug 1 — False PASS on negated evidence (PMID 39643914, 41969468)

**Problem:** Records with `assay="IHC"` but `evidence_level="no evidence"` were getting `ManualReviewStatus="PASS"`.

**Root cause:** The validation agent checked assay keywords before checking evidence_level. Since "ihc" appeared in the assay field, it matched `_PROTEIN_ASSAYS` and was labelled PASS regardless of the LLM saying "no evidence".

**Fix B:** Added a guard at the very top of the validation loop: if `evidence_level` contains "no evidence" OR `key_sentence` is empty → immediately assign `NA` and `continue`. This short-circuits all subsequent assay keyword checks.

---

### Bug 2 — False NA on positive IF evidence (PMID 41800236)

**Problem:** A record with clear immunofluorescence evidence was labelled `NA` instead of `PASS`.

**Root cause:** The negation pattern `r"(not|no|never|...)"` had no word boundaries. The text contained *"normal fibroblast tissue ... immunofluorescence"*. The substring `"no"` matched the start of `"normal"`, triggering `_is_negated()` to return `True`, which forced the status to `NA`.

**Fix A:** Added `\b` word boundaries to all negation keywords: `\bnot\b`, `\bno\b`, `\bnever\b`, etc. The word `"normal"` no longer matches `\bno\b`.

---

### Bug 3 — Gene alias confusion: KLRC2 vs KLRC1 (NKG2C vs NKG2A)

**Problem:** The gene synonym expansion for `KLRC2` included `NKG2A` (which belongs to `KLRC1`, a different gene in the same family). This caused retrieval of papers about NKG2A-expressing cells and inflation of false positives.

**Root cause:** The synonym expansion LLM prompt had no disambiguation instruction, so the model listed all NKG2 family members as aliases for KLRC2.

**Fix:** Added explicit disambiguation instruction to both the synonym expansion prompt (`expand_gene_synonyms`) and `planner_prompt.txt`:
> *"Only include aliases that refer to EXACTLY this gene. Do NOT include aliases for related paralogs or family members. Example: if gene is KLRC2 (encodes NKG2C), do NOT include NKG2A or KLRC1."*

---

### Bug 4 — CAF abbreviation not expanded in queries

**Problem:** Queries for `CAF` were not generating results. Papers use `"cancer-associated fibroblast"` as the full term, not the abbreviation.

**Root cause:** The planner_prompt.txt had no abbreviation expansion instruction.

**Fix D:** Added an explicit abbreviation expansion table to `planner_prompt.txt`:
```
CAF → "cancer-associated fibroblast" or "carcinoma-associated fibroblast"
TIL → "tumor-infiltrating lymphocyte"
NK  → "natural killer cell"
DC  → "dendritic cell"
TAM → "tumor-associated macrophage"
```
The LLM is instructed to always include both the abbreviation and full form in queries.

---

### Bug 5 — EuropePMC date bias: 2025/2026 papers ranked above established literature

**Problem:** EuropePMC default sort is `date desc` (newest first). This flooded retrieval with brand-new preprints and drug-target papers from 2025–2026 that lacked validated protein-level evidence, while established 2019–2023 papers with strong IHC data were buried.

**Root cause:** EuropePMC default sort behaviour.

**Fix C:** Added `"sort": "cited desc"` to EuropePMC request parameters. Citation count sorting promotes established, well-cited papers over brand-new publications.

---

### Bug 6 — CLEC9A + myeloid cells: missing BDCA3+ dendritic cell papers

**Problem:** Pipeline returned sparse results for CLEC9A + myeloid cells because papers consistently say "BDCA3+ dendritic cells expressed CLEC9A" — never "myeloid cells expressed CLEC9A."

**Root cause:** No mechanism to expand broad cell types to specific subtypes at query time OR extraction time.

**Fix:** Two-part fix:
1. Added `_CELL_LINEAGE_MAP` to `planner_agent.py` with `expand_cell_subtypes()`. The planner now generates subtype-specific queries (e.g. "CLEC9A BDCA3+ dendritic cells IHC breast cancer").
2. Added Rule 4 to `_EXTRACT_PROMPT` in `extraction_agent.py`: LLM is given the `cell_subtypes` list and instructed to accept evidence from any subtype, filling `cell_type` with the exact subtype name from the paper.

---

### Bug 7 — SLC5A8 + cancer epithelial: all records "No evidence"

**Problem:** Pipeline returned all `NA` records for SLC5A8 in cancer epithelial cells. This is a well-known tumor suppressor gene with many IHC papers showing loss of expression in multiple cancers.

**Root cause (three contributing factors):**

a) `"cancer epithelial"` was not in `_CELL_LINEAGE_MAP` → no subtype expansion → queries too generic.

b) Extraction prompt had no concept of absent/silenced expression as valid evidence. When the LLM saw "SLC5A8 expression was lost in breast tumor cells by IHC," it returned `evidence_level="no evidence"` because the gene was absent — there was nothing "expressed" to report.

c) Many SLC5A8 papers study multiple cancer types simultaneously (colorectal + breast + thyroid in one paper). The strict Rule 3 in the extraction prompt rejected these papers because they also studied non-breast-cancer diseases.

**Fix 1:** Added `"cancer epithelial"`, `"cancer epithelial cells"`, `"tumor cells"`, `"cancer cells"`, `"carcinoma cells"` to `_CELL_LINEAGE_MAP` with disease-specific subtype terms.

**Fix 2:** Added `expression_status` field to extraction JSON schema. Added Rule 7 to extraction prompt explicitly instructing the LLM to treat IHC/IF/FACS/Flow showing ABSENCE as valid `direct` evidence with `expression_status = "not expressed"`. Added `NOT_EXPRESSED` validation status. Added `Primary-not-expressed` scoring tier (score 0.85–0.9). Added `Expression Status` output column.

**Fix 3:** Relaxed Rule 3 in the extraction prompt to accept multi-cancer papers. Changed from strict rejection to: "REJECT ONLY papers with zero mention of the target disease."

---

## 17. Speed Optimisations Applied

The original pipeline ran at ~328 seconds for a typical query. After optimisations:

| Optimisation | Label | Saving |
|---|---|---|
| Skip full-text fetch if gene not in abstract | S1 | ~4–8s per irrelevant paper |
| Cap literature queries at 5 (was 8) | S2 | Fewer redundant query/source combinations |
| Cap output at 20 papers (was unlimited) | S3 | Extraction LLM only runs on top 20 |
| Concurrent retrieval via ThreadPoolExecutor (10 workers) | — | All sources fire simultaneously |
| Concurrent extraction via ThreadPoolExecutor (5 workers) | — | All PMIDs extracted in parallel |
| EuropePMC `sort=cited desc` to cut noise | S4 | Better quality papers → fewer wasted LLM calls |
| Semantic Scholar: retry only on HTTP 429, no unconditional sleep | — | Eliminates unnecessary wait time |

Typical run time post-optimisation: **120–180 seconds** for a full query (varies with network latency and number of papers retrieved).

---

## 18. Known Limitations and Edge Cases

### Limitation 1 — smFISH for lncRNA only

The pipeline accepts smFISH as Tier-1 evidence. The extraction prompt does not currently enforce that smFISH is *only* valid for non-coding RNA. If a paper uses smFISH for a protein-coding gene, it will still score as Tier-1. This is conservative — smFISH is a valid single-molecule method.

### Limitation 2 — Western Blot as non-localisation evidence

Western blot is included in `_PROTEIN_ASSAYS` (enabling PASS status) but does not prove cell-type-specific expression from tissue sections. WB shows protein presence in a lysate — it doesn't demonstrate which cell type expresses the protein in situ. No downgrade mechanism currently exists for WB-only records.

### Limitation 3 — Gene names that are common words

Short or ambiguous gene symbols (e.g. `"SET"`, `"COP"`, `"GAS"`) can generate false matches in text search. The gene disambiguation prompt helps but does not fully solve this for retrieval.

### Limitation 4 — Abstract-only papers

~30–50% of retrieved papers will be abstract-only (full text behind paywall, not in PMC). Abstracts are typically 200–300 words — evidence in the full Results section is not accessible. The `access_type` field in the output records this.

### Limitation 5 — CellMarker HTML scraping

CellMarker's website structure can change. If the scraper returns empty results, a code update to the BeautifulSoup selector may be needed.

### Limitation 6 — No contradiction detection between papers

The pipeline does not detect when two papers contradict each other (Paper A: expressed by IHC; Paper B: not detected by IHC). Both records will appear in the table. The user must cross-reference by reviewing the Key Evidence Sentences.

### Limitation 7 — Expression_status from LLM can be "unknown" for valid absence records

In some cases the LLM sets `expression_status="unknown"` even when the key sentence describes loss of expression. The keyword fallback in `_is_tier1_absence()` catches many of these, but the LLM's confidence about absence is lower than for positive expression. Workaround: check both `Expression Status` column and `Key Evidence Sentence` manually.

---

## 19. Testing

```bash
python -m pytest tests/test_pipeline.py -v

# or run directly (no pytest needed):
python tests/test_pipeline.py
```

**What is tested (no LLM calls — fully offline):**

| Test | What it checks |
|---|---|
| `test_input_handler_dict` | Dict input parsed correctly |
| `test_input_handler_string` | Free-text string "Gene + Cell + Disease" parsed |
| `test_input_handler_no_disease` | Missing disease → empty string (not None) |
| `test_rag_store_ingest` | Papers → chunks > 0, size matches |
| `test_rag_store_retrieve` | Retrieval returns results with `rrf_score` and `chunk_text` |
| `test_rag_store_clear` | `store.size == 0` after clear |
| `test_rag_context_builder` | Context contains `[SECTION` headers and gene name |
| `test_validation_agent` | IHC→PASS, RNA-seq→FAIL, no evidence→NA, FACS→PASS |
| `test_scoring_agent` | Primary ≥0.8, Tertiary ≤0.4 |
| `test_output_agent` | DataFrame has `Marker Gene`, `Evidence Tier`, `Validation Status` columns |

Tests do NOT cover: Planner (LLM), Retrieval (network), Extraction (LLM + network). Integration tests for those require live API keys and network access.

---

## 20. Design Decisions and Rationale

### Why FAISS instead of ChromaDB?

- No external server required — FAISS runs in-process, in-memory
- Full control over index type (Flat vs IVF vs HNSW) and distance metric
- No persistence overhead for a per-query ephemeral store
- ChromaDB adds operational complexity (server management, collection lifecycle) for no benefit in this single-query-per-run use case

### Why RRF instead of a cross-encoder?

- Cross-encoder reranking adds significant latency (~1–2s per 20 candidates)
- RRF is parameter-free and robust — requires no training or fine-tuning
- In biomedical retrieval, BM25 is particularly strong for exact gene symbol + assay keyword matching, which is exactly the vocabulary the pipeline needs
- RRF fuses the two rank lists without requiring calibrated scores from either

### Why one record per PMID (not per chunk)?

The output table is intended to be readable by a researcher. Showing 30+ rows from 10 papers (3 chunks per paper) creates unreadable noise. One consolidated record per paper keeps the table at 3–10 rows — comparable to what a manual literature review would produce.

### Why is the disease filter in Agent 3 (extraction) not in Agent 2 (retrieval)?

Retrieval is imprecise — papers can mention a disease in their abstract without the full text containing relevant evidence (and vice versa). Filtering at retrieval time would miss papers where breast cancer evidence appears in the full text but not the abstract. The LLM at extraction time reads the actual content and can make a more accurate inclusion/exclusion decision.

### Why 6 separate agents instead of one LLM call?

Each stage requires very different logic — keyword generation, HTTP requests, chunking, per-paper JSON extraction, rule-based classification, and scoring. A single monolithic LLM call would: (a) hallucinate retrieval instead of actually calling APIs, (b) exceed context window for large paper sets, (c) make debugging impossible. Separating stages enables targeted testing, optimisation of each step independently, and clear attribution of errors to specific agents.

---

*Last updated: April 2026. Reflects all fixes from iterations 1–7 of the simplified pipeline.*
