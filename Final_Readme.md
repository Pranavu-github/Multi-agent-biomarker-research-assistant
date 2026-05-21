# 🧬 Biomarker Research Assistant

An AI-powered 6-agent pipeline that identifies and validates biomarker expression evidence for a given gene, cell type, and disease context. Retrieves literature and biological database records, applies hybrid RAG-based extraction, and returns structured, evidence-tiered results through a Streamlit web interface.

---

## What It Does

Given a **gene**, **cell type**, and optional **disease**, the pipeline:

1. Expands gene synonyms and cell subtypes, generates optimised queries
2. Retrieves evidence from PubMed, Europe PMC, Semantic Scholar, HPA, CellMarker, UniProt
3. Extracts structured evidence using hybrid RAG (MedCPT + FAISS + BM25)
4. Validates each record using biological rules (protein vs RNA, negation detection)
5. Scores and tiers evidence (Tier 1 → Tier 3)
6. Returns a ranked evidence table with PMID links

---

## Project Structure

```
biomarker-simplified-workflow/
│
├── agents/
│   ├── planner_agent.py        # Agent 1 — query planning, synonym + cell expansion
│   ├── retrieval_agent.py      # Agent 2 — multi-source parallel retrieval
│   ├── extraction_agent.py     # Agent 3 — RAG-powered LLM evidence extraction
│   ├── validation_agent.py     # Agent 4 — biological validation (PASS/FAIL/NA)
│   ├── scoring_agent.py        # Agent 5 — evidence tiering and scoring (0.0–1.0)
│   └── output_agent.py         # Agent 6 — structured DataFrame output
│
├── utils/
│   ├── rag_store.py            # Hybrid FAISS + BM25 RAG store (MedCPT, RRF fusion)
│   ├── rag_context_builder.py  # Context assembly for LLM prompting
│   ├── llm_client.py           # LLM API client (Anthropic / Groq fallback)
│   ├── full_text_fetcher.py    # PMC / Europe PMC full-text retrieval
│   └── input_handler.py        # Input normalisation utilities
│
├── prompts/
│   └── planner_prompt.txt      # Prompt template for query generation
│
├── mcp_server/
│   └── server.py               # MCP server (tool registration via FastMCP)
│
├── tests/
│   └── test_pipeline.py        # End-to-end test cases
│
├── app.py                      # Streamlit frontend
├── backend.py                  # FastAPI backend (async job queue)
├── Dockerfile.backend          # Docker image — FastAPI + MedCPT models
├── Dockerfile.frontend         # Docker image — Streamlit UI (lightweight)
├── docker-compose.yml          # Orchestrates backend + frontend containers
├── .dockerignore               # Excludes secrets and caches from Docker build
├── requirements.txt
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Requirements

- Python 3.10+
- API keys (see Setup below)
- Docker Desktop (for containerised deployment)

---

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/biomarker-simplified-workflow.git
cd biomarker-simplified-workflow
```

**2. Create your `.env` file:**
```bash
cp .env.example .env
```

Fill in your keys:

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ Yes | LLM calls (Claude) |
| `ENTREZ_EMAIL` | ✅ Yes | PubMed / PMC access |
| `GROQ_API_KEY` | Optional | LLM fallback |
| `UNPAYWALL_EMAIL` | Optional | Full-text paper access |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Higher API rate limits |

> **Important:** No inline comments after values in `.env`. Comments must be on their own line.

---

## Running the Application

### Option A — Docker (Recommended)

Requires Docker Desktop to be running.

```bash
# First run (builds images + downloads MedCPT models ~900 MB, takes ~10 min)
docker compose up --build

# Subsequent runs (uses cached images, starts in seconds)
docker compose up
```

Open: **http://localhost:8501**

To stop:
```bash
docker compose down
```

---

### Option B — Local Development

Open two terminals in the project directory.

**Terminal 1 — Backend:**
```bash
uvicorn backend:app --host 127.0.0.1 --port 8001 --reload
```

**Terminal 2 — Frontend:**
```bash
streamlit run app.py
```

Open: **http://localhost:8501**

> **Note:** Never run Docker and local dev at the same time — they both use port 8001 and will conflict.

---

## How to Use

1. Enter a **Gene Symbol** — e.g. `SOX18`, `SLC5A8`, `CLEC9A`
2. Enter a **Cell Type** — e.g. `endothelial cells`, `myeloid cells`, `NK cells`
3. Enter a **Disease** (optional) — e.g. `breast cancer`, `NSCLC`  
   Leave blank for normal tissue context
4. Click **Run Pipeline**
5. Watch the live progress bar through all 6 agent stages
6. Results appear as a ranked, colour-coded evidence table

---

## Pipeline Agents

| Agent | Role |
|---|---|
| **1 — Planner** | Expands gene synonyms, cell subtypes, generates search queries |
| **2 — Retrieval** | Fetches papers from PubMed, Europe PMC, Semantic Scholar, HPA, CellMarker, UniProt |
| **3 — Extraction** | Hybrid RAG (MedCPT + FAISS + BM25 + RRF) + LLM evidence extraction |
| **4 — Validation** | Classifies each record: PASS / NOT_EXPRESSED / FAIL / NA |
| **5 — Scoring** | Assigns evidence tier and confidence score (0.0–1.0) |
| **6 — Output** | Structures results into ranked evidence table |

---

## Validation Status

| Status | Meaning |
|---|---|
| `PASS` | Protein-level assay confirms expression (IHC, IF, FACS, smFISH) |
| `NOT_EXPRESSED` | Protein-level assay confirms absence or loss (valid for tumor suppressors) |
| `FAIL` | RNA-only evidence — assay tier insufficient |
| `NA` | Ambiguous, no evidence, or negated assay context |

---

## Evidence Tiers

| Tier | Description | Score |
|---|---|---|
| Primary-experimental | IHC / IF / FACS confirms expression | 0.8–1.0 |
| Primary-not-expressed | IHC / IF / FACS confirms absence | 0.85–0.9 |
| Secondary-review | Review papers, clinical studies | 0.5–0.7 |
| RNA-only | RNA-seq, qPCR, transcriptomics only | 0.2 |
| Tertiary-database | HPA, CellMarker, UniProt records | 0.3 |
| No evidence | No relevant evidence found | 0.0 |

---

## Disease Context Rules

- **Disease provided** → all steps restricted to that disease context. Cancer-specific datasets used.
- **No disease provided** → normal tissue only. Cancer-specific evidence excluded.

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-6`) — Groq as fallback |
| Dense embeddings | MedCPT asymmetric dual-encoder (PubMed-trained, 768-dim) |
| Dense retrieval | FAISS (cosine similarity, `IndexFlatIP`) |
| Sparse retrieval | BM25Okapi (`rank-bm25`) with biomedical tokenisation |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Literature retrieval | Biopython Entrez, Europe PMC REST, Semantic Scholar API |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerisation | Docker + Docker Compose |
| MCP server | FastMCP |

---

## MedCPT Embedding Models

The RAG store uses **asymmetric dual-encoders** trained on 255M PubMed click-through logs:

- `ncbi/MedCPT-Query-Encoder` — encodes search queries
- `ncbi/MedCPT-Article-Encoder` — encodes document chunks
- `ncbi/MedCPT-Cross-Encoder` — optional reranker (flag-gated)

Models are downloaded automatically from HuggingFace (~440 MB each) on first use, or baked into the Docker image at build time for zero cold-start.

Fallback: if MedCPT is unavailable, `all-MiniLM-L6-v2` is used automatically.

---

## Running Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

Test cases:
- `SOX18` + `endothelial cells` (normal tissue)
- `SOX18` + `endothelial cells` + `breast cancer`
- `CLEC9A` + `myeloid cells` (lineage expansion)
- `SLC5A8` + `cancer epithelial` + `breast cancer` (tumor suppressor silencing)

---

## License

For research and evaluation use. Ensure compliance with the terms of service of all queried data sources: PubMed, Europe PMC, Human Protein Atlas, CellMarker, UniProt, Semantic Scholar.
