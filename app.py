"""
app.py — Streamlit frontend for the Biomarker Validation Pipeline.

Requires the FastAPI backend to be running first:
    Terminal 1:  uvicorn backend:app --host 127.0.0.1 --port 8001 --reload
    Terminal 2:  streamlit run app.py
"""

from __future__ import annotations

import sys
import time
import json
import os
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import requests
import streamlit as st
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
# BACKEND_URL env var is set by docker-compose so frontend finds backend service.
# Falls back to localhost for local development (non-Docker).
API_BASE = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

# Keep STAGES in sync with backend.py
STAGES = [
    "Agent 1 — Planner (synonyms + queries)",
    "Agent 2 — Retrieval (PubMed · EPMC · SS · databases)",
    "Agent 3 — Extraction (FAISS+BM25 RAG + LLM)",
    "Agent 4 — Validation (PASS / FAIL / NA)",
    "Agent 5 — Scoring (evidence tiers)",
    "Agent 6 — Output (structuring table)",
]

st.set_page_config(
    page_title="🧬 Biomarker Research Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 5) -> dict:
    resp = requests.get(f"{API_BASE}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: dict, timeout: int = 10) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def backend_is_up() -> bool:
    try:
        api_get("/health", timeout=2)
        return True
    except Exception:
        return False


# ── Styling helpers ───────────────────────────────────────────────────────────

def _colour_status(val: str) -> str:
    return {
        "PASS": "background-color: #d4edda; color: #155724",
        "FAIL": "background-color: #f8d7da; color: #721c24",
        "NA":   "background-color: #fff3cd; color: #856404",
    }.get(str(val).upper(), "")


def _colour_tier(val: str) -> str:
    return {
        "Primary-experimental": "background-color: #cce5ff; color: #004085",
        "Secondary-review":     "background-color: #e2d9f3; color: #432874",
        "Tertiary-database":    "background-color: #d1ecf1; color: #0c5460",
        "No evidence":          "background-color: #f5f5f5; color: #666",
    }.get(str(val), "")


def _style_df(df: pd.DataFrame):
    style = df.style
    if "Validation Status" in df.columns:
        style = style.applymap(_colour_status, subset=["Validation Status"])
    if "Evidence Tier" in df.columns:
        style = style.applymap(_colour_tier, subset=["Evidence Tier"])
    if "Evidence Score" in df.columns:
        style = style.background_gradient(
            subset=["Evidence Score"], cmap="RdYlGn", vmin=0, vmax=1
        )
    return style


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧬 Biomarker Research")
    st.markdown("---")

    # Backend status
    if backend_is_up():
        st.success("Backend ✅  running")
    else:
        st.error(
            "Backend ❌  not reachable\n\n"
            "**Docker:** ensure both containers are running:\n```\ndocker compose up\n```\n\n"
            "**Local dev:** start the backend first:\n```\nuvicorn backend:app --host 127.0.0.1 --port 8001 --reload\n```"
        )

    st.markdown("---")
    st.subheader("Query Parameters")
    gene    = st.text_input("Gene / Protein Symbol", value="SOX18",
                            help="e.g. SOX18, DARC, IFNGR1")
    cell    = st.text_input("Cell Type", value="Endothelial cells",
                            help="e.g. Endothelial cells, NK cells, T cells")
    disease = st.text_input("Disease (optional)", value="",
                            placeholder="breast cancer, NSCLC, …",
                            help="Leave blank for normal tissue context")

    st.markdown("---")
    st.subheader("Options")
    max_per_lit = st.slider("Max papers per source", min_value=2, max_value=10, value=5)
    show_raw    = st.checkbox("Show raw JSON output", value=False)

    st.markdown("---")
    run_btn = st.button("▶ Run Pipeline", type="primary",
                        use_container_width=True,
                        disabled=not backend_is_up())
    clr_btn = st.button("🗑 Clear Results", use_container_width=True)


# ── Main header ───────────────────────────────────────────────────────────────
st.title("🧬 Biomarker Validation Pipeline")
st.caption(
    "Multi-source literature + database mining · FAISS + BM25 hybrid RAG · "
    "6-agent extraction · Protein-level evidence prioritised"
)

# ── Clear ─────────────────────────────────────────────────────────────────────
if clr_btn:
    old_job = st.session_state.pop("job_id", None)
    if old_job:
        try:
            requests.delete(f"{API_BASE}/pipeline/job/{old_job}", timeout=3)
        except Exception:
            pass
    for key in ["result_df", "run_meta"]:
        st.session_state.pop(key, None)
    st.rerun()

# ── Submit job ────────────────────────────────────────────────────────────────
if run_btn:
    if not gene.strip():
        st.error("Please enter a gene symbol.")
    elif not cell.strip():
        st.error("Please enter a cell type.")
    else:
        for key in ["result_df", "run_meta", "job_id"]:
            st.session_state.pop(key, None)
        try:
            resp = api_post("/pipeline/run", {
                "gene":        gene.strip(),
                "cell":        cell.strip(),
                "disease":     disease.strip(),
                "max_per_lit": max_per_lit,
            })
            st.session_state["job_id"] = resp["job_id"]
        except Exception as exc:
            st.error(f"Could not start pipeline: {exc}")

# ── Live progress while running ───────────────────────────────────────────────
if "job_id" in st.session_state and "result_df" not in st.session_state:
    job_id = st.session_state["job_id"]

    try:
        job = api_get(f"/pipeline/status/{job_id}")
    except Exception as exc:
        st.error(f"Status fetch error: {exc}")
        st.stop()

    status  = job.get("status", "running")
    stage   = job.get("stage",  "…")
    pct     = job.get("pct",    0)
    elapsed = job.get("elapsed", 0)

    st.markdown(f"### ⏳ `{stage}`")
    st.progress(pct / 100)

    # Per-agent checklist
    for i, s in enumerate(STAGES, start=1):
        threshold = int((i / len(STAGES)) * 100)
        if pct >= threshold:
            icon = "✅"
        elif stage == s:
            icon = "🔄"
        else:
            icon = "⬜"
        st.caption(f"{icon}  {s}")

    st.caption(f"⏱ Elapsed: {elapsed}s")

    if status == "done":
        warning = job.get("warning", "")
        if warning:
            st.warning(warning)
        records = job.get("records", [])
        df = pd.DataFrame(records) if records else pd.DataFrame()
        st.session_state["result_df"] = df
        st.session_state["run_meta"]  = {
            "gene":    job.get("gene", gene),
            "cell":    job.get("cell", cell),
            "disease": job.get("disease", disease),
            "elapsed": elapsed,
            "n":       len(df),
        }
        st.rerun()

    elif status == "error":
        st.error(f"Pipeline error: {job.get('error', 'Unknown')}")
        with st.expander("Traceback"):
            st.code(job.get("trace", ""))
        st.session_state.pop("job_id", None)

    else:
        time.sleep(2)
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if "result_df" in st.session_state:
    df   = st.session_state["result_df"]
    meta = st.session_state.get("run_meta", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence Records", meta.get("n", len(df)))
    c2.metric("Gene",    meta.get("gene", ""))
    c3.metric("Disease", meta.get("disease") or "Normal tissue")
    c4.metric("Runtime", f"{meta.get('elapsed', '?')}s")

    st.markdown("---")

    if df.empty:
        st.warning("No evidence records found. Try broader inputs or a different disease context.")
    else:
        # Evidence tier summary
        if "Evidence Tier" in df.columns:
            with st.expander("📊 Evidence Tier Summary", expanded=True):
                tier_counts = df["Evidence Tier"].value_counts()
                sc1, sc2 = st.columns([1, 2])
                with sc1:
                    st.dataframe(tier_counts.rename("Count"), use_container_width=True)
                with sc2:
                    st.bar_chart(tier_counts)

        # Main table
        st.subheader(f"Evidence Table — {meta.get('gene','')} in {meta.get('cell','')}")

        available_cols = df.columns.tolist()
        default_cols   = [c for c in [
            "Marker Gene", "Cell Type (Query)", "Disease Context",
            "Assay", "Key Evidence Sentence", "Evidence Tier",
            "Evidence Score", "Validation Status", "Reference (PMID)", "Data Source",
        ] if c in available_cols]

        selected_cols = st.multiselect(
            "Select columns to display", available_cols, default=default_cols
        )
        display_df = df[selected_cols] if selected_cols else df

        try:
            st.dataframe(_style_df(display_df), use_container_width=True, height=480)
        except Exception:
            st.dataframe(display_df, use_container_width=True, height=480)

        # Filters
        with st.expander("🔍 Filter Results"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                status_filter = (
                    st.multiselect("Validation Status",
                                   df["Validation Status"].unique().tolist(),
                                   default=df["Validation Status"].unique().tolist())
                    if "Validation Status" in df.columns else None
                )
            with fc2:
                tier_filter = (
                    st.multiselect("Evidence Tier",
                                   df["Evidence Tier"].unique().tolist(),
                                   default=df["Evidence Tier"].unique().tolist())
                    if "Evidence Tier" in df.columns else None
                )
            with fc3:
                min_score = st.slider("Min Evidence Score", 0.0, 1.0, 0.0, step=0.05)

            filtered = df.copy()
            if status_filter and "Validation Status" in filtered.columns:
                filtered = filtered[filtered["Validation Status"].isin(status_filter)]
            if tier_filter and "Evidence Tier" in filtered.columns:
                filtered = filtered[filtered["Evidence Tier"].isin(tier_filter)]
            if "Evidence Score" in filtered.columns:
                filtered = filtered[filtered["Evidence Score"] >= min_score]

            st.write(f"Filtered: **{len(filtered)}** records")
            try:
                st.dataframe(
                    _style_df(filtered[selected_cols if selected_cols else filtered.columns]),
                    use_container_width=True, height=300,
                )
            except Exception:
                st.dataframe(filtered, use_container_width=True, height=300)

        # Downloads
        st.markdown("---")
        dc1, dc2 = st.columns(2)
        with dc1:
            buf = StringIO()
            df.to_csv(buf, index=False)
            st.download_button(
                "⬇ Download CSV",
                data=buf.getvalue(),
                file_name=f"biomarker_{meta.get('gene','')}_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dc2:
            st.download_button(
                "⬇ Download JSON",
                data=df.to_json(orient="records", indent=2),
                file_name=f"biomarker_{meta.get('gene','')}_results.json",
                mime="application/json",
                use_container_width=True,
            )

        if show_raw:
            with st.expander("🔧 Raw JSON"):
                st.json(json.loads(df.to_json(orient="records")))

    # ── Pipeline Diagnostics ─────────────────────────────────────────────────
    job_id = st.session_state.get("job_id")
    if job_id:
        try:
            job  = api_get(f"/pipeline/status/{job_id}")
            diag = job.get("diag", {})
        except Exception:
            diag = {}
    else:
        diag = {}

    if diag:
        with st.expander("🔬 Pipeline Diagnostics", expanded=False):
            st.caption(
                "Use this to diagnose why certain papers were retrieved or missed, "
                "and why extraction returned 'No evidence'."
            )
            tabs = st.tabs([
                "1 · Planner",
                "2 · Retrieval",
                "3 · Extraction",
                "4–5 · Validation & Scoring",
            ])

            # ── Tab 1: Planner ───────────────────────────────────────────────
            with tabs[0]:
                pl = diag.get("planner", {})
                if not pl:
                    st.info("Planner diagnostic not available.")
                else:
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown("**Gene synonyms used**")
                        st.write(pl.get("synonyms", []))
                        st.markdown("**Cell subtypes expanded**")
                        subtypes = pl.get("cell_subtypes", [])
                        st.write(subtypes if subtypes else "*(cell type already specific — no expansion)*")

                        # Disease expansion (new)
                        st.markdown("**Disease expansion**")
                        d_orig = pl.get("disease_original", "")
                        d_exp  = pl.get("disease_expanded", "")
                        d_syns = pl.get("disease_synonyms", [])
                        if d_orig or d_exp:
                            st.write(f"`{d_orig}` → **`{d_exp}`**" if d_orig != d_exp else f"*(unchanged)* `{d_exp}`")
                            if d_syns:
                                st.write(f"Synonyms: {d_syns}")
                        else:
                            st.write("*(no disease — normal tissue context)*")

                    with dc2:
                        st.markdown("**Anchor queries** *(always run)*")
                        for i, q in enumerate(pl.get("anchor_queries", []), 1):
                            st.code(f"[A{i}] {q}", language=None)
                        st.markdown("**LLM-generated queries**")
                        for i, q in enumerate(pl.get("llm_queries", []), 1):
                            st.code(f"[L{i}] {q}", language=None)
                    st.markdown(
                        f"**Queries sent to retrieval** (top 5): "
                        f"`{len(pl.get('queries_to_retrieval', []))}` queries"
                    )
                    for q in pl.get("queries_to_retrieval", []):
                        tag = "🔒 ANCHOR" if q in pl.get("anchor_queries", []) else "🤖 LLM"
                        st.markdown(f"- {tag} &nbsp; `{q}`", unsafe_allow_html=True)

            # ── Tab 2: Retrieval ─────────────────────────────────────────────
            with tabs[1]:
                ret = diag.get("retrieval", {})
                if not ret:
                    st.info("Retrieval diagnostic not available.")
                else:
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.metric("Papers retrieved", ret.get("papers_retrieved", 0))
                    with rc2:
                        by_src = ret.get("by_source", {})
                        st.markdown("**By source**")
                        for src, cnt in sorted(by_src.items()):
                            st.write(f"- {src}: **{cnt}**")

                    st.markdown("---")
                    st.markdown("**Retrieved papers**")
                    papers_info = ret.get("papers", [])
                    if papers_info:
                        rows = []
                        for p in papers_info:
                            rows.append({
                                "PMID":              p.get("pmid", "N/A"),
                                "Source":            p.get("source", "?"),
                                "Gene in abstract":  "✅" if p.get("gene_in_abstract") else "❌",
                                "Access type":       p.get("access_type", "?"),
                                "Abstract snippet":  p.get("snippet", "")[:120] + "…",
                            })
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.warning("No papers returned by retrieval.")

            # ── Tab 3: Extraction ────────────────────────────────────────────
            with tabs[2]:
                ext = diag.get("extraction", {})
                if not ext:
                    st.info("Extraction diagnostic not available.")
                else:
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.metric("Records extracted",   ext.get("records_extracted", 0))
                    ec2.metric("No-evidence records", ext.get("no_evidence_count", 0))
                    ec3.metric("Evidence found",       ext.get("evidence_found_count", 0))

                    st.markdown("---")
                    st.markdown("**Per-PMID extraction detail**")
                    per_pmid = ext.get("per_pmid", {})
                    if per_pmid:
                        rows = []
                        for pmid, info in per_pmid.items():
                            ev_level = info.get("evidence_level", "")
                            rows.append({
                                "PMID":              pmid,
                                "Source":            info.get("source", "?"),
                                "Evidence level":    ev_level,
                                "Assay":             info.get("assay", ""),
                                "Expression status": info.get("expression_status", ""),
                                "RAG chunks used":   info.get("rag_chunks_used", 0),
                                "Key sentence":      (info.get("key_sentence") or "")[:120] + "…",
                            })
                        ext_df = pd.DataFrame(rows)

                        def _hi(row):
                            if row.get("Evidence level") == "no evidence":
                                return ["background-color:#fff3cd"] * len(row)
                            elif row.get("Evidence level") == "direct":
                                return ["background-color:#d4edda"] * len(row)
                            return [""] * len(row)

                        try:
                            st.dataframe(
                                ext_df.style.apply(_hi, axis=1),
                                use_container_width=True, hide_index=True,
                            )
                        except Exception:
                            st.dataframe(ext_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No per-PMID extraction data available.")

                    st.caption(
                        "🟡 Yellow = no evidence returned by LLM | 🟢 Green = direct evidence found. "
                        "'Gene in abstract' ❌ means the paper abstract didn't mention the gene "
                        "— extraction will likely return no evidence for such papers."
                    )

            # ── Tab 4–5: Validation & Scoring ────────────────────────────────
            with tabs[3]:
                val  = diag.get("validation", {})
                scor = diag.get("scoring", {})
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.markdown("**Validation status counts**")
                    if val:
                        for vstatus, cnt in sorted(val.items()):
                            colour = {"PASS": "🟢", "NOT_EXPRESSED": "🟠",
                                      "FAIL": "🔴", "NA": "⚪"}.get(vstatus, "⚫")
                            st.write(f"{colour} **{vstatus}**: {cnt}")
                    else:
                        st.info("No validation data.")
                with vc2:
                    st.markdown("**Evidence tier counts**")
                    if scor.get("tier_summary"):
                        for tier, cnt in sorted(scor["tier_summary"].items()):
                            st.write(f"- **{tier}**: {cnt}")
                        st.metric("Top evidence score", round(scor.get("top_score", 0), 2))
                    else:
                        st.info("No scoring data.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Sources: PubMed · EuropePMC · Semantic Scholar · Human Protein Atlas · "
    "CellMarker · UniProt | "
    "RAG: FAISS (MedCPT) + BM25 hybrid | "
    "Backend: FastAPI on :8001"
)
