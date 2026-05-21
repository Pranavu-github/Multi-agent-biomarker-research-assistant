"""
utils/rag_store.py — Hybrid FAISS + BM25 RAG Store (MedCPT edition).

Changes from all-MiniLM-L6-v2 version:
  - Dense  : FAISS with MedCPT asymmetric encoders
               Query  → ncbi/MedCPT-Query-Encoder
               Chunks → ncbi/MedCPT-Article-Encoder
  - Sparse : BM25Okapi with biomedical-aware tokenisation
  - Fusion : RRF (unchanged — still the right choice, see notes)
  - Rerank : Optional MedCPT-Cross-Encoder (off by default, flag-gated)

Why MedCPT over all-MiniLM-L6-v2:
  - Trained on 255 M PubMed click-through logs → understands gene/assay/disease vocab
  - Asymmetric design: query encoder ≠ article encoder (critical — do NOT use
    the same encoder for both sides; that negates the contrastive training)
  - 768-dim vs 384-dim → richer representation for biomedical entities

Why RRF is still the right fusion strategy (not changed):
  - BM25 exact-keyword matching is especially valuable for gene symbols, assay
    names, and PMID-specific vocabulary that dense models can miss
  - RRF is rank-based → no score calibration needed between FAISS and BM25
  - Cross-encoder adds ~1–2s per 20 candidates; optional flag lets you trade
    latency for precision on smaller result sets

Cross-encoder recommendation:
  - Use ncbi/MedCPT-Cross-Encoder (same family, PubMed-trained)
  - Enable only when top_k ≤ 15 and latency budget allows
  - Flag: BiomarkerRAGStore(use_cross_encoder=True)
"""

from __future__ import annotations

import re
import uuid
import numpy as np
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
QUERY_ENCODER_MODEL   = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER_MODEL = "ncbi/MedCPT-Article-Encoder"
CROSS_ENCODER_MODEL   = "ncbi/MedCPT-Cross-Encoder"

# Fallback (if MedCPT / transformers not available)
FALLBACK_EMBED_MODEL  = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 768    # increased from 512 — biomedical sentences are long
CHUNK_OVERLAP = 120    # slightly wider overlap to preserve sentence boundaries
RRF_K         = 60

# BM25 stopwords — generic words that add noise in biomedical retrieval
_BM25_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "that", "this", "these", "those", "with",
    "for", "on", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "each", "which", "who", "whom", "when", "where", "how",
})

# Lazy-loaded singletons — one per process, shared across all store instances
_query_encoder   = None
_article_encoder = None
_cross_encoder   = None
_use_medcpt      = None   # set on first load attempt


# ── Encoder loading ───────────────────────────────────────────────────────────

class _HashEmbedder:
    """
    Hash-based fallback embedder when transformers/torch unavailable.
    BM25 still provides keyword-exact retrieval; dense quality is lower.
    FOR DEVELOPMENT / TESTING ONLY — do not use in production.
    """
    DIM = 384   # match MiniLM dim so FAISS index is consistent in tests

    def encode(self, texts, batch_size=64, show_progress_bar=False, **kwargs):
        out = []
        for text in texts:
            vec = np.zeros(self.DIM, dtype=np.float32)
            tokens = text.lower().split()
            for i, tok in enumerate(tokens):
                h = hash(tok) % self.DIM
                vec[h] += 1.0 / (i + 1)
            # Normalise so cosine == inner product
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            out.append(vec)
        return np.array(out, dtype=np.float32)

    @property
    def dim(self):
        return self.DIM


def _load_encoders():
    """
    Try to load MedCPT asymmetric encoders.
    Falls back to all-MiniLM-L6-v2 (symmetric) if MedCPT unavailable.
    Sets _use_medcpt flag so retrieve() knows which path to take.
    """
    global _query_encoder, _article_encoder, _use_medcpt

    if _use_medcpt is not None:
        return  # already loaded

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        logger.info("[rag_store] Loading MedCPT encoders (Query + Article)...")

        _query_encoder   = _MedCPTEncoder(QUERY_ENCODER_MODEL)
        _article_encoder = _MedCPTEncoder(ARTICLE_ENCODER_MODEL)
        _use_medcpt = True

        logger.info("[rag_store] MedCPT encoders loaded successfully")

    except (ImportError, OSError, Exception) as exc:
        logger.warning(
            f"[rag_store] MedCPT not available ({exc}). "
            "Falling back to all-MiniLM-L6-v2 (symmetric, lower biomedical accuracy). "
            "Install: pip install transformers torch sentence-transformers"
        )
        try:
            from sentence_transformers import SentenceTransformer
            _fallback        = SentenceTransformer(FALLBACK_EMBED_MODEL)
            _query_encoder   = _fallback
            _article_encoder = _fallback
            _use_medcpt = False
            logger.info(f"[rag_store] Fallback loaded: {FALLBACK_EMBED_MODEL}")
        except (ImportError, Exception) as exc2:
            logger.warning(
                f"[rag_store] sentence-transformers also unavailable ({exc2}). "
                "Using hash-based embedder — DEV/TEST ONLY."
            )
            _hash            = _HashEmbedder()
            _query_encoder   = _hash
            _article_encoder = _hash
            _use_medcpt = False


def _load_cross_encoder():
    """Load MedCPT cross-encoder (called only if use_cross_encoder=True)."""
    global _cross_encoder
    if _cross_encoder is not None:
        return True
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        _cross_encoder = _MedCPTCrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("[rag_store] MedCPT Cross-Encoder loaded")
        return True
    except (ImportError, OSError, Exception) as exc:
        logger.warning(f"[rag_store] Cross-encoder unavailable ({exc}) — skipping rerank step")
        return False


# ── MedCPT encoder wrappers ───────────────────────────────────────────────────

class _MedCPTEncoder:
    """
    Thin wrapper around a HuggingFace AutoModel for MedCPT-style mean-pooled
    sentence embeddings.  Matches the official MedCPT inference recipe.
    """
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model     = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._torch  = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        logger.debug(f"[rag_store] {model_name} on {self._device}")

    @property
    def dim(self) -> int:
        return self._model.config.hidden_size  # 768 for MedCPT

    def encode(
        self,
        texts:             list[str],
        batch_size:        int  = 32,
        show_progress_bar: bool = False,
        max_length:        int  = 512,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode a list of texts → L2-normalised float32 numpy array.
        Uses mean pooling over the last hidden states (official MedCPT recipe).
        """
        import torch
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch   = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**encoded)

            # Mean pool over token dimension (ignore padding via attention mask)
            attention_mask   = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            embeddings = embeddings / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = embeddings.cpu().numpy().astype(np.float32)

            # L2-normalise for cosine similarity via inner product
            norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-10, None)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


class _MedCPTCrossEncoder:
    """
    MedCPT cross-encoder wrapper.
    Input: (query, passage) pairs → relevance score (higher = more relevant).
    """
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._torch  = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def predict(
        self,
        pairs:      list[tuple[str, str]],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Score a list of (query, passage) pairs.
        Returns float32 array of logit scores (higher = more relevant).
        """
        import torch
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch    = pairs[i : i + batch_size]
            queries  = [p[0] for p in batch]
            passages = [p[1] for p in batch]
            encoded  = self._tokenizer(
                queries,
                passages,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(-1)
            all_scores.extend(logits.cpu().numpy().tolist())

        return np.array(all_scores, dtype=np.float32)


# ── BM25 tokenisation ─────────────────────────────────────────────────────────

def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Biomedical-aware BM25 tokenisation.

    Improvements over naive .lower().split():
      1. Preserves hyphenated terms (e.g. 'CD8-positive', 'HER-2', 'smFISH')
      2. Preserves slash-joined terms (e.g. 'IHC/IF', 'CD4+/CD8+')
      3. Removes pure stopwords but keeps gene symbols and assay abbreviations
      4. Keeps numeric-alpha tokens (e.g. 'CD4', 'IL-6', 'MCF-7')
    """
    text   = re.sub(r"\s+", " ", text.lower().strip())
    tokens = re.split(r"[,;:\(\)\[\]{}\"\']|\s+", text)

    cleaned = []
    for tok in tokens:
        tok = tok.strip(".")
        if not tok:
            continue
        if tok in _BM25_STOPWORDS and len(tok) > 3:
            continue
        if re.search(r"[a-z0-9]", tok):
            cleaned.append(tok)

    return cleaned


# ── Section detection ─────────────────────────────────────────────────────────

_SECTION_PATTERNS = {
    "Abstract":     r"\b(abstract)\b",
    "Introduction": r"\b(introduction|background)\b",
    "Methods":      r"\b(methods?|materials?\s+and\s+methods?|methodology|experimental\s+procedure)\b",
    "Results":      r"\b(results?|findings?|observations?|outcomes?)\b",
    "Discussion":   r"\b(discussion|interpretation)\b",
    "Conclusion":   r"\b(conclusion|summary|concluding\s+remarks)\b",
}
_HIGH_VALUE_SECTIONS = {"Results", "Methods", "Discussion"}


def _detect_section(text: str) -> str:
    first = text[:120].lower()
    for label, pat in _SECTION_PATTERNS.items():
        if re.search(pat, first, re.IGNORECASE):
            return label
    return "Body"


# ── Chunk metadata ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:      str
    text:          str
    pmid:          str
    source:        str
    section:       str
    gene:          str
    disease:       str
    is_high_value: bool = False

    def to_dict(self) -> dict:
        return {
            "chunk_id":      self.chunk_id,
            "chunk_text":    self.text,
            "pmid":          self.pmid,
            "source":        self.source,
            "section":       self.section,
            "gene":          self.gene,
            "disease":       self.disease,
            "is_high_value": self.is_high_value,
        }


# ── Main store class ──────────────────────────────────────────────────────────

class BiomarkerRAGStore:
    """
    In-memory FAISS + BM25 hybrid RAG store.

    Encoding:
        - Documents : MedCPT-Article-Encoder  (768-dim, PubMed-trained)
        - Queries   : MedCPT-Query-Encoder    (768-dim, asymmetric dual-encoder)
        - Fallback  : all-MiniLM-L6-v2        (384-dim, symmetric)

    Retrieval:
        - RRF fusion of FAISS (dense) + BM25 (sparse)
        - Optional: MedCPT-Cross-Encoder reranking (flag-gated)

    Lifecycle:
        store = BiomarkerRAGStore()
        store.ingest(papers, gene, disease)
        chunks = store.retrieve(query, top_k)
        store.clear()
    """

    def __init__(self, use_cross_encoder: bool = False):
        """
        Args:
            use_cross_encoder: Enable MedCPT-Cross-Encoder reranking.
                               Adds ~1–2s latency per retrieve() call.
                               Recommended only when top_k ≤ 15.
        """
        self._chunks:           list[Chunk] = []
        self._embeddings:       Optional[np.ndarray] = None
        self._faiss_index       = None
        self._bm25              = None
        self._use_cross_encoder = use_cross_encoder
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        _load_encoders()
        if use_cross_encoder:
            _load_cross_encoder()

    # ── Ingest ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        papers:  list[dict],
        gene:    str,
        disease: str = "",
    ) -> int:
        """
        Chunk and index a list of paper dicts.

        Paper dict format (flexible):
            {
                "pmid":     str,
                "source":   str,
                "sections": [{"section": str, "text": str}],  # preferred
                "text":     str,                               # legacy fallback
                "abstract": str,                               # legacy fallback
            }

        Returns:
            Number of chunks indexed.
        """
        self.clear()
        all_chunks: list[Chunk] = []

        for paper in papers:
            pmid   = str(paper.get("pmid", "unknown"))
            source = paper.get("source", "unknown")

            sections: list[dict] = paper.get("sections") or []
            if not sections:
                raw = paper.get("text") or paper.get("abstract") or ""
                if raw:
                    sections = [{"section": "Body", "text": raw}]

            for sec in sections:
                sec_label = sec.get("section") or _detect_section(sec.get("text", ""))
                sec_text  = sec.get("text", "").strip()
                if not sec_text or len(sec_text) < 40:
                    continue

                sub_chunks = self._splitter.split_text(sec_text)
                for sub in sub_chunks:
                    sub = sub.strip()
                    if len(sub) < 30:
                        continue
                    all_chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            text=sub,
                            pmid=pmid,
                            source=source,
                            section=sec_label,
                            gene=gene,
                            disease=disease,
                            is_high_value=(sec_label in _HIGH_VALUE_SECTIONS),
                        )
                    )

        if not all_chunks:
            logger.warning("[rag_store] No chunks produced during ingest")
            return 0

        self._chunks = all_chunks
        self._build_faiss()
        self._build_bm25()
        logger.info(
            f"[rag_store] Ingested {len(papers)} papers → {len(all_chunks)} chunks | "
            f"Encoder: {'MedCPT' if _use_medcpt else 'MiniLM-fallback'}"
        )
        return len(all_chunks)

    # ── FAISS index ────────────────────────────────────────────────────────

    def _build_faiss(self):
        import faiss

        texts = [c.text for c in self._chunks]
        # KEY: use Article encoder for chunks, Query encoder for queries (asymmetric)
        embs  = _article_encoder.encode(texts, batch_size=32, show_progress_bar=False)
        embs  = np.array(embs, dtype=np.float32)

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs  = embs / np.clip(norms, 1e-10, None)
        self._embeddings = embs

        d     = embs.shape[1]
        index = faiss.IndexFlatIP(d)   # cosine on L2-normalised vectors
        index.add(embs)
        self._faiss_index = index
        logger.debug(f"[rag_store] FAISS built: {index.ntotal} vectors, dim={d}")

    # ── BM25 index ─────────────────────────────────────────────────────────

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        tokenised  = [_tokenize_for_bm25(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.debug(f"[rag_store] BM25 built: {len(tokenised)} docs")

    # ── Retrieve ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Hybrid FAISS + BM25 retrieval with RRF fusion.
        Optionally followed by MedCPT-Cross-Encoder reranking.

        Returns:
            List of chunk dicts sorted by final score (highest first).
            Each dict includes: chunk_text, pmid, source, section,
                                gene, disease, is_high_value, rrf_score,
                                and (if cross-encoder) cross_score.
        """
        if not self._chunks:
            return []

        n           = len(self._chunks)
        candidate_k = min(
            n,
            max(top_k * 4, 40) if self._use_cross_encoder else max(top_k * 3, 30)
        )

        # ── Dense (FAISS) — use QUERY encoder for the query ──────────────
        q_emb  = _query_encoder.encode([query], show_progress_bar=False)
        q_emb  = np.array(q_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb  = q_emb / np.clip(q_norm, 1e-10, None)

        _, faiss_idxs = self._faiss_index.search(q_emb, candidate_k)
        dense_rank = {int(idx): rank for rank, idx in enumerate(faiss_idxs[0]) if idx >= 0}

        # ── Sparse (BM25) — biomedical tokenisation ───────────────────────
        q_tokens    = _tokenize_for_bm25(query)
        bm25_scores = self._bm25.get_scores(q_tokens)
        sparse_idxs = np.argsort(bm25_scores)[::-1][:candidate_k]
        sparse_rank = {int(idx): rank for rank, idx in enumerate(sparse_idxs)}

        # ── RRF fusion ───────────────────────────────────────────────────
        all_idxs    = set(dense_rank.keys()) | set(sparse_rank.keys())
        rrf_scores: dict[int, float] = {}
        for idx in all_idxs:
            score = 0.0
            if idx in dense_rank:
                score += 1.0 / (RRF_K + dense_rank[idx] + 1)
            if idx in sparse_rank:
                score += 1.0 / (RRF_K + sparse_rank[idx] + 1)
            rrf_scores[idx] = score

        rrf_top_k = top_k * 3 if self._use_cross_encoder else top_k
        ranked    = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:rrf_top_k]

        # ── Optional Cross-Encoder reranking ─────────────────────────────
        if self._use_cross_encoder and _cross_encoder is not None and ranked:
            pairs  = [(query, self._chunks[idx].text) for idx, _ in ranked]
            scores = _cross_encoder.predict(pairs)
            ranked = sorted(
                zip([idx for idx, _ in ranked], scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            results = []
            for idx, ce_score in ranked:
                chunk = self._chunks[idx]
                d = chunk.to_dict()
                d["rrf_score"]   = round(rrf_scores.get(idx, 0.0), 6)
                d["cross_score"] = round(float(ce_score), 4)
                results.append(d)
        else:
            ranked  = ranked[:top_k]
            results = []
            for idx, score in ranked:
                chunk = self._chunks[idx]
                d = chunk.to_dict()
                d["rrf_score"] = round(score, 6)
                results.append(d)

        return results

    # ── Utilities ──────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def encoder_info(self) -> dict:
        return {
            "dense_query":   QUERY_ENCODER_MODEL   if _use_medcpt else FALLBACK_EMBED_MODEL,
            "dense_article": ARTICLE_ENCODER_MODEL if _use_medcpt else FALLBACK_EMBED_MODEL,
            "sparse":        "BM25Okapi",
            "fusion":        "RRF",
            "cross_encoder": CROSS_ENCODER_MODEL if (self._use_cross_encoder and _cross_encoder) else None,
            "medcpt_active": bool(_use_medcpt),
        }

    def clear(self):
        self._chunks      = []
        self._embeddings  = None
        self._faiss_index = None
        self._bm25        = None


# ── Module-level convenience factory ──────────────────────────────────────────

_active_store: Optional[BiomarkerRAGStore] = None


def get_store(use_cross_encoder: bool = False) -> BiomarkerRAGStore:
    """Return the module-level store (creates if needed)."""
    global _active_store
    if _active_store is None:
        _active_store = BiomarkerRAGStore(use_cross_encoder=use_cross_encoder)
    return _active_store


def reset_store(use_cross_encoder: bool = False) -> None:
    """Clear and reset the module-level store (call before each pipeline run)."""
    global _active_store
    if _active_store is not None:
        _active_store.clear()
    _active_store = BiomarkerRAGStore(use_cross_encoder=use_cross_encoder)
    logger.info("[rag_store] Store reset")
