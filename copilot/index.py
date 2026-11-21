"""Hybrid retrieval over 10-K chunks: FAISS HNSW (dense) + BM25 (sparse), fused with
reciprocal rank fusion, then reranked with a cross-encoder. Supports metadata filters
(ticker, fiscal year, 10-K item) and a per-user ticker allow-list for access control."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were what which with how did does do".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()) if t not in STOP]


def chunk_header(c: dict) -> str:
    """Contextual header so a chunk is still findable when its text never names the company."""
    return f"{c['company']} fiscal {c['fiscal_year']} 10-K, Item {c['item']} {c['section']}: "


@dataclass
class Filters:
    tickers: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    items: list[str] = field(default_factory=list)

    def empty(self) -> bool:
        return not (self.tickers or self.fiscal_years or self.items)


def extract_filters(query: str) -> Filters:
    """Metadata extraction from the question: company names/tickers and fiscal years."""
    q = query.lower()
    tickers = [t for t, names in config.TICKERS.items()
               if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names) or re.search(rf"\b{t}\b", query)]
    years = sorted({int(y) for y in re.findall(r"\b(?:fy\s?|fiscal\s(?:year\s)?)?(20[12]\d)\b", q)})
    return Filters(tickers=tickers, fiscal_years=years)


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL, max_length=512)


class HybridIndex:
    def __init__(self, strategy: str = "semantic"):
        self.strategy = strategy
        path = config.PROCESSED / f"chunks_{strategy}.jsonl"
        self.chunks = [json.loads(line) for line in path.open()]
        self.texts = [chunk_header(c) + c["text"] for c in self.chunks]
        self.emb = self._embeddings()
        self.hnsw = faiss.IndexHNSWFlat(self.emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        self.hnsw.hnsw.efSearch = 128
        self.hnsw.add(self.emb)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.ticker = np.array([c["ticker"] for c in self.chunks])
        self.year = np.array([c["fiscal_year"] for c in self.chunks])
        self.item = np.array([c["item"] for c in self.chunks])

    def _embeddings(self) -> np.ndarray:
        cache = config.PROCESSED / f"emb_{self.strategy}.npy"
        if cache.exists():
            emb = np.load(cache)
            if len(emb) == len(self.chunks):
                return emb
        emb = embedder().encode(self.texts, normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False).astype("float32")
        np.save(cache, emb)
        return emb

    # ---- candidate mask -------------------------------------------------------------
    def _mask(self, f: Filters, allowed: list[str] | None) -> np.ndarray | None:
        m = np.ones(len(self.chunks), bool)
        if f.tickers:
            m &= np.isin(self.ticker, f.tickers)
        if f.fiscal_years:
            m &= np.isin(self.year, f.fiscal_years)
        if f.items:
            m &= np.isin(self.item, f.items)
        if allowed is not None:  # access control: never return chunks the user cannot see
            m &= np.isin(self.ticker, allowed)
        if not m.any() and allowed is None and (f.fiscal_years or f.items):
            # a year we have no filing for: fall back to the company-only filter
            return self._mask(Filters(tickers=f.tickers), None)
        return None if m.all() else m

    # ---- retrievers ------------------------------------------------------------------
    def dense(self, query: str, k: int, mask: np.ndarray | None) -> list[int]:
        q = embedder().encode([config.EMBED_QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        if mask is None:  # approximate nearest neighbour search over everything
            _, idx = self.hnsw.search(q, k)
            return [int(i) for i in idx[0] if i >= 0]
        cand = np.flatnonzero(mask)  # filtered subset is small: exact cosine similarity
        scores = self.emb[cand] @ q[0]
        return [int(cand[i]) for i in np.argsort(-scores)[:k]]

    def sparse(self, query: str, k: int, mask: np.ndarray | None) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [int(i) for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    @staticmethod
    def rrf(*rankings: list[int]) -> list[int]:
        score: dict[int, float] = {}
        for r in rankings:
            for rank, i in enumerate(r):
                score[i] = score.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        return sorted(score, key=score.get, reverse=True)

    def search(self, query: str, k: int = config.CONTEXT_K, mode: str = "hybrid_rerank",
               filters: Filters | None = None, allowed_tickers: list[str] | None = None,
               auto_filters: bool = True) -> list[dict]:
        """Return the top-k chunks (dicts with an added 'score')."""
        if filters is None:
            filters = extract_filters(query) if auto_filters else Filters()
        mask = self._mask(filters, allowed_tickers)
        if mask is not None and not mask.any():
            return []
        n = config.RETRIEVE_K
        if mode == "dense":
            ids, scores = self.dense(query, k, mask), None
        elif mode == "bm25":
            ids, scores = self.sparse(query, k, mask), None
        else:
            fused = self.rrf(self.dense(query, n, mask), self.sparse(query, n, mask))
            if mode == "hybrid":
                ids, scores = fused[:k], None
            else:
                pairs = [(query, self.texts[i]) for i in fused]
                rs = reranker().predict(pairs, batch_size=32, show_progress_bar=False) if pairs else []
                order = np.argsort(-np.asarray(rs))[:k]
                ids, scores = [fused[i] for i in order], [float(rs[i]) for i in order]
        out = []
        for r, i in enumerate(ids[:k]):
            c = dict(self.chunks[i])
            c["score"] = scores[r] if scores else None
            out.append(c)
        return out


@lru_cache(maxsize=2)
def get_index(strategy: str = "semantic") -> HybridIndex:
    return HybridIndex(strategy)
