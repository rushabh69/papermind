"""Hybrid retrieval: dense (FAISS) + BM25 keyword search, fused with reciprocal
rank fusion, then re-ranked with a cross-encoder.

retrieve() supports mode = "hybrid" | "dense" | "bm25" and an optional re-rank
toggle, so the evaluation harness can compare configurations.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from . import config, db, embeddings, vectorstore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# BM25 index is cheap to build per document; cache it in-process.
_bm25_cache: dict[int, tuple] = {}


def clear_cache(document_id: int | None = None):
    if document_id is None:
        _bm25_cache.clear()
    else:
        _bm25_cache.pop(document_id, None)


def _get_bm25(document_id: int):
    if document_id not in _bm25_cache:
        chunks = db.get_chunks(document_id)
        corpus = [_tokenize(c["chunk_text"]) for c in chunks]
        bm25 = BM25Okapi(corpus) if corpus else None
        _bm25_cache[document_id] = (bm25, [c["id"] for c in chunks])
    return _bm25_cache[document_id]


def _dense(document_id: int, query: str, k: int):
    qv = embeddings.embed_query(query)
    return vectorstore.search(document_id, qv, k)          # [(chunk_id, score)]


def _bm25(document_id: int, query: str, k: int):
    bm25, ids = _get_bm25(document_id)
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(query))
    order = np.argsort(scores)[::-1][:k]
    return [(ids[i], float(scores[i])) for i in order if scores[i] > 0]


def _rrf(rankings: list[list[tuple]], k: int = config.RRF_K):
    """Reciprocal rank fusion over several (chunk_id, score) ranking lists."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def retrieve(document_id: int, query: str, top_k: int = config.FINAL_TOP_K,
             use_rerank: bool = True, mode: str = "hybrid") -> list[dict]:
    """Return the top chunks (as dicts with an added 'score') for a query."""
    dense = _dense(document_id, query, config.DENSE_TOP_K)
    if mode == "hybrid":
        fused = _rrf([dense, _bm25(document_id, query, config.BM25_TOP_K)])
    elif mode == "dense":
        fused = dense
    elif mode == "bm25":
        fused = _bm25(document_id, query, config.DENSE_TOP_K)
    else:
        raise ValueError(f"unknown mode: {mode}")

    cand_ids = [cid for cid, _ in fused[:config.RERANK_CANDIDATES]]
    chunk_map = db.get_chunks_by_ids(cand_ids)
    candidates = [chunk_map[cid] for cid in cand_ids if cid in chunk_map]
    if not candidates:
        return []

    if use_rerank:
        scores = embeddings.rerank(query, [c["chunk_text"] for c in candidates])
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        candidates.sort(key=lambda c: -c["score"])
    else:
        fused_score = dict(fused)
        for c in candidates:
            c["score"] = fused_score.get(c["id"], 0.0)

    return candidates[:top_k]