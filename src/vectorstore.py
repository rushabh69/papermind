"""Per-document FAISS vector store.

One small IndexFlatIP per document (inner product on normalised vectors = cosine).
Row i of the index corresponds to chunk_ids[i], so a search returns real chunk ids
we can look up in SQLite. Index + id map are written to disk so they survive
restarts.
"""
from __future__ import annotations

import numpy as np
import faiss

from . import config


def _paths(document_id: int):
    idx = config.INDEX_DIR / f"{document_id}.faiss"
    ids = config.INDEX_DIR / f"{document_id}_ids.npy"
    return idx, ids


def build_index(document_id: int, embeddings: np.ndarray, chunk_ids: list[int]):
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    idx_path, ids_path = _paths(document_id)
    faiss.write_index(index, str(idx_path))
    np.save(ids_path, np.asarray(chunk_ids, dtype=np.int64))
    return index


def load_index(document_id: int):
    idx_path, ids_path = _paths(document_id)
    if not idx_path.exists() or not ids_path.exists():
        return None, None
    return faiss.read_index(str(idx_path)), np.load(ids_path)


def search(document_id: int, query_vec: np.ndarray, top_k: int):
    """Return [(chunk_id, score)] for the nearest chunks."""
    index, ids = load_index(document_id)
    if index is None or index.ntotal == 0:
        return []
    q = np.ascontiguousarray(query_vec.reshape(1, -1).astype("float32"))
    k = min(top_k, index.ntotal)
    scores, rows = index.search(q, k)
    out = []
    for score, row in zip(scores[0], rows[0]):
        if row >= 0:
            out.append((int(ids[row]), float(score)))
    return out


def delete_index(document_id: int):
    for p in _paths(document_id):
        if p.exists():
            p.unlink()