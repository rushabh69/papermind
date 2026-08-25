"""Local embeddings + cross-encoder re-ranking (sentence-transformers, CPU, free).

Models are loaded lazily and cached so the (slow) first load happens once.
Embeddings are L2-normalised so inner product == cosine similarity, which lets
the FAISS IndexFlatIP double as a cosine index.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.CROSS_ENCODER_MODEL)


def embed_texts(texts, batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    model = _embedder()
    emb = model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(emb, dtype="float32")


def embed_query(text: str, normalize: bool = True) -> np.ndarray:
    return embed_texts([text], normalize=normalize)[0]


def rerank(query: str, candidate_texts: list[str]) -> list[float]:
    """Cross-encoder relevance scores for (query, candidate) pairs."""
    if not candidate_texts:
        return []
    ce = _cross_encoder()
    scores = ce.predict([(query, t) for t in candidate_texts])
    return [float(s) for s in scores]