"""Central configuration for PaperMind - one place for every tunable."""
from __future__ import annotations

import os
from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"        # original PDFs
INDEX_DIR = DATA_DIR / "index"            # per-document fAISS index + chunk store
DB_PATH = DATA_DIR / "papermind.db"       # SQLite (docs, chat history, quizzes)

for _d in (DATA_DIR, UPLOADS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# embedding + re-ranking models (local, CPU, free)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DIM = 384                            # all-MiniLM-L6-v2 output size

# chunking (~300-500 tokens with overlap; tokens approximated as chars/4)
CHUNK_TARGET_TOKENS = 400
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 60
CHARS_PER_TOKEN = 4                        # rough heuristic, no tokenizer needed

# retrieval
DENSE_TOP_K = 20                           # candidates from embedding search
BM25_TOP_K = 20                           # candidates from keyword search
RRF_K = 60                                 # reciprocal-rank-fusion constant
RERANK_CANDIDATES = 20                     # how many fused hits to cross-encode
FINAL_TOP_K = 5                            # chunks kept as LLM context

# Groq (LLM inference, free tier)
GROQ_MODEL = "llama-3.3-70b-versatile"     # main answering / generation model
GROQ_MODEL_FAST = "llama-3.1-8b-instant"   # cheaper model for grading etc.
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 1500


def groq_api_key():
    """Groq key from env var or Streamlit secrets (returns None if unset)."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None