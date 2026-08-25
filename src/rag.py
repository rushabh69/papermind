"""High-level orchestration used by the Streamlit pages.

One import surface for: ingest a PDF end-to-end, ask a question (and persist the
turn), locate a document's PDF for the preview panel, and delete a document.
"""
from __future__ import annotations

import re

from . import config, db, ingest, chunking, embeddings, vectorstore, retrieval, qa


def init():
    db.init_db()


def _safe_name(filename: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", filename).strip("_") or "document.pdf"


def upload_path(document_id: int, filename: str):
    return config.UPLOADS_DIR / f"{document_id}_{filename}"


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    """Extract -> chunk -> store -> embed -> index. Returns the new document_id."""
    filename = _safe_name(filename)
    tmp = config.UPLOADS_DIR / f"_tmp_{filename}"
    tmp.write_bytes(file_bytes)

    doc = ingest.extract_document(str(tmp))
    chunks = chunking.chunk_blocks(doc["blocks"])

    document_id = db.add_document(filename, doc["total_pages"], doc["sections"])
    tmp.replace(upload_path(document_id, filename))        # keep the pdf for preview

    chunk_ids = db.add_chunks(document_id, chunks)
    if chunks:
        vecs = embeddings.embed_texts([c["chunk_text"] for c in chunks])
        vectorstore.build_index(document_id, vecs, chunk_ids)
    retrieval.clear_cache(document_id)
    return document_id


def pdf_path(document_id: int):
    doc = db.get_document(document_id)
    if not doc:
        return None
    p = upload_path(document_id, doc["filename"])
    return p if p.exists() else None


def ask(document_id: int, question: str, mode: str = "hybrid",
        use_rerank: bool = True, save: bool = True) -> dict:
    result = qa.answer(document_id, question, mode=mode, use_rerank=use_rerank)
    if save:
        db.add_conversation(document_id, question, result["answer"], result["sources"])
    return result


def delete_document(document_id: int):
    p = pdf_path(document_id)
    if p and p.exists():
        p.unlink()
    vectorstore.delete_index(document_id)
    retrieval.clear_cache(document_id)
    db.delete_document(document_id)