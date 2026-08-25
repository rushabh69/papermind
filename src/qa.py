"""RAG answer generation: retrieve context, answer ONLY from it, cite sources.

To keep citations honest, the model doesn't invent page numbers - it answers from
numbered sources we give it and tells us which source numbers it used; we then map
those back to the real chunk metadata (page, section, snippet).
"""
from __future__ import annotations

from . import retrieval, llm


SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using ONLY "
    "the provided numbered sources. Do not use outside knowledge. If the sources "
    "do not contain enough information, set \"found\" to false and say you could "
    "not find it in the document. Cite the sources you used by their number. Be "
    "concise and accurate."
)


def answer(document_id: int, question: str, mode: str = "hybrid",
           use_rerank: bool = True) -> dict:
    """Return {answer, sources:[{page,section,snippet}], chunks:[...]}."""
    chunks = retrieval.retrieve(document_id, question, use_rerank=use_rerank, mode=mode)
    if not chunks:
        return {"answer": "Not found in the document.", "sources": [], "chunks": []}

    context = "\n\n".join(
        f"[Source {i + 1}] (page {c['page_number']}, "
        f"section: {c.get('section_title') or 'n/a'})\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    )
    user = (
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Respond as JSON with keys: "
        "\"answer\" (string), "
        "\"used_sources\" (list of the source numbers you actually used), "
        "\"found\" (boolean, false if the sources don't answer the question)."
    )
    data = llm.chat_json([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ])

    ans = str(data.get("answer", "")).strip()
    found = bool(data.get("found", True))
    used = data.get("used_sources") or []

    sources = []
    if found:
        for n in used:
            try:
                i = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(chunks):
                c = chunks[i]
                sources.append({
                    "page": c["page_number"],
                    "section": c.get("section_title"),
                    "snippet": c["chunk_text"][:400],
                })
    return {"answer": ans or "Not found in the document.",
            "sources": sources, "chunks": chunks}