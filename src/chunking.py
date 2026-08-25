"""Structure-aware chunking.

Walks the ordered line-blocks from ingest.extract_document and groups them into
chunks of ~CHUNK_TARGET_TOKENS. Section headings start a fresh chunk (so a chunk
never straddles two sections) and are carried as the chunk's section_title;
within a long section we split on the token budget with a small overlap so
context isn't cut mid-thought. Tokens are approximated as chars / 4 (no tokenizer
dependency needed).
"""
from __future__ import annotations

from . import config


def _tok(text: str) -> int:
    return max(1, len(text) // config.CHARS_PER_TOKEN)


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    """Turn ingest blocks into [{chunk_index, page_number, section_title, chunk_text}]."""
    target = config.CHUNK_TARGET_TOKENS
    overlap = config.CHUNK_OVERLAP_TOKENS

    chunks: list[dict] = []
    cur_section: str | None = None
    buf: list[tuple[int, str]] = []          # (page_number, text)
    idx = 0

    def emit(with_overlap: bool):
        nonlocal buf, idx
        text = " ".join(t for _, t in buf).strip()
        if text:
            chunks.append({
                "chunk_index": idx,
                "page_number": buf[0][0],
                "section_title": cur_section,
                "chunk_text": text,
            })
            idx += 1
        if not with_overlap:
            buf = []
            return
        # keep a short tail so the next chunk overlaps this one
        tail, ttok = [], 0
        for p, t in reversed(buf):
            tail.insert(0, (p, t))
            ttok += _tok(t)
            if ttok >= overlap:
                break
        buf = tail

    for b in blocks:
        if b["is_heading"]:
            if buf:
                emit(with_overlap=False)     # clean break between sections
            cur_section = b["text"]
            buf = [(b["page_number"], b["text"])]   # seed chunk with the heading
            continue

        buf.append((b["page_number"], b["text"]))
        if sum(_tok(t) for _, t in buf) >= target:
            emit(with_overlap=True)

    if buf:
        emit(with_overlap=False)

    # drop trivial chunks that are essentially just a heading with no body
    if len(chunks) > 1:
        chunks = [c for c in chunks if _tok(c["chunk_text"]) >= 12]
        for i, c in enumerate(chunks):       # reindex after filtering
            c["chunk_index"] = i
    return chunks