"""PDF ingestion with PyMuPDF: page-accurate text + heading/section detection.

extract_document() returns an ordered list of line "blocks" (each tagged with its
page number and whether it looks like a heading), which the chunker turns into
structure-aware chunks. We also expose a page-image renderer used by the preview
panel to show (and highlight) the exact source of an answer.
"""
from __future__ import annotations

import re
from collections import Counter

import fitz  # PyMuPDF


BOLD_FLAG = 16  # span["flags"] bit for bold


def _line_text(line) -> str:
    return "".join(span["text"] for span in line["spans"]).strip()


def _looks_like_heading(text: str, max_size: float, bold: bool,
                        heading_min: float) -> bool:
    if not text or len(text) > 140:
        return False
    words = text.split()
    if not words:
        return False
    # numbered / named headings: "3.2 Methods", "Chapter 4", "Section 2"
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", text):
        return True
    if re.match(r"^(chapter|section|appendix|part)\s+\d", text, re.I):
        return True
    # big font, short line
    if max_size >= heading_min and len(words) <= 14:
        return True
    # bold-ish short line slightly above body size
    if bold and len(words) <= 10 and max_size >= heading_min * 0.98:
        return True
    # ALL CAPS short line
    if text.isupper() and 1 < len(words) <= 10:
        return True
    return False


def _dedupe_keep_order(items):
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def extract_document(pdf_path: str) -> dict:
    """Return {total_pages, blocks, headings, sections}.

    blocks: ordered [{page_number, text, is_heading}] across the whole document.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    # first pass: find the body font size (most common size, weighted by chars)
    size_counter: Counter = Counter()
    page_dicts = []
    for pno in range(total_pages):
        d = doc[pno].get_text("dict")
        page_dicts.append(d)
        for block in d["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    size_counter[round(span["size"])] += len(span["text"])
    body_size = size_counter.most_common(1)[0][0] if size_counter else 10
    heading_min = body_size * 1.15

    blocks, headings = [], []
    for pno in range(total_pages):
        for block in page_dicts[pno]["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                txt = _line_text(line)
                if not txt:
                    continue
                sizes = [s["size"] for s in line["spans"]]
                max_size = max(sizes) if sizes else body_size
                bold = any((s["flags"] & BOLD_FLAG) for s in line["spans"])
                is_heading = _looks_like_heading(txt, max_size, bold, heading_min)
                blocks.append({"page_number": pno + 1, "text": txt,
                               "is_heading": is_heading})
                if is_heading:
                    headings.append({"page_number": pno + 1, "title": txt})
    doc.close()

    sections = _dedupe_keep_order([h["title"] for h in headings])
    return {"total_pages": total_pages, "blocks": blocks,
            "headings": headings, "sections": sections}


def page_text(pdf_path: str, page_number: int) -> str:
    """Plain text of a single page (1-based)."""
    doc = fitz.open(pdf_path)
    try:
        return doc[page_number - 1].get_text("text")
    finally:
        doc.close()


def render_page_png(pdf_path: str, page_number: int, highlight: str | None = None,
                    zoom: float = 2.0) -> bytes:
    """Render a page to PNG bytes, optionally highlighting a snippet."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        if highlight:
            needle = highlight.strip()[:80]
            try:
                for inst in page.search_for(needle):
                    page.add_highlight_annot(inst)
            except Exception:
                pass
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()