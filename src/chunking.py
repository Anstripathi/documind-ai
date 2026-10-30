"""Page-aware character chunking.

Strategy: chunks never cross page boundaries, so every chunk maps to exactly one page.
Limitation: a sentence or table that spans two pages is split between chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf_loader import LoadedDocument


@dataclass(frozen=True)
class Chunk:
    chunk_id: str   # stable: "<doc_id>-p<page>-c<index>"
    doc_id: str
    doc_name: str
    page: int
    index: int      # index within the page
    text: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = _normalize(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # try to end on a space within the last 20% of the window
            window_start = start + int(chunk_size * 0.8)
            cut = text.rfind(" ", window_start, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # guarantees forward progress
    return pieces


def chunk_document(doc: LoadedDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        for i, piece in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-p{page.page_number}-c{i}",
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    page=page.page_number,
                    index=i,
                    text=piece,
                )
            )
    return chunks
