"""Vector index (FAISS when available, exact NumPy search otherwise) and a Retriever."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder

try:  # optional dependency
    if os.getenv("DOCUMIND_NO_FAISS"):
        raise ImportError
    import faiss  # type: ignore
except ImportError:
    faiss = None


class VectorIndex:
    """Inner-product index over L2-normalised vectors (= cosine similarity)."""

    def __init__(self, vectors: np.ndarray):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("vectors must be a non-empty 2-D array")
        self.dim = vectors.shape[1]
        self.size = vectors.shape[0]
        self.backend = "faiss" if faiss is not None else "numpy"
        if faiss is not None:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(vectors)
        else:
            self._vecs = vectors

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = max(1, min(k, self.size))
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError("query dimension mismatch")
        if faiss is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        scores = (self._vecs @ q[0]).astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity; NOT a probability of correctness
    rank: int


class Retriever:
    """One Retriever per uploaded document/session -> no cross-session leakage."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._index: VectorIndex | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        vecs = self.embedder.embed([c.text for c in chunks])
        if vecs.shape[0] != len(chunks):
            raise ValueError("Embedder returned wrong number of vectors")
        self.chunks = list(chunks)
        self._index = VectorIndex(vecs)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if self._index is None or not query or not query.strip():
            return []
        qvec = self.embedder.embed([query.strip()])[0]
        hits = self._index.search(qvec, k)
        return [
            RetrievedChunk(chunk=self.chunks[i], score=s, rank=r + 1)
            for r, (i, s) in enumerate(hits)
        ]
