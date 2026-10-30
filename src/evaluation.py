"""Retrieval metrics (page-level relevance labels)."""
from __future__ import annotations


def recall_at_k(ranked_pages: list[int], relevant_pages: set[int], k: int) -> float:
    """1.0 if any relevant page appears in the top-k results, else 0.0 (hit-rate style)."""
    if not relevant_pages:
        raise ValueError("relevant_pages must not be empty")
    return 1.0 if any(p in relevant_pages for p in ranked_pages[:k]) else 0.0


def reciprocal_rank(ranked_pages: list[int], relevant_pages: set[int]) -> float:
    if not relevant_pages:
        raise ValueError("relevant_pages must not be empty")
    for i, p in enumerate(ranked_pages, start=1):
        if p in relevant_pages:
            return 1.0 / i
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
