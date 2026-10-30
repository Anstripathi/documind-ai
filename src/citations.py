"""Extract and validate page citations in LLM output against retrieved evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass

_BRACKET = re.compile(r"\[([^\[\]\n]{1,200})\]")
_PAGES = re.compile(
    r"(?:pages?|pp?\.)\s*(\d+(?:\s*(?:,|and|&)\s*(?:pages?\s*|pp?\.\s*)?\d+)*)", re.I
)
REMOVED_MARK = "[citation removed: page not in retrieved evidence]"


@dataclass
class CitationReport:
    text: str                 # answer text with invalid citations rewritten/removed
    cited_pages: list[int]    # every page the model cited (ordered, unique)
    valid_pages: list[int]
    invalid_pages: list[int]

    @property
    def has_valid_citation(self) -> bool:
        return bool(self.valid_pages)


def _pages_in_group(group: str) -> list[int]:
    pages: list[int] = []
    for m in _PAGES.finditer(group):
        pages.extend(int(n) for n in re.findall(r"\d+", m.group(1)))
    return pages


def extract_cited_pages(text: str) -> list[int]:
    seen: list[int] = []
    for m in _BRACKET.finditer(text):
        for p in _pages_in_group(m.group(1)):
            if p not in seen:
                seen.append(p)
    return seen


def validate_citations(text: str, allowed_pages: set[int]) -> CitationReport:
    cited: list[int] = []

    def _fix(m: re.Match) -> str:
        pages = _pages_in_group(m.group(1))
        if not pages:
            return m.group(0)  # not a page citation (e.g. "[1]")
        for p in pages:
            if p not in cited:
                cited.append(p)
        valid = [p for p in dict.fromkeys(pages) if p in allowed_pages]
        if len(valid) == len(set(pages)):
            return m.group(0)
        if valid:
            return "[" + ", ".join(f"Page {p}" for p in valid) + "]"
        return REMOVED_MARK

    cleaned = _BRACKET.sub(_fix, text)
    valid_pages = [p for p in cited if p in allowed_pages]
    invalid_pages = [p for p in cited if p not in allowed_pages]
    return CitationReport(cleaned, cited, valid_pages, invalid_pages)
