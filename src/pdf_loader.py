"""PDF validation and page-by-page text extraction (in memory, nothing written to disk)."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field


class PDFValidationError(Exception):
    """Raised with a user-readable message when an upload cannot be used."""


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based
    text: str


@dataclass
class LoadedDocument:
    name: str
    doc_id: str
    total_pages: int
    pages: list[PageText]
    empty_pages: list[int] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


def load_pdf(data: bytes, filename: str, max_bytes: int, max_pages: int) -> LoadedDocument:
    """Validate and extract text from PDF bytes. Treats the input as untrusted."""
    name = os.path.basename(filename or "document.pdf")
    if not name.lower().endswith(".pdf"):
        raise PDFValidationError("Only .pdf files are supported.")
    if not data:
        raise PDFValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise PDFValidationError(
            f"File is too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Limit is {max_bytes / 1024 / 1024:.0f} MB."
        )
    if b"%PDF-" not in data[:1024]:
        raise PDFValidationError("This file does not look like a valid PDF (missing PDF header).")

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise PDFValidationError("PyMuPDF is not installed.") from exc

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFValidationError("The PDF is corrupt or could not be opened.") from exc

    try:
        if doc.needs_pass:
            raise PDFValidationError("Password-protected PDFs are not supported.")
        total = doc.page_count
        if total == 0:
            raise PDFValidationError("The PDF has no pages.")
        if total > max_pages:
            raise PDFValidationError(f"The PDF has {total} pages; the limit is {max_pages}.")

        pages: list[PageText] = []
        empty: list[int] = []
        for i in range(total):
            try:
                text = doc.load_page(i).get_text("text") or ""
            except Exception:
                text = ""
            text = text.strip()
            if text:
                pages.append(PageText(page_number=i + 1, text=text))
            else:
                empty.append(i + 1)
    except PDFValidationError:
        raise
    except Exception as exc:
        raise PDFValidationError("The PDF could not be read (possibly corrupt).") from exc
    finally:
        doc.close()

    if not pages:
        raise PDFValidationError(
            "No extractable text found. The PDF may be scanned images (OCR is not supported yet)."
        )

    doc_id = hashlib.sha256(data).hexdigest()[:12]
    return LoadedDocument(name=name, doc_id=doc_id, total_pages=total, pages=pages, empty_pages=empty)
