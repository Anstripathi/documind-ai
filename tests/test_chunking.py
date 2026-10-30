import pytest

from src.chunking import chunk_document, split_text
from src.pdf_loader import LoadedDocument, PageText


def _doc(pages):
    return LoadedDocument("t.pdf", "abc123", len(pages), [PageText(n, t) for n, t in pages])


def test_short_text_single_chunk():
    assert split_text("hello   world", 100, 10) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n  ", 100, 10) == []


def test_long_text_overlaps_and_covers():
    text = " ".join(f"w{i}" for i in range(400))
    pieces = split_text(text, 200, 50)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    assert pieces[0].startswith("w0")
    assert pieces[-1].endswith("w399")
    # consecutive chunks share some text (overlap)
    assert any(tok in pieces[1] for tok in pieces[0].split()[-3:])


def test_no_space_text_still_terminates():
    pieces = split_text("x" * 5000, 300, 50)
    assert len(pieces) > 10
    assert all(len(p) <= 300 for p in pieces)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text("abc", 0, 0)
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        split_text("abc", 100, -1)


def test_page_provenance_and_stable_ids():
    doc = _doc([(1, "alpha " * 300), (4, "beta short")])
    a = chunk_document(doc, 200, 40)
    b = chunk_document(doc, 200, 40)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]  # deterministic
    assert len({c.chunk_id for c in a}) == len(a)              # unique
    assert {c.page for c in a} == {1, 4}
    assert all("beta" not in c.text for c in a if c.page == 1)  # never crosses pages
    assert all(c.doc_id == "abc123" for c in a)
