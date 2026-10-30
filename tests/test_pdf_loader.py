import pytest

from src.pdf_loader import PDFValidationError, load_pdf

fitz = pytest.importorskip("fitz")  # skipped if PyMuPDF is not installed
MB = 1024 * 1024


def _pdf(pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_valid_pdf_extracts_pages_with_numbers():
    d = load_pdf(_pdf(["First page text", "Second page text"]), "a.pdf", 5 * MB, 50)
    assert d.total_pages == 2 and [p.page_number for p in d.pages] == [1, 2]
    assert "Second" in d.pages[1].text and len(d.doc_id) == 12


def test_blank_page_in_middle_keeps_page_numbers():
    d = load_pdf(_pdf(["one", "", "three"]), "a.pdf", 5 * MB, 50)
    assert [p.page_number for p in d.pages] == [1, 3] and d.empty_pages == [2]


def test_all_blank_pdf_rejected():
    with pytest.raises(PDFValidationError):
        load_pdf(_pdf(["", ""]), "a.pdf", 5 * MB, 50)


def test_rejections():
    good = _pdf(["hello"])
    for data, name, mb, pages in [
        (b"", "a.pdf", 5 * MB, 50),
        (b"not a pdf at all", "a.pdf", 5 * MB, 50),
        (b"%PDF-1.4 garbage garbage", "a.pdf", 5 * MB, 50),
        (good, "a.txt", 5 * MB, 50),
        (good, "a.pdf", 10, 50),
        (_pdf(["a", "b", "c"]), "a.pdf", 5 * MB, 2),
    ]:
        with pytest.raises(PDFValidationError):
            load_pdf(data, name, mb, pages)


def test_filename_path_stripped():
    assert load_pdf(_pdf(["x"]), "../../evil/a.pdf", 5 * MB, 5).name == "a.pdf"
