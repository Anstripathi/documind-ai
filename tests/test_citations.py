from src.citations import REMOVED_MARK, extract_cited_pages, validate_citations


def test_extract_pages():
    assert extract_cited_pages("A [Page 3]. B [Page 5, Page 7]. C [p. 3]") == [3, 5, 7]


def test_non_page_brackets_ignored():
    r = validate_citations("See ref [1] and [note].", {1})
    assert r.cited_pages == [] and r.text == "See ref [1] and [note]."


def test_valid_citation_kept():
    r = validate_citations("Fact [Page 2].", {2, 3})
    assert r.text == "Fact [Page 2]." and r.valid_pages == [2] and r.invalid_pages == []
    assert r.has_valid_citation


def test_invented_page_removed():
    r = validate_citations("Fact [Page 99].", {2})
    assert r.invalid_pages == [99] and REMOVED_MARK in r.text and "99" not in r.text
    assert not r.has_valid_citation


def test_mixed_citation_keeps_only_valid():
    r = validate_citations("Fact [Page 2, Page 50].", {2})
    assert r.text == "Fact [Page 2]." and r.invalid_pages == [50]


def test_multi_page_list_form():
    assert extract_cited_pages("x [Pages 3, 5]") == [3, 5]
