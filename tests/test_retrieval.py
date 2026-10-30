import numpy as np

from src.retrieval import Retriever, VectorIndex
from tests.helpers import SAMPLE_CHUNKS, HashingEmbedder, make_chunk


def _retriever(chunks=SAMPLE_CHUNKS):
    r = Retriever(HashingEmbedder())
    r.build(chunks)
    return r


def test_known_question_retrieves_correct_page():
    r = _retriever()
    assert r.retrieve("how does photosynthesis use chlorophyll", k=1)[0].chunk.page == 1
    assert r.retrieve("when did the French Revolution begin", k=1)[0].chunk.page == 2
    assert r.retrieve("mutable lists versus tuples", k=1)[0].chunk.page == 3


def test_irrelevant_query_scores_low():
    res = _retriever().retrieve("quantum chromodynamics gluon", k=1)
    assert res[0].score < 0.1


def test_scores_sorted_and_ranks():
    res = _retriever().retrieve("python tuples", k=3)
    assert [x.rank for x in res] == [1, 2, 3]
    assert res[0].score >= res[1].score >= res[2].score


def test_k_larger_than_corpus():
    assert len(_retriever().retrieve("python", k=50)) == 3


def test_empty_query_and_unbuilt_retriever():
    assert _retriever().retrieve("   ", k=3) == []
    assert Retriever(HashingEmbedder()).retrieve("python", k=3) == []


def test_metadata_preserved():
    top = _retriever().retrieve("French Revolution", k=1)[0].chunk
    assert (top.doc_id, top.page, top.doc_name) == ("d1", 2, "doc.pdf")


def test_sessions_are_isolated():
    a = _retriever([make_chunk("A", 1, "apples are red fruit")])
    b = _retriever([make_chunk("B", 7, "bananas are yellow fruit")])
    assert {x.chunk.doc_id for x in a.retrieve("fruit", 5)} == {"A"}
    assert {x.chunk.doc_id for x in b.retrieve("fruit", 5)} == {"B"}


def test_vector_index_exact_cosine():
    v = np.eye(3, dtype=np.float32)
    hits = VectorIndex(v).search(np.array([0, 1, 0], dtype=np.float32), 2)
    assert hits[0][0] == 1 and abs(hits[0][1] - 1.0) < 1e-6
