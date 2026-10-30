import pytest

from src.evaluation import mean, recall_at_k, reciprocal_rank


def test_recall_and_mrr():
    assert recall_at_k([4, 2, 9], {2}, 1) == 0.0
    assert recall_at_k([4, 2, 9], {2}, 2) == 1.0
    assert reciprocal_rank([4, 2, 9], {2}) == 0.5
    assert reciprocal_rank([4, 9], {2}) == 0.0
    assert mean([1.0, 0.0]) == 0.5 and mean([]) == 0.0


def test_empty_relevance_rejected():
    with pytest.raises(ValueError):
        recall_at_k([1], set(), 1)
