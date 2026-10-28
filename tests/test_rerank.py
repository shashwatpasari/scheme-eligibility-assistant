"""Tests for re-ranking, using fake re-rankers."""

import pytest

from scheme_assistant import rerank as rerank_module
from scheme_assistant.rerank import rerank
from scheme_assistant.search import SearchHit
from tests.fake_reranker import ReverseReranker, WordOverlapReranker
from tests.test_search import SCHEMES

SCHEMES_BY_ID = {scheme.id: scheme for scheme in SCHEMES}


@pytest.fixture(autouse=True)
def schemes_from_memory(monkeypatch):
    # The test schemes exist only in memory, not in data/schemes.
    monkeypatch.setattr(rerank_module, "load_scheme", SCHEMES_BY_ID.__getitem__)


def search_order(*scheme_ids: str) -> list[SearchHit]:
    """Hits in the given order, as search would return them."""
    return [SearchHit(scheme_id, scheme_id, score=0.0) for scheme_id in scheme_ids]


def test_rerank_orders_by_reranker_score():
    hits = search_order("farmer-cash", "street-vendor-loan", "widow-pension-ladakh")
    reranked = rerank(WordOverlapReranker(), "widow with no income", hits, limit=10)
    assert reranked[0].scheme_id == "widow-pension-ladakh"


def test_rerank_keeps_only_the_limit():
    hits = search_order("farmer-cash", "street-vendor-loan", "widow-pension-ladakh")
    reranked = rerank(ReverseReranker(), "any question", hits, limit=2)
    assert [hit.scheme_id for hit in reranked] == ["widow-pension-ladakh", "street-vendor-loan"]


def test_rerank_attaches_reranker_scores():
    hits = search_order("farmer-cash", "street-vendor-loan")
    reranked = rerank(ReverseReranker(), "any question", hits, limit=10)
    assert [hit.score for hit in reranked] == [1.0, 0.0]


def test_rerank_handles_no_hits():
    assert rerank(WordOverlapReranker(), "widow", [], limit=10) == []
