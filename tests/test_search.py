"""Tests for scheme retrieval using a fake embedder and an in-memory Qdrant."""

import pytest
from qdrant_client import QdrantClient

from scheme_assistant.models import Scheme
from scheme_assistant.search import (
    MODES,
    SearchHit,
    build_index,
    scheme_text,
    search,
    weighted_rrf,
)
from tests.fake_embedder import FakeEmbedder


def hits(*scheme_ids: str) -> list[SearchHit]:
    """Build a ranked hit list from scheme ids, best first."""
    return [SearchHit(scheme_id=scheme_id, name=scheme_id, score=0.0) for scheme_id in scheme_ids]


def fused_ids(dense: list[SearchHit], sparse: list[SearchHit], dense_weight: float) -> list[str]:
    return [hit.scheme_id for hit in weighted_rrf(dense, sparse, dense_weight, limit=10)]


def test_weighted_rrf_equal_weights_rewards_agreement():
    # "b" is 2nd in both lists, so it beats "a" and "c", which each top only one list.
    assert fused_ids(hits("a", "b"), hits("c", "b"), 0.5)[0] == "b"


def test_weighted_rrf_score_matches_formula():
    fused = weighted_rrf(hits("a"), hits("a"), 0.8, limit=10)
    assert fused[0].score == pytest.approx(0.8 / 2 + 0.2 / 2)


def test_weighted_rrf_full_dense_weight_keeps_dense_order():
    assert fused_ids(hits("a", "b", "c"), hits("c", "b", "a"), 1.0) == ["a", "b", "c"]


def test_weighted_rrf_zero_dense_weight_keeps_sparse_order():
    assert fused_ids(hits("a", "b", "c"), hits("c", "b", "a"), 0.0) == ["c", "b", "a"]


def test_weighted_rrf_heavier_weight_wins_disagreement():
    dense, sparse = hits("a", "b"), hits("b", "a")
    assert fused_ids(dense, sparse, 0.8)[0] == "a"
    assert fused_ids(dense, sparse, 0.2)[0] == "b"


def test_weighted_rrf_respects_limit():
    assert len(weighted_rrf(hits("a", "b", "c"), hits("d", "e"), 0.5, limit=2)) == 2


@pytest.mark.parametrize("bad_weight", [-0.1, 1.5])
def test_weighted_rrf_rejects_out_of_range_weight(bad_weight: float):
    with pytest.raises(ValueError, match="dense_weight"):
        weighted_rrf(hits("a"), hits("a"), bad_weight, limit=10)


def make_scheme(scheme_id: str, eligibility: str, state: str | None = None) -> Scheme:
    """Build a minimal scheme; central unless a state is given."""
    return Scheme.model_validate(
        {
            "id": scheme_id,
            "name": scheme_id.upper(),
            "level": "state" if state else "central",
            "state": state,
            "details": "A government scheme.",
            "benefits": "Financial help.",
            "eligibility": eligibility,
            "documents": [],
            "how_to_apply": "Apply online.",
            "source_url": f"https://www.myscheme.gov.in/schemes/{scheme_id}",
            "fetched_on": "2026-07-05",
        }
    )


SCHEMES = [
    make_scheme("farmer-cash", "farmers who own cultivable land"),
    make_scheme("street-vendor-loan", "street vendors selling from a cart"),
    make_scheme("widow-pension-ladakh", "widow with no income and bpl card", state="Ladakh"),
    make_scheme("widow-pension-goa", "widow with no income and bpl card", state="Goa"),
]


@pytest.fixture
def client() -> QdrantClient:
    client = QdrantClient(":memory:")
    build_index(client, FakeEmbedder(), SCHEMES)
    return client


def retrieve_ids(
    client: QdrantClient, query: str, mode: str, user_state: str | None = None
) -> list[str]:
    hits = search(client, FakeEmbedder(), query, mode, user_state)
    return [hit.scheme_id for hit in hits]


def test_build_index_returns_indexed_count():
    assert build_index(QdrantClient(":memory:"), FakeEmbedder(), SCHEMES) == len(SCHEMES)


def test_scheme_text_includes_targeting_fields():
    text = scheme_text(SCHEMES[0])
    assert "FARMER-CASH" in text
    assert "cultivable land" in text
    assert "Financial help." in text


@pytest.mark.parametrize("mode", MODES)
def test_best_match_ranks_first(client: QdrantClient, mode: str):
    retrieved = retrieve_ids(client, "I sell vegetables from a cart as a street vendor", mode)
    assert retrieved[0] == "street-vendor-loan"


@pytest.mark.parametrize("mode", MODES)
def test_state_filter_keeps_central_and_own_state(client: QdrantClient, mode: str):
    retrieved = retrieve_ids(client, "widow farmers with no income", mode, user_state="Ladakh")
    assert "widow-pension-ladakh" in retrieved
    assert "farmer-cash" in retrieved
    assert "widow-pension-goa" not in retrieved


@pytest.mark.parametrize("mode", MODES)
def test_no_state_searches_all_states(client: QdrantClient, mode: str):
    retrieved = retrieve_ids(client, "widow with no income", mode)
    assert "widow-pension-ladakh" in retrieved
    assert "widow-pension-goa" in retrieved


def test_unknown_mode_raises(client: QdrantClient):
    with pytest.raises(ValueError, match="Unknown mode"):
        retrieve_ids(client, "widow", "magic")


def test_rebuild_replaces_existing_points(client: QdrantClient):
    build_index(client, FakeEmbedder(), SCHEMES[:1])
    assert retrieve_ids(client, "widow", "dense") == ["farmer-cash"]
