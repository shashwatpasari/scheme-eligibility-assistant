"""Tests for retrieval metrics and evaluation setups."""

import pytest

from scheme_assistant.evaluate import (
    HYBRID_DENSE_WEIGHTS,
    RERANK_CANDIDATE_COUNTS,
    Setup,
    build_setups,
    compute_metrics,
    first_relevant_rank,
)


def test_first_relevant_rank_returns_position():
    assert first_relevant_rank(["a", "b", "c"], ["c"]) == 3


def test_first_relevant_rank_uses_earliest_match():
    assert first_relevant_rank(["a", "b", "c"], ["c", "b"]) == 2


def test_first_relevant_rank_returns_none_when_absent():
    assert first_relevant_rank(["a", "b"], ["z"]) is None


def test_compute_metrics():
    metrics = compute_metrics([1, 2, 6, None])
    assert metrics["hit@1"] == 0.25
    assert metrics["hit@5"] == 0.5
    assert metrics["hit@10"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 6) / 4)


def test_setup_name_for_single_retrievers():
    assert Setup("dense", use_state_filter=False).name() == "dense"
    assert Setup("sparse", use_state_filter=True).name() == "sparse + state filter"


def test_setup_name_shows_hybrid_weights():
    assert Setup("hybrid", False, dense_weight=0.5).name() == "hybrid 50/50"
    assert Setup("hybrid", True, dense_weight=0.8).name() == "hybrid 80/20 + state filter"


def test_setup_name_shows_rerank_candidates():
    setup = Setup("hybrid", True, rerank_candidates=20)
    assert setup.name() == "hybrid 70/30 + state filter + rerank 20"


def test_build_setups_sweeps_hybrid_weights_with_state_filter():
    swept = [
        setup.dense_weight
        for setup in build_setups()
        if setup.mode == "hybrid" and setup.use_state_filter and setup.rerank_candidates is None
    ]
    assert swept == HYBRID_DENSE_WEIGHTS


def test_build_setups_ends_with_rerank_setups():
    reranked = [setup.rerank_candidates for setup in build_setups() if setup.rerank_candidates]
    assert reranked == RERANK_CANDIDATE_COUNTS
