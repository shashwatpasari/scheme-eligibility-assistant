"""Checks that the evaluation query sets are valid and consistent with the scheme data."""

from pathlib import Path

import pytest

from scheme_assistant.evaluate import SPLITS, load_queries
from scheme_assistant.models import RetrievalQuery, Scheme

SCHEMES_DIR = Path(__file__).parent.parent / "data" / "schemes"
QUERIES_BY_SPLIT = {split: load_queries(split) for split in SPLITS}
ALL_QUERIES = [query for split in SPLITS for query in QUERIES_BY_SPLIT[split]]


def load_scheme(scheme_id: str) -> Scheme | None:
    path = SCHEMES_DIR / f"{scheme_id}.json"
    if not path.exists():
        return None
    return Scheme.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("split", SPLITS)
def test_each_split_has_40_queries(split: str):
    assert len(QUERIES_BY_SPLIT[split]) == 40


def test_query_ids_are_unique_across_splits():
    query_ids = [query.id for query in ALL_QUERIES]
    assert len(query_ids) == len(set(query_ids))


def test_splits_do_not_share_target_schemes():
    dev_targets = {scheme_id for q in QUERIES_BY_SPLIT["dev"] for scheme_id in q.relevant}
    test_targets = {scheme_id for q in QUERIES_BY_SPLIT["test"] for scheme_id in q.relevant}
    assert dev_targets.isdisjoint(test_targets)


@pytest.mark.parametrize("query", ALL_QUERIES, ids=lambda query: query.id)
def test_relevant_schemes_exist_and_match_user_state(query: RetrievalQuery):
    for scheme_id in query.relevant:
        scheme = load_scheme(scheme_id)
        assert scheme is not None, f"{scheme_id} not in data/schemes"
        # The state filter would hide a state scheme from users in other states.
        if scheme.state is not None:
            assert scheme.state == query.user_state, f"{scheme_id} belongs to {scheme.state}"
