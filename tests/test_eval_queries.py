"""Checks that the search test set is valid and matches the scheme data."""

import json
from pathlib import Path

import pytest

from scheme_assistant.models import RetrievalQuery, Scheme

DATA_DIR = Path(__file__).parent.parent / "data"
QUERIES_PATH = DATA_DIR / "eval" / "retrieval_queries.jsonl"


def load_queries() -> list[RetrievalQuery]:
    lines = QUERIES_PATH.read_text(encoding="utf-8").splitlines()
    return [RetrievalQuery.model_validate_json(line) for line in lines if line.strip()]


def load_scheme(scheme_id: str) -> Scheme | None:
    path = DATA_DIR / "schemes" / f"{scheme_id}.json"
    return Scheme.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


QUERIES = load_queries()


def test_query_ids_are_unique():
    ids = [q.id for q in QUERIES]
    assert len(ids) == len(set(ids))


def test_every_line_is_valid_json():
    for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines():
        json.loads(line)


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.id)
def test_relevant_schemes_exist_and_fit_user_state(query: RetrievalQuery):
    for scheme_id in query.relevant:
        scheme = load_scheme(scheme_id)
        assert scheme is not None, f"{scheme_id} not in data/schemes"
        # A state filter would hide this scheme if the user lived somewhere else.
        if scheme.state is not None:
            assert scheme.state == query.user_state, f"{scheme_id} is for {scheme.state}"
