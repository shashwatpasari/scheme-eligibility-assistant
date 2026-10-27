"""Tests for query rewriting and its evaluation helpers."""

import json
from pathlib import Path

import pytest

from scheme_assistant.evaluate_rewrite import check_states, search_text
from scheme_assistant.models import RetrievalQuery
from scheme_assistant.rewrite import (
    STATES,
    SYSTEM_PROMPT,
    QueryRewrite,
    normalise_state,
    rewrite_query,
)
from tests.fake_llm import FakeLLM

SCHEMES_DIR = Path(__file__).parent.parent / "data" / "schemes"


def make_rewrite(state: str | None = None) -> QueryRewrite:
    return QueryRewrite(search_query="pension for a widow", state=state, facts=["widow"])


def make_query(user_state: str | None) -> RetrievalQuery:
    return RetrievalQuery(
        id="q001", query="widow pension", user_state=user_state, style="keyword", relevant=["x"]
    )


# --- normalise_state ----------------------------------------------------------


def test_normalise_state_ignores_case_and_spaces():
    assert normalise_state("  tamil nadu ") == "Tamil Nadu"


def test_normalise_state_rejects_unknown_names():
    assert normalise_state("Atlantis") is None


def test_normalise_state_keeps_none():
    assert normalise_state(None) is None


def test_states_match_scheme_data():
    scheme_files = list(SCHEMES_DIR.glob("*.json"))
    if not scheme_files:
        pytest.skip("scheme data not built")
    data_states = {json.loads(path.read_text())["state"] for path in scheme_files} - {None}
    assert data_states == set(STATES)


# --- rewrite_query --------------------------------------------------------------


def test_rewrite_query_sends_question_to_llm():
    llm = FakeLLM(make_rewrite())
    rewrite_query(llm, "Main vidhwa hoon")
    assert llm.prompts == [(SYSTEM_PROMPT, "Main vidhwa hoon")]


def test_rewrite_query_normalises_state():
    assert rewrite_query(FakeLLM(make_rewrite("kerala")), "q").state == "Kerala"


def test_rewrite_query_drops_unknown_state():
    assert rewrite_query(FakeLLM(make_rewrite("Atlantis")), "q").state is None


def test_system_prompt_lists_every_state():
    assert all(state in SYSTEM_PROMPT for state in STATES)


# --- evaluation helpers ---------------------------------------------------------


def test_search_text_variants():
    rewrite = make_rewrite()
    assert search_text("original", "vidhwa pension", rewrite) == "vidhwa pension"
    assert search_text("rewrite", "vidhwa pension", rewrite) == "pension for a widow"
    assert search_text("original + rewrite", "vidhwa pension", rewrite) == (
        "vidhwa pension\npension for a widow"
    )


def test_search_text_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        search_text("summary", "q", make_rewrite())


def test_check_states_counts_correct_none_and_wrong():
    queries = [make_query("Kerala"), make_query("Kerala"), make_query("Goa")]
    rewrites = [make_rewrite("Kerala"), make_rewrite(None), make_rewrite("Delhi")]
    assert check_states(queries, rewrites) == {"correct": 1, "none": 1, "wrong": 1}
