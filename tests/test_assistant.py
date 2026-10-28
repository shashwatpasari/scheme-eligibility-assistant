"""Tests for the end-to-end assistant, with a fake LLM, fake embedder and in-memory Qdrant."""

import pytest
from qdrant_client import QdrantClient

from scheme_assistant import assistant
from scheme_assistant import rerank as rerank_module
from scheme_assistant.assistant import answer_question
from scheme_assistant.eligibility import EligibilityDecision
from scheme_assistant.llm import LLMError
from scheme_assistant.rewrite import QueryRewrite
from scheme_assistant.search import build_index
from tests.fake_embedder import FakeEmbedder
from tests.fake_reranker import ReverseReranker
from tests.test_search import SCHEMES

SCHEMES_BY_ID = {scheme.id: scheme for scheme in SCHEMES}


class VerdictByScheme:
    """Fake LLM that finds the state "Ladakh" and gives a fixed verdict per scheme."""

    model = "fake-llm"

    def __init__(self, verdicts: dict[str, str]) -> None:
        self.verdicts = verdicts  # scheme name -> verdict
        self.rewrite_calls = 0

    def generate(self, system_prompt, user_prompt, response_model):
        if response_model is QueryRewrite:
            self.rewrite_calls += 1
            return QueryRewrite(search_query=user_prompt, state="Ladakh", facts=[])
        scheme_name = user_prompt.split("\n")[0].removeprefix("Scheme: ")
        if self.verdicts.get(scheme_name) == "no_reply":
            raise LLMError("no valid reply")
        return EligibilityDecision(
            verdict=self.verdicts.get(scheme_name, "need_more_info"),
            reason="test",
            missing_information=[],
        )


@pytest.fixture
def client() -> QdrantClient:
    client = QdrantClient(":memory:")
    build_index(client, FakeEmbedder(), SCHEMES)
    return client


@pytest.fixture(autouse=True)
def schemes_from_memory(monkeypatch):
    # The test schemes exist only in memory, not in data/schemes.
    monkeypatch.setattr(assistant, "load_scheme", SCHEMES_BY_ID.__getitem__)
    monkeypatch.setattr(rerank_module, "load_scheme", SCHEMES_BY_ID.__getitem__)


def test_state_is_read_from_question_when_not_given(client):
    llm = VerdictByScheme({})
    answer = answer_question(llm, client, FakeEmbedder(), "widow with no income")
    assert answer.user_state == "Ladakh"
    assert llm.rewrite_calls == 1
    # The Goa scheme is filtered out because the user lives in Ladakh.
    assert "widow-pension-goa" not in [result.scheme_id for result in answer.results]


def test_given_state_skips_the_llm_lookup(client):
    llm = VerdictByScheme({})
    answer = answer_question(llm, client, FakeEmbedder(), "widow with no income", "Goa")
    assert answer.user_state == "Goa"
    assert llm.rewrite_calls == 0


def test_results_are_sorted_by_verdict(client):
    llm = VerdictByScheme(
        {
            "FARMER-CASH": "not_eligible",
            "STREET-VENDOR-LOAN": "eligible",
            "WIDOW-PENSION-LADAKH": "need_more_info",
        }
    )
    answer = answer_question(llm, client, FakeEmbedder(), "widow farmer vendor", "Ladakh")
    verdicts = [result.decision.verdict for result in answer.results]
    assert verdicts == ["eligible", "need_more_info", "not_eligible"]
    # Search ranks are kept, so the original search order is still known.
    assert sorted(result.search_rank for result in answer.results) == [1, 2, 3]


def test_scheme_without_valid_reply_is_left_out(client):
    llm = VerdictByScheme({"FARMER-CASH": "no_reply"})
    answer = answer_question(llm, client, FakeEmbedder(), "widow farmer vendor", "Ladakh")
    assert "farmer-cash" not in [result.scheme_id for result in answer.results]
    assert len(answer.results) == 2


def scheme_ids_by_rank(answer) -> list[str]:
    results = sorted(answer.results, key=lambda result: result.search_rank)
    return [result.scheme_id for result in results]


def test_reranker_sets_the_order_and_can_bring_in_deeper_results(client):
    llm = VerdictByScheme({})
    question = "widow farmer vendor"
    plain = answer_question(llm, client, FakeEmbedder(), question, "Ladakh", limit=3)
    reranked = answer_question(
        llm, client, FakeEmbedder(), question, "Ladakh", limit=2, reranker=ReverseReranker()
    )
    # Search order reversed; the third search result is now first, despite limit=2.
    assert scheme_ids_by_rank(reranked) == scheme_ids_by_rank(plain)[::-1][:2]
