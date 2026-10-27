"""Tests for end-to-end scoring and per-kind eligibility accuracy."""

import pytest

from scheme_assistant.assistant import Answer, SchemeResult
from scheme_assistant.eligibility import EligibilityDecision
from scheme_assistant.evaluate_eligibility import accuracy_by_kind
from scheme_assistant.evaluate_pipeline import score_case, summarise
from scheme_assistant.models import EligibilityCase


def make_case(verdict: str, kind: str = "basic") -> EligibilityCase:
    return EligibilityCase(
        id="e001",
        kind=kind,
        scheme_id="target",
        user_state=None,
        user_description="I am a 30-year-old farmer.",
        verdict=verdict,
        note="test",
    )


def make_answer(*verdicts_by_scheme: tuple[str, str]) -> Answer:
    """Build an answer from (scheme_id, verdict) pairs, in search order."""
    results = [
        SchemeResult(
            scheme_id=scheme_id,
            name=scheme_id,
            search_rank=rank,
            decision=EligibilityDecision(verdict=verdict, reason="", missing_information=[]),
        )
        for rank, (scheme_id, verdict) in enumerate(verdicts_by_scheme, start=1)
    ]
    return Answer(user_state=None, results=results)


# --- score_case -----------------------------------------------------------------


def test_found_with_correct_verdict_succeeds():
    outcome = score_case(
        make_case("eligible"), make_answer(("other", "eligible"), ("target", "eligible")), 1.0
    )
    assert outcome.success
    assert outcome.search_rank == 2
    assert outcome.eligible_count == 2


def test_found_with_wrong_verdict_fails():
    outcome = score_case(make_case("eligible"), make_answer(("target", "need_more_info")), 1.0)
    assert not outcome.success
    assert not outcome.false_eligible


@pytest.mark.parametrize("expected", ["eligible", "need_more_info"])
def test_missing_scheme_fails_when_user_should_see_it(expected: str):
    outcome = score_case(make_case(expected), make_answer(("other", "eligible")), 1.0)
    assert not outcome.found
    assert not outcome.success


def test_missing_scheme_succeeds_when_user_is_not_eligible():
    outcome = score_case(make_case("not_eligible"), make_answer(("other", "eligible")), 1.0)
    assert outcome.success


def test_shown_as_eligible_when_not_is_false_eligible():
    outcome = score_case(make_case("not_eligible"), make_answer(("target", "eligible")), 1.0)
    assert outcome.false_eligible
    assert not outcome.success


# --- summarise ------------------------------------------------------------------


def test_summarise_rates():
    outcomes = [
        score_case(make_case("eligible"), make_answer(("target", "eligible")), 2.0),
        score_case(make_case("eligible"), make_answer(("target", "not_eligible")), 2.0),
        score_case(make_case("eligible", kind="hinglish"), make_answer(), 2.0),
        score_case(make_case("not_eligible"), make_answer(), 2.0),
    ]
    summary = summarise(outcomes)
    assert summary["success"] == 0.5
    assert summary["found"] == 0.5
    assert summary["verdict_accuracy_when_found"] == 0.5
    assert summary["mean_seconds"] == 2.0
    assert summary["success_by_kind"] == {"basic": pytest.approx(2 / 3), "hinglish": 0.0}


# --- accuracy_by_kind -----------------------------------------------------------


def test_accuracy_by_kind():
    cases = [make_case("eligible"), make_case("eligible", kind="boundary"), make_case("eligible")]
    predicted = ["eligible", "not_eligible", "need_more_info"]
    assert accuracy_by_kind(cases, predicted) == {"basic": 0.5, "boundary": 0.0}
