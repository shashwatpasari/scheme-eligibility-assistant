"""Tests for the eligibility check, its evaluation metrics and the labelled cases."""

from collections import Counter

import pytest

from scheme_assistant.eligibility import (
    SYSTEM_PROMPT,
    EligibilityDecision,
    build_prompt,
    check_eligibility,
)
from scheme_assistant.evaluate import SPLITS
from scheme_assistant.evaluate_eligibility import compute_metrics, load_cases
from scheme_assistant.models import EligibilityCase
from scheme_assistant.search import SCHEMES_DIR, load_scheme
from tests.fake_llm import FakeLLM
from tests.test_search import make_scheme

CASES_BY_SPLIT = {split: load_cases(split) for split in SPLITS}
ALL_CASES = [case for split in SPLITS for case in CASES_BY_SPLIT[split]]


# --- check_eligibility ----------------------------------------------------------


def test_build_prompt_includes_scheme_and_user():
    scheme = make_scheme("widow-pension", "widow aged 40-79", state="Ladakh")
    prompt = build_prompt(scheme, "I am a 55-year-old widow", "Ladakh")
    assert "Scheme: WIDOW-PENSION" in prompt
    assert "Applies in: Ladakh" in prompt
    assert "widow aged 40-79" in prompt
    assert "Person's state: Ladakh" in prompt
    assert "I am a 55-year-old widow" in prompt


def test_build_prompt_marks_central_schemes_and_missing_state():
    prompt = build_prompt(make_scheme("farmer-cash", "farmers"), "I farm", None)
    assert "Applies in: all of India (central scheme)" in prompt
    assert "Person's state: not given" in prompt


def test_check_eligibility_returns_llm_decision():
    decision = EligibilityDecision(verdict="eligible", reason="Meets age.", missing_information=[])
    llm = FakeLLM(decision)
    result = check_eligibility(llm, make_scheme("farmer-cash", "farmers"), "I farm", "Bihar")
    assert result == decision
    assert llm.prompts[0][0] == SYSTEM_PROMPT


# --- metrics --------------------------------------------------------------------


def test_compute_metrics_accuracy_and_recall():
    true = ["eligible", "eligible", "not_eligible", "need_more_info"]
    predicted = ["eligible", "need_more_info", "not_eligible", "need_more_info"]
    metrics = compute_metrics(true, predicted)
    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == {"eligible": 0.5, "not_eligible": 1.0, "need_more_info": 1.0}
    assert metrics["confusion_matrix"]["eligible"]["need_more_info"] == 1


def test_compute_metrics_counts_false_eligible():
    true = ["not_eligible", "need_more_info", "eligible"]
    predicted = ["eligible", "eligible", "eligible"]
    assert compute_metrics(true, predicted)["false_eligible"] == 2


# --- labelled cases -------------------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_each_split_is_balanced(split: str):
    counts = Counter(case.verdict for case in CASES_BY_SPLIT[split])
    assert counts == {"eligible": 14, "not_eligible": 14, "need_more_info": 14}


def test_case_ids_are_unique_across_splits():
    case_ids = [case.id for case in ALL_CASES]
    assert len(case_ids) == len(set(case_ids))


def test_splits_do_not_share_schemes():
    dev_schemes = {case.scheme_id for case in CASES_BY_SPLIT["dev"]}
    test_schemes = {case.scheme_id for case in CASES_BY_SPLIT["test"]}
    assert dev_schemes.isdisjoint(test_schemes)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.id)
def test_case_scheme_exists_and_matches_state(case: EligibilityCase):
    if not (SCHEMES_DIR / f"{case.scheme_id}.json").exists():
        pytest.fail(f"{case.scheme_id} not in data/schemes")
    scheme = load_scheme(case.scheme_id)
    if scheme.state is not None:
        assert case.user_state == scheme.state
