"""Eligibility check: decide whether a user qualifies for one scheme.

An LLM reads the scheme's own text and the user's description, and returns one of:
    eligible        What the user said covers every key condition.
    not_eligible    Something the user said breaks a condition.
    need_more_info  Nothing is broken, but a key condition is not covered yet.

Usage:
    uv run python -m scheme_assistant.eligibility pm-kisan "I farm 3 acres in my name"
"""

import argparse

from pydantic import BaseModel

from scheme_assistant.llm import DEFAULT_MODEL, LLMClient, make_llm
from scheme_assistant.models import Scheme, Verdict
from scheme_assistant.search import load_scheme

SYSTEM_PROMPT = """\
You check whether a person qualifies for an Indian government scheme, using only the \
scheme text provided. Do not use outside knowledge about the scheme.

Choose one verdict:
- not_eligible: something the person said breaks at least one eligibility condition.
- need_more_info: nothing the person said breaks a condition, but at least one key \
condition (such as age, income, caste or category, occupation, land, disability, \
education, or registration with a board) is not covered by what they said.
- eligible: what the person said covers every key condition and breaks none.

Assume routine conditions are met unless the person says otherwise: Indian \
citizenship; having Aadhaar, a bank account, a mobile phone or the required documents; \
registering on a portal; and exclusions such as not already receiving this or a \
similar benefit, or not being a government employee. The person lives in the state \
given, which covers residence conditions for that state.

Return:
- verdict
- reason: one or two sentences naming the conditions that decided the verdict.
- missing_information: for need_more_info, short questions to ask the person, \
e.g. "What is your age?". Otherwise an empty list.
"""


class EligibilityDecision(BaseModel):
    """The LLM's verdict for one user and one scheme."""

    verdict: Verdict
    reason: str
    missing_information: list[str]


def build_prompt(scheme: Scheme, user_description: str, user_state: str | None) -> str:
    """Combine the scheme text and the user's details into one prompt."""
    applies_in = scheme.state or "all of India (central scheme)"
    return (
        f"Scheme: {scheme.name}\n"
        f"Applies in: {applies_in}\n\n"
        f"Details:\n{scheme.details}\n\n"
        f"Eligibility:\n{scheme.eligibility}\n\n"
        "---\n"
        f"Person's state: {user_state or 'not given'}\n"
        f"Person's description: {user_description}"
    )


def check_eligibility(
    llm: LLMClient, scheme: Scheme, user_description: str, user_state: str | None = None
) -> EligibilityDecision:
    """Ask the LLM whether the user qualifies for the scheme.

    Args:
        llm: Client used to call the model.
        scheme: The scheme to check against.
        user_description: The user's situation in their own words.
        user_state: The state the user lives in, if known.

    Returns:
        The verdict, the reason for it, and any questions to ask the user.
    """
    prompt = build_prompt(scheme, user_description, user_state)
    return llm.generate(SYSTEM_PROMPT, prompt, EligibilityDecision)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a user against one scheme.")
    parser.add_argument("scheme_id", help="scheme id, e.g. pm-kisan")
    parser.add_argument("description", help="the user's situation")
    parser.add_argument("--state", help="the state the user lives in")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model name (Gemini, Gemma or Claude)"
    )
    args = parser.parse_args()

    scheme = load_scheme(args.scheme_id)
    decision = check_eligibility(make_llm(args.model), scheme, args.description, args.state)
    print(decision.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
