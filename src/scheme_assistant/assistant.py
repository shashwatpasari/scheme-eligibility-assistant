"""End-to-end assistant: from a user's question to the schemes they may qualify for.

Steps:
    1. If the user's state is not given, read it from the question with the LLM.
    2. Search for matching schemes using the original question. (Rewriting the
       question before searching did not improve results; see data/eval/README.md.)
    3. Check the user against each scheme found.
    4. Return the results with eligible schemes first.

Usage (requires a built index and GEMINI_API_KEY):
    uv run python -m scheme_assistant.assistant "I am a 55-year-old widow in Leh with no income"
    uv run python -m scheme_assistant.assistant "I farm 2 acres" --state Bihar
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass

from qdrant_client import QdrantClient

from scheme_assistant.eligibility import EligibilityDecision, check_eligibility
from scheme_assistant.llm import DEFAULT_GEMINI_MODEL, GeminiClient, LLMClient
from scheme_assistant.rewrite import rewrite_query
from scheme_assistant.search import QDRANT_PATH, BgeM3Embedder, load_scheme, search

SCHEMES_TO_CHECK = 10  # the right scheme is in the top 10 for 93% of test queries
PARALLEL_CHECKS = 5  # eligibility checks are independent, so run several at once
VERDICT_ORDER = {"eligible": 0, "need_more_info": 1, "not_eligible": 2}


@dataclass
class SchemeResult:
    """One scheme found for the user, with the eligibility decision for it."""

    scheme_id: str
    name: str
    decision: EligibilityDecision


@dataclass
class Answer:
    """Everything the assistant found for one question."""

    user_state: str | None
    results: list[SchemeResult]  # eligible first, then need_more_info, then not_eligible


def answer_question(
    llm: LLMClient,
    client: QdrantClient,
    embedder: BgeM3Embedder,
    question: str,
    user_state: str | None = None,
    limit: int = SCHEMES_TO_CHECK,
) -> Answer:
    """Find schemes for a user and check their eligibility for each.

    Args:
        llm: Client used for the state lookup and the eligibility checks.
        client: Qdrant client holding the scheme index.
        embedder: Embedder used to encode the question.
        question: The user's question in their own words.
        user_state: The state the user lives in; read from the question if None.
        limit: Number of schemes to retrieve and check.

    Returns:
        The state used and the checked schemes, eligible first.
    """
    if user_state is None:
        user_state = rewrite_query(llm, question).state

    hits = search(client, embedder, question, user_state=user_state, limit=limit)

    def check(scheme_id: str) -> EligibilityDecision:
        return check_eligibility(llm, load_scheme(scheme_id), question, user_state)

    with ThreadPoolExecutor(max_workers=PARALLEL_CHECKS) as pool:
        decisions = list(pool.map(check, [hit.scheme_id for hit in hits]))

    results = [
        SchemeResult(scheme_id=hit.scheme_id, name=hit.name, decision=decision)
        for hit, decision in zip(hits, decisions)
    ]
    # sort() is stable, so schemes with the same verdict keep their search order.
    results.sort(key=lambda result: VERDICT_ORDER[result.decision.verdict])
    return Answer(user_state=user_state, results=results)


def print_answer(answer: Answer) -> None:
    """Print the results grouped by verdict."""
    print(f"State: {answer.user_state or 'not given (central schemes only)'}")
    headings = {
        "eligible": "Eligible",
        "need_more_info": "Need more information",
        "not_eligible": "Not eligible",
    }
    for verdict, heading in headings.items():
        group = [result for result in answer.results if result.decision.verdict == verdict]
        if not group:
            continue
        print(f"\n{heading}")
        for result in group:
            print(f"  - {result.name} ({result.scheme_id})")
            print(f"    {result.decision.reason}")
            for question in result.decision.missing_information:
                print(f"    ? {question}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find schemes a user may qualify for.")
    parser.add_argument("question", help="the user's question")
    parser.add_argument("--state", help="the state the user lives in")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name")
    parser.add_argument("--limit", type=int, default=SCHEMES_TO_CHECK, help="schemes to check")
    args = parser.parse_args()

    llm = GeminiClient(args.model)
    with closing(QdrantClient(path=str(QDRANT_PATH))) as client:
        answer = answer_question(
            llm, client, BgeM3Embedder(), args.question, args.state, args.limit
        )
    print_answer(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
