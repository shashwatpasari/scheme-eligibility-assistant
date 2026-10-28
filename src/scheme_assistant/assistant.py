"""End-to-end assistant: from a user's question to the schemes they may qualify for.

Steps:
    1. If the user's state is not given, read it from the question with the LLM.
    2. Search for matching schemes using the original question. (Rewriting the
       question before searching did not improve results; see data/eval/README.md.)
    3. Re-rank the search results with a cross-encoder (can be turned off).
    4. Check the user against each scheme found.
    5. Return the results with eligible schemes first.

Usage (requires a built index and an API key in .env):
    uv run python -m scheme_assistant.assistant "I am a 55-year-old widow in Leh with no income"
    uv run python -m scheme_assistant.assistant "I farm 2 acres" --state Bihar --no-rerank
"""

import argparse
from contextlib import closing
from dataclasses import dataclass

from langsmith import traceable
from langsmith.utils import ContextThreadPoolExecutor
from qdrant_client import QdrantClient

from scheme_assistant.eligibility import EligibilityDecision, check_eligibility
from scheme_assistant.llm import DEFAULT_MODEL, LLMClient, LLMError, make_llm
from scheme_assistant.rerank import RERANK_CANDIDATES, BgeReranker, Reranker, rerank
from scheme_assistant.rewrite import rewrite_query
from scheme_assistant.search import QDRANT_PATH, BgeM3Embedder, load_scheme, search

SCHEMES_TO_CHECK = 10  # with re-ranking, the right scheme is in the top 10 for 95% of test queries
PARALLEL_CHECKS = 5  # eligibility checks are independent, so run several at once
VERDICT_ORDER = {"eligible": 0, "need_more_info": 1, "not_eligible": 2}


@dataclass
class SchemeResult:
    """One scheme found for the user, with the eligibility decision for it."""

    scheme_id: str
    name: str
    search_rank: int  # 1 = best match (after re-ranking, if used)
    decision: EligibilityDecision


@dataclass
class Answer:
    """Everything the assistant found for one question."""

    user_state: str | None
    results: list[SchemeResult]  # eligible first, then need_more_info, then not_eligible


@traceable
def answer_question(
    llm: LLMClient,
    client: QdrantClient,
    embedder: BgeM3Embedder,
    question: str,
    user_state: str | None = None,
    limit: int = SCHEMES_TO_CHECK,
    reranker: Reranker | None = None,
) -> Answer:
    """Find schemes for a user and check their eligibility for each.

    Args:
        llm: Client used for the state lookup and the eligibility checks.
        client: Qdrant client holding the scheme index.
        embedder: Embedder used to encode the question.
        question: The user's question in their own words.
        user_state: The state the user lives in; read from the question if None.
        limit: Number of schemes to retrieve and check.
        reranker: If given, the top RERANK_CANDIDATES search results are re-ranked
            and the best `limit` are checked.

    Returns:
        The state used and the checked schemes, eligible first.
    """
    if user_state is None:
        user_state = rewrite_query(llm, question).state

    if reranker is None:
        hits = search(client, embedder, question, user_state=user_state, limit=limit)
    else:
        candidates = max(limit, RERANK_CANDIDATES)
        hits = search(client, embedder, question, user_state=user_state, limit=candidates)
        hits = rerank(reranker, question, hits, limit)

    def check(scheme_id: str) -> EligibilityDecision | None:
        try:
            return check_eligibility(llm, load_scheme(scheme_id), question, user_state)
        except LLMError:
            return None  # the model gave no valid reply; leave this scheme out

    # Unlike ThreadPoolExecutor, this passes the trace to the worker threads, so the
    # checks are traced under this question instead of as separate traces.
    with ContextThreadPoolExecutor(max_workers=PARALLEL_CHECKS) as pool:
        decisions = list(pool.map(check, [hit.scheme_id for hit in hits]))

    results = [
        SchemeResult(hit.scheme_id, hit.name, search_rank=rank, decision=decision)
        for rank, (hit, decision) in enumerate(zip(hits, decisions), start=1)
        if decision is not None
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
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model name (Gemini, Gemma or Claude)"
    )
    parser.add_argument("--limit", type=int, default=SCHEMES_TO_CHECK, help="schemes to check")
    parser.add_argument("--no-rerank", action="store_true", help="skip re-ranking")
    args = parser.parse_args()

    llm = make_llm(args.model)
    reranker = None if args.no_rerank else BgeReranker()
    with closing(QdrantClient(path=str(QDRANT_PATH))) as client:
        answer = answer_question(
            llm, client, BgeM3Embedder(), args.question, args.state, args.limit, reranker
        )
    print_answer(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
