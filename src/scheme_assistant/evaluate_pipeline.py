"""End-to-end evaluation of the assistant on the labelled eligibility cases.

Each case's description is given to the full assistant (search + eligibility checks),
with the user's state given as it would be from a dropdown. The case's scheme is then
looked up in the answer. A case succeeds when the user would get the right answer:
    eligible or need_more_info  The scheme is found and given that verdict.
    not_eligible                The scheme is not shown as eligible (it may be left
                                out of the results, since the user cannot use it).

Also reported: how often the scheme is found at all, verdict accuracy when found, and
"false eligible" (the scheme shown as eligible when it should not be).

Results are written to data/eval/results/pipeline_<split>_<model>[_rerank].json.

Usage (requires a built index and an API key in .env):
    uv run python -m scheme_assistant.evaluate_pipeline --split dev
    uv run python -m scheme_assistant.evaluate_pipeline --split test --model gemini-3.8-flash
    uv run python -m scheme_assistant.evaluate_pipeline --split dev --no-rerank
"""

import argparse
import json
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date

from qdrant_client import QdrantClient

from scheme_assistant.assistant import SCHEMES_TO_CHECK, Answer, answer_question
from scheme_assistant.evaluate import EVAL_DIR, SPLITS
from scheme_assistant.evaluate_eligibility import load_cases, print_cost, usage_report
from scheme_assistant.hf_dataset import PROJECT_ROOT
from scheme_assistant.llm import DEFAULT_MODEL, make_llm
from scheme_assistant.models import EligibilityCase
from scheme_assistant.rerank import RERANK_CANDIDATES, RERANKER_MODEL, BgeReranker
from scheme_assistant.search import QDRANT_PATH, BgeM3Embedder


@dataclass
class CaseOutcome:
    """What happened to the case's scheme in the assistant's answer."""

    case_id: str
    kind: str
    expected: str
    found: bool
    search_rank: int | None  # None if the scheme was not retrieved
    verdict: str | None  # None if the scheme was not retrieved
    success: bool
    false_eligible: bool
    eligible_count: int  # schemes shown as eligible in the whole answer
    seconds: float


def score_case(case: EligibilityCase, answer: Answer, seconds: float) -> CaseOutcome:
    """Compare the assistant's answer with the case's expected verdict."""
    target = next((r for r in answer.results if r.scheme_id == case.scheme_id), None)
    verdict = target.decision.verdict if target else None

    if target is None:
        success = case.verdict == "not_eligible"
    else:
        success = verdict == case.verdict

    return CaseOutcome(
        case_id=case.id,
        kind=case.kind,
        expected=case.verdict,
        found=target is not None,
        search_rank=target.search_rank if target else None,
        verdict=verdict,
        success=success,
        false_eligible=verdict == "eligible" and case.verdict != "eligible",
        eligible_count=sum(r.decision.verdict == "eligible" for r in answer.results),
        seconds=seconds,
    )


def rate(outcomes: list[CaseOutcome], field: str) -> float:
    """Share of outcomes where a boolean field is true."""
    return sum(getattr(outcome, field) for outcome in outcomes) / len(outcomes)


def summarise(outcomes: list[CaseOutcome]) -> dict:
    """Compute the headline metrics, overall and per group."""
    found = [outcome for outcome in outcomes if outcome.found]

    def success_by(attribute: str) -> dict[str, float]:
        groups: dict[str, list[CaseOutcome]] = {}
        for outcome in outcomes:
            groups.setdefault(getattr(outcome, attribute), []).append(outcome)
        return {name: rate(group, "success") for name, group in sorted(groups.items())}

    return {
        "success": rate(outcomes, "success"),
        "found": rate(outcomes, "found"),
        "verdict_accuracy_when_found": (
            sum(outcome.verdict == outcome.expected for outcome in found) / len(found)
            if found
            else 0.0
        ),
        "false_eligible": sum(outcome.false_eligible for outcome in outcomes),
        "mean_eligible_shown": sum(o.eligible_count for o in outcomes) / len(outcomes),
        "mean_seconds": sum(outcome.seconds for outcome in outcomes) / len(outcomes),
        "success_by_expected": success_by("expected"),
        "success_by_kind": success_by("kind"),
    }


def print_summary(
    split: str, model: str, reranked: bool, summary: dict, outcomes: list[CaseOutcome]
) -> None:
    """Print the headline metrics and the failed cases."""
    search_setup = f"re-rank top {RERANK_CANDIDATES}" if reranked else "no re-rank"
    print(
        f"{split} set, model {model}, {search_setup}, {len(outcomes)} cases, "
        f"top {SCHEMES_TO_CHECK} checked\n"
    )
    print(f"Success (user gets the right answer): {summary['success']:.2f}")
    print(f"Scheme found in results:              {summary['found']:.2f}")
    print(f"Verdict correct when found:           {summary['verdict_accuracy_when_found']:.2f}")
    print(f"False eligible:                       {summary['false_eligible']}")
    print(f"Schemes shown as eligible per answer: {summary['mean_eligible_shown']:.1f}")
    print(f"Mean time per question:               {summary['mean_seconds']:.1f} s")
    for title, key in [("expected verdict", "success_by_expected"), ("kind", "success_by_kind")]:
        values = "  ".join(f"{name} {value:.2f}" for name, value in summary[key].items())
        print(f"Success by {title}: {values}")

    print("\nFailed cases:")
    for outcome in outcomes:
        if not outcome.success:
            where = f"rank {outcome.search_rank}" if outcome.found else "not found"
            print(
                f"  {outcome.case_id} {outcome.kind:15} expected {outcome.expected:15} "
                f"got {outcome.verdict or '-':15} ({where})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the assistant end to end.")
    parser.add_argument("--split", choices=SPLITS, default="dev", help="case set to evaluate")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model name (Gemini, Gemma or Claude)"
    )
    parser.add_argument("--no-rerank", action="store_true", help="skip re-ranking")
    args = parser.parse_args()

    cases = load_cases(args.split)
    llm = make_llm(args.model)
    embedder = BgeM3Embedder()
    reranked = not args.no_rerank
    reranker = BgeReranker() if reranked else None

    outcomes = []
    with closing(QdrantClient(path=str(QDRANT_PATH))) as client:
        for case in cases:
            start = time.perf_counter()
            answer = answer_question(
                llm,
                client,
                embedder,
                case.user_description,
                case.user_state,
                reranker=reranker,
            )
            outcomes.append(score_case(case, answer, time.perf_counter() - start))

    summary = summarise(outcomes)
    print_summary(args.split, args.model, reranked, summary, outcomes)
    print_cost(args.model, llm.usage, len(cases))

    report = {
        "date": date.today().isoformat(),
        "split": args.split,
        "model": args.model,
        "schemes_checked": SCHEMES_TO_CHECK,
        "reranker": (
            {"model": RERANKER_MODEL, "candidates": RERANK_CANDIDATES} if reranked else None
        ),
        "usage": usage_report(args.model, llm.usage, len(cases)),
        "summary": summary,
        "cases": [asdict(outcome) for outcome in outcomes],
    }
    suffix = "_rerank" if reranked else ""
    results_path = EVAL_DIR / "results" / f"pipeline_{args.split}_{args.model}{suffix}.json"
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
