"""Evaluate the eligibility check on the labelled cases.

Each case pairs a user description with one scheme and the correct verdict. Reported:
    accuracy           Share of cases with the correct verdict.
    recall per verdict Share of each true verdict that was predicted correctly.
    false eligible     Cases wrongly marked eligible. This is the most harmful error:
                       the user is told they qualify when they do not, or may not.
    confusion matrix   Counts of (true verdict, predicted verdict).

Results are written to data/eval/results/eligibility_<split>_<model>.json.

Usage (requires an API key in .env):
    uv run python -m scheme_assistant.evaluate_eligibility --split dev
    uv run python -m scheme_assistant.evaluate_eligibility --split test --model gemini-3.8-flash
"""

import argparse
import json
import time
from datetime import date
from typing import get_args

from scheme_assistant.eligibility import EligibilityDecision, check_eligibility
from scheme_assistant.evaluate import EVAL_DIR, SPLITS
from scheme_assistant.hf_dataset import PROJECT_ROOT
from scheme_assistant.llm import (
    DEFAULT_MODEL,
    LLMClient,
    LLMError,
    TokenUsage,
    cost_usd,
    make_llm,
)
from scheme_assistant.models import EligibilityCase, Verdict
from scheme_assistant.search import load_scheme

VERDICTS: list[str] = list(get_args(Verdict))
NO_REPLY = "no_reply"  # recorded when the model gives no valid reply; counts as wrong


def load_cases(split: str) -> list[EligibilityCase]:
    """Load the labelled eligibility cases of one split ("dev" or "test")."""
    path = EVAL_DIR / f"eligibility_{split}.jsonl"
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EligibilityCase.model_validate_json(line))
    return cases


# --- Metrics ----------------------------------------------------------------


def confusion_matrix(true: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    """Count cases for each (true verdict, predicted verdict) pair."""
    matrix = {row: {column: 0 for column in VERDICTS} for row in VERDICTS}
    for true_verdict, predicted_verdict in zip(true, predicted):
        if predicted_verdict in matrix[true_verdict]:  # NO_REPLY has no column
            matrix[true_verdict][predicted_verdict] += 1
    return matrix


def compute_metrics(true: list[str], predicted: list[str]) -> dict:
    """Compute accuracy, recall per verdict and false-eligible count.

    Args:
        true: The labelled verdict of each case.
        predicted: The verdict the model gave for each case, in the same order.

    Returns:
        Mapping with "accuracy", "recall" (per verdict), "false_eligible", "no_reply"
        and "confusion_matrix".
    """
    matrix = confusion_matrix(true, predicted)
    correct = sum(matrix[verdict][verdict] for verdict in VERDICTS)
    recall = {}
    for verdict in VERDICTS:
        verdict_total = sum(matrix[verdict].values())
        recall[verdict] = matrix[verdict][verdict] / verdict_total if verdict_total else 0.0
    false_eligible = sum(
        matrix[verdict]["eligible"] for verdict in VERDICTS if verdict != "eligible"
    )
    return {
        "accuracy": correct / len(true),
        "recall": recall,
        "false_eligible": false_eligible,
        "no_reply": predicted.count(NO_REPLY),
        "confusion_matrix": matrix,
    }


def accuracy_by_kind(cases: list[EligibilityCase], predicted: list[str]) -> dict[str, float]:
    """Compute accuracy separately for each case kind (basic, boundary, ...)."""
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for case, verdict in zip(cases, predicted):
        totals[case.kind] = totals.get(case.kind, 0) + 1
        correct[case.kind] = correct.get(case.kind, 0) + (verdict == case.verdict)
    return {kind: correct[kind] / totals[kind] for kind in sorted(totals)}


# --- Evaluation -------------------------------------------------------------


def check_all(
    llm: LLMClient, cases: list[EligibilityCase]
) -> tuple[list[EligibilityDecision | None], list[float]]:
    """Run the eligibility check on every case, returning decisions and seconds taken.

    A decision is None when the model gave no valid reply.
    """
    decisions, durations = [], []
    for case in cases:
        scheme = load_scheme(case.scheme_id)
        start = time.perf_counter()
        try:
            decision = check_eligibility(llm, scheme, case.user_description, case.user_state)
        except LLMError:
            decision = None
        decisions.append(decision)
        durations.append(time.perf_counter() - start)
    return decisions, durations


def print_summary(
    split: str, model: str, metrics: dict, by_kind: dict[str, float], mean_seconds: float
) -> None:
    """Print the metrics, accuracy per case kind and the confusion matrix."""
    print(f"{split} set, model {model}\n")
    print(f"Accuracy:       {metrics['accuracy']:.2f}")
    print(f"False eligible: {metrics['false_eligible']}")
    print(f"No reply:       {metrics['no_reply']}")
    print(f"Mean time:      {mean_seconds:.1f} s")
    print("Accuracy by kind: " + "  ".join(f"{k} {v:.2f}" for k, v in by_kind.items()) + "\n")
    print(f"{'true \\ predicted':18}" + "".join(f"{verdict:>16}" for verdict in VERDICTS))
    for verdict in VERDICTS:
        row = metrics["confusion_matrix"][verdict]
        counts = "".join(f"{row[column]:>16}" for column in VERDICTS)
        print(f"{verdict:18}{counts}   recall {metrics['recall'][verdict]:.2f}")


def usage_report(model: str, usage: TokenUsage, case_count: int) -> dict:
    """Token totals and cost for a run; cost is None if the price is unknown."""
    cost = cost_usd(model, usage)
    return {
        "calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "invalid_replies_retried": usage.invalid_replies,
        "cost_usd": cost,
        "cost_per_case_usd": cost / case_count if cost is not None else None,
    }


def print_cost(model: str, usage: TokenUsage, case_count: int) -> None:
    """Print tokens used and the cost at paid-tier prices."""
    report = usage_report(model, usage, case_count)
    tokens = f"{usage.input_tokens:,} input + {usage.output_tokens:,} output tokens"
    if report["cost_usd"] is None:
        print(f"\nUsage: {tokens} (price unknown)")
    else:
        print(
            f"\nUsage: {tokens}, ${report['cost_usd']:.3f} total, "
            f"${report['cost_per_case_usd']:.5f} per case"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the eligibility check.")
    parser.add_argument("--split", choices=SPLITS, default="dev", help="case set to evaluate")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model name (Gemini, Gemma or Claude)"
    )
    args = parser.parse_args()

    cases = load_cases(args.split)
    llm = make_llm(args.model)
    decisions, durations = check_all(llm, cases)

    true = [case.verdict for case in cases]
    predicted = [decision.verdict if decision else NO_REPLY for decision in decisions]
    metrics = compute_metrics(true, predicted)
    by_kind = accuracy_by_kind(cases, predicted)
    mean_seconds = sum(durations) / len(durations)
    print_summary(args.split, args.model, metrics, by_kind, mean_seconds)
    print_cost(args.model, llm.usage, len(cases))

    errors = [case.id for case, verdict in zip(cases, predicted) if verdict != case.verdict]
    print(f"\nWrong: {', '.join(errors) or 'none'}")

    report = {
        "date": date.today().isoformat(),
        "split": args.split,
        "model": args.model,
        "mean_seconds": round(mean_seconds, 2),
        "usage": usage_report(args.model, llm.usage, len(cases)),
        "metrics": metrics,
        "accuracy_by_kind": by_kind,
        "decisions": {
            case.id: {
                "kind": case.kind,
                "true": case.verdict,
                **(decision.model_dump() if decision else {"verdict": NO_REPLY}),
            }
            for case, decision in zip(cases, decisions)
        },
    }
    results_path = EVAL_DIR / "results" / f"eligibility_{args.split}_{args.model}.json"
    results_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {results_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
