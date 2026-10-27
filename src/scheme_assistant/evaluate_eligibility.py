"""Evaluate the eligibility check on the labelled cases.

Each case pairs a user description with one scheme and the correct verdict. Reported:
    accuracy           Share of cases with the correct verdict.
    recall per verdict Share of each true verdict that was predicted correctly.
    false eligible     Cases wrongly marked eligible. This is the most harmful error:
                       the user is told they qualify when they do not, or may not.
    confusion matrix   Counts of (true verdict, predicted verdict).

Results are written to data/eval/results/eligibility_<split>_<model>.json.

Usage (requires GEMINI_API_KEY):
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
from scheme_assistant.llm import DEFAULT_GEMINI_MODEL, GeminiClient, LLMClient
from scheme_assistant.models import EligibilityCase, Verdict
from scheme_assistant.search import load_scheme

VERDICTS: list[str] = list(get_args(Verdict))


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
        matrix[true_verdict][predicted_verdict] += 1
    return matrix


def compute_metrics(true: list[str], predicted: list[str]) -> dict:
    """Compute accuracy, recall per verdict and false-eligible count.

    Args:
        true: The labelled verdict of each case.
        predicted: The verdict the model gave for each case, in the same order.

    Returns:
        Mapping with "accuracy", "recall" (per verdict), "false_eligible" and
        "confusion_matrix".
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
        "confusion_matrix": matrix,
    }


# --- Evaluation -------------------------------------------------------------


def check_all(
    llm: LLMClient, cases: list[EligibilityCase]
) -> tuple[list[EligibilityDecision], list[float]]:
    """Run the eligibility check on every case, returning decisions and seconds taken."""
    decisions, durations = [], []
    for case in cases:
        scheme = load_scheme(case.scheme_id)
        start = time.perf_counter()
        decisions.append(check_eligibility(llm, scheme, case.user_description, case.user_state))
        durations.append(time.perf_counter() - start)
    return decisions, durations


def print_summary(split: str, model: str, metrics: dict, mean_seconds: float) -> None:
    """Print the metrics and confusion matrix."""
    print(f"{split} set, model {model}\n")
    print(f"Accuracy:       {metrics['accuracy']:.2f}")
    print(f"False eligible: {metrics['false_eligible']}")
    print(f"Mean time:      {mean_seconds:.1f} s\n")
    print(f"{'true \\ predicted':18}" + "".join(f"{verdict:>16}" for verdict in VERDICTS))
    for verdict in VERDICTS:
        row = metrics["confusion_matrix"][verdict]
        counts = "".join(f"{row[column]:>16}" for column in VERDICTS)
        print(f"{verdict:18}{counts}   recall {metrics['recall'][verdict]:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the eligibility check.")
    parser.add_argument("--split", choices=SPLITS, default="dev", help="case set to evaluate")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name")
    args = parser.parse_args()

    cases = load_cases(args.split)
    decisions, durations = check_all(GeminiClient(args.model), cases)

    true = [case.verdict for case in cases]
    predicted = [decision.verdict for decision in decisions]
    metrics = compute_metrics(true, predicted)
    mean_seconds = sum(durations) / len(durations)
    print_summary(args.split, args.model, metrics, mean_seconds)

    errors = [case.id for case, verdict in zip(cases, predicted) if verdict != case.verdict]
    print(f"\nWrong: {', '.join(errors) or 'none'}")

    report = {
        "date": date.today().isoformat(),
        "split": args.split,
        "model": args.model,
        "mean_seconds": round(mean_seconds, 2),
        "metrics": metrics,
        "decisions": {
            case.id: {"true": case.verdict, **decision.model_dump()}
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
