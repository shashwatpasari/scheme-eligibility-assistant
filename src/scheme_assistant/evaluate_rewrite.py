"""Evaluate query rewriting on the labelled query sets.

Each query is rewritten by the LLM, then searched three ways with the best search
setup (hybrid + state filter):
    original            The question as the user typed it (the baseline).
    rewrite             The LLM's English search query only.
    original + rewrite  Both texts joined, so no detail from either is lost.

The state the LLM extracts is checked against the labelled state. Most queries do not
mention a state, so the main check is that the LLM never returns a wrong one.

Results are written to data/eval/results/rewrite_<split>_<model>.json.

Usage (requires a built index and GEMINI_API_KEY):
    uv run python -m scheme_assistant.evaluate_rewrite --split dev
    uv run python -m scheme_assistant.evaluate_rewrite --split test --model gemini-3.8-flash
"""

import argparse
import json
import time
from contextlib import closing
from datetime import date

from qdrant_client import QdrantClient

from scheme_assistant.evaluate import (
    EVAL_DIR,
    SPLITS,
    TOP_K,
    Setup,
    compute_metrics,
    compute_metrics_by_style,
    load_queries,
    rank_queries,
)
from scheme_assistant.hf_dataset import PROJECT_ROOT
from scheme_assistant.llm import DEFAULT_GEMINI_MODEL, GeminiClient, LLMClient
from scheme_assistant.models import RetrievalQuery
from scheme_assistant.rewrite import QueryRewrite, rewrite_query
from scheme_assistant.search import QDRANT_PATH, BgeM3Embedder

SEARCH_SETUP = Setup("hybrid", use_state_filter=True)
VARIANTS = ["original", "rewrite", "original + rewrite"]


def search_text(variant: str, query: str, rewrite: QueryRewrite) -> str:
    """Return the text to search with for one variant."""
    if variant == "original":
        return query
    if variant == "rewrite":
        return rewrite.search_query
    if variant == "original + rewrite":
        return f"{query}\n{rewrite.search_query}"
    raise ValueError(f"Unknown variant: {variant!r}")


def check_states(queries: list[RetrievalQuery], rewrites: list[QueryRewrite]) -> dict[str, int]:
    """Count extracted states that are correct, missing (None) or wrong."""
    counts = {"correct": 0, "none": 0, "wrong": 0}
    for query, rewrite in zip(queries, rewrites):
        if rewrite.state is None:
            counts["none"] += 1
        elif rewrite.state == query.user_state:
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return counts


def rewrite_all(
    llm: LLMClient, queries: list[RetrievalQuery]
) -> tuple[list[QueryRewrite], list[float]]:
    """Rewrite every query, returning the rewrites and the seconds each one took."""
    rewrites, durations = [], []
    for query in queries:
        start = time.perf_counter()
        rewrites.append(rewrite_query(llm, query.query))
        durations.append(time.perf_counter() - start)
    return rewrites, durations


def print_summary(
    split: str, model: str, results: dict[str, dict], state_counts: dict[str, int]
) -> None:
    """Print search metrics for each variant and the state check."""
    print(f"{split} set, rewrite model {model}, search: {SEARCH_SETUP.name()}\n")
    print(f"{'search text':20} {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>6}  Hit@10 by style")
    for variant, result in results.items():
        metrics = result["overall"]
        by_style = "  ".join(
            f"{style} {style_metrics['hit@10']:.2f}"
            for style, style_metrics in result["by_style"].items()
        )
        print(
            f"{variant:20} {metrics['hit@1']:6.2f} {metrics['hit@5']:6.2f} "
            f"{metrics['hit@10']:7.2f} {metrics['mrr']:6.2f}  {by_style}"
        )
    print(
        f"\nExtracted state: {state_counts['correct']} correct, "
        f"{state_counts['none']} none, {state_counts['wrong']} wrong"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LLM query rewriting.")
    parser.add_argument("--split", choices=SPLITS, default="dev", help="query set to evaluate")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name")
    args = parser.parse_args()

    queries = load_queries(args.split)
    rewrites, durations = rewrite_all(GeminiClient(args.model), queries)

    embedder = BgeM3Embedder()
    results = {}
    with closing(QdrantClient(path=str(QDRANT_PATH))) as client:
        for variant in VARIANTS:
            texts = [search_text(variant, q.query, r) for q, r in zip(queries, rewrites)]
            ranks = rank_queries(client, SEARCH_SETUP, queries, embedder.embed(texts))
            results[variant] = {
                "overall": compute_metrics(ranks),
                "by_style": compute_metrics_by_style(queries, ranks),
                "ranks": {query.id: rank for query, rank in zip(queries, ranks)},
            }

    state_counts = check_states(queries, rewrites)
    print_summary(args.split, args.model, results, state_counts)
    mean_seconds = sum(durations) / len(durations)
    print(f"Mean rewrite time: {mean_seconds:.1f} s")

    report = {
        "date": date.today().isoformat(),
        "split": args.split,
        "rewrite_model": args.model,
        "search_setup": SEARCH_SETUP.name(),
        "top_k": TOP_K,
        "mean_rewrite_seconds": round(mean_seconds, 2),
        "state_check": state_counts,
        "variants": results,
        "rewrites": {query.id: rewrite.model_dump() for query, rewrite in zip(queries, rewrites)},
    }
    results_path = EVAL_DIR / "results" / f"rewrite_{args.split}_{args.model}.json"
    results_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {results_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
