"""Retrieval evaluation on the labelled query sets.

Two query sets are used:
    dev   For comparing setups and choosing settings (such as the hybrid weight).
    test  Held out; used only to report the final numbers.

Each query is run through every setup, the rank of its first relevant scheme is
recorded, and Hit@k and MRR are reported. Results are written to
data/eval/results/retrieval_<split>.json.

Usage (requires a built index):
    uv run python -m scheme_assistant.evaluate --split dev
    uv run python -m scheme_assistant.evaluate --split test
"""

import argparse
import json
from contextlib import closing
from dataclasses import dataclass
from datetime import date

from qdrant_client import QdrantClient

from scheme_assistant.hf_dataset import PROJECT_ROOT
from scheme_assistant.models import RetrievalQuery
from scheme_assistant.search import (
    DEFAULT_DENSE_WEIGHT,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    BgeM3Embedder,
    Embedding,
    search_by_embedding,
)

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
SPLITS = ["dev", "test"]
TOP_K = 10
HYBRID_DENSE_WEIGHTS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


@dataclass
class Setup:
    """One retrieval configuration to evaluate."""

    mode: str
    use_state_filter: bool
    dense_weight: float = DEFAULT_DENSE_WEIGHT  # used by hybrid mode only

    def name(self) -> str:
        """Readable label, e.g. "dense", "hybrid 70/30 + state filter"."""
        label = self.mode
        if self.mode == "hybrid":
            dense_percent = round(self.dense_weight * 100)
            label = f"hybrid {dense_percent}/{100 - dense_percent}"
        if self.use_state_filter:
            label += " + state filter"
        return label


def build_setups() -> list[Setup]:
    """Single retrievers and default hybrid, then a sweep of hybrid weights with the filter."""
    setups = [
        Setup("dense", use_state_filter=False),
        Setup("sparse", use_state_filter=False),
        Setup("hybrid", use_state_filter=False),
        Setup("dense", use_state_filter=True),
        Setup("sparse", use_state_filter=True),
    ]
    for dense_weight in HYBRID_DENSE_WEIGHTS:
        setups.append(Setup("hybrid", use_state_filter=True, dense_weight=dense_weight))
    return setups


def load_queries(split: str) -> list[RetrievalQuery]:
    """Load the labelled queries of one split ("dev" or "test")."""
    path = EVAL_DIR / f"retrieval_{split}.jsonl"
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            queries.append(RetrievalQuery.model_validate_json(line))
    return queries


# --- Metrics ----------------------------------------------------------------


def first_relevant_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> int | None:
    """Return the 1-based rank of the first relevant scheme, or None if absent."""
    for rank, scheme_id in enumerate(retrieved_ids, start=1):
        if scheme_id in relevant_ids:
            return rank
    return None


def compute_metrics(ranks: list[int | None]) -> dict[str, float]:
    """Compute Hit@1, Hit@5, Hit@10 and MRR.

    Args:
        ranks: For each query, the rank of its first relevant scheme (None if not retrieved).

    Returns:
        Mapping of metric name to value, each between 0 and 1.
    """
    found_ranks = [rank for rank in ranks if rank is not None]
    query_count = len(ranks)
    return {
        "hit@1": sum(1 for rank in found_ranks if rank <= 1) / query_count,
        "hit@5": sum(1 for rank in found_ranks if rank <= 5) / query_count,
        "hit@10": sum(1 for rank in found_ranks if rank <= 10) / query_count,
        "mrr": sum(1 / rank for rank in found_ranks) / query_count,
    }


def compute_metrics_by_style(
    queries: list[RetrievalQuery], ranks: list[int | None]
) -> dict[str, dict[str, float]]:
    """Compute metrics separately for each query style (persona, hindi, ...)."""
    ranks_by_style: dict[str, list[int | None]] = {}
    for query, rank in zip(queries, ranks):
        ranks_by_style.setdefault(query.style, []).append(rank)
    return {style: compute_metrics(ranks_by_style[style]) for style in sorted(ranks_by_style)}


# --- Evaluation -------------------------------------------------------------


def rank_queries(
    client: QdrantClient,
    setup: Setup,
    queries: list[RetrievalQuery],
    query_embeddings: list[Embedding],
) -> list[int | None]:
    """Run every query through one setup and return the first-relevant rank of each."""
    ranks = []
    for query, embedding in zip(queries, query_embeddings):
        user_state = query.user_state if setup.use_state_filter else None
        hits = search_by_embedding(
            client,
            embedding,
            mode=setup.mode,
            user_state=user_state,
            limit=TOP_K,
            dense_weight=setup.dense_weight,
        )
        retrieved_ids = [hit.scheme_id for hit in hits]
        ranks.append(first_relevant_rank(retrieved_ids, query.relevant))
    return ranks


def print_summary(split: str, queries: list[RetrievalQuery], results: dict[str, dict]) -> None:
    """Print the metrics table and details for the best setup."""
    print(f"{split} set: {len(queries)} queries, top {TOP_K}; hybrid D/S = dense/sparse weight\n")
    print(f"{'setup':30} {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>6}")
    for name, result in results.items():
        metrics = result["overall"]
        print(
            f"{name:30} {metrics['hit@1']:6.2f} {metrics['hit@5']:6.2f} "
            f"{metrics['hit@10']:7.2f} {metrics['mrr']:6.2f}"
        )

    best_name = max(results, key=lambda name: results[name]["overall"]["mrr"])
    best = results[best_name]
    print(f"\nBest setup by MRR: {best_name}")
    print("Hit@10 by query style:")
    for style, metrics in best["by_style"].items():
        print(f"  {style:9} {metrics['hit@10']:.2f}")
    missed_ids = [query_id for query_id, rank in best["ranks"].items() if rank is None]
    print(f"Missed (not in top {TOP_K}): {', '.join(missed_ids) or 'none'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate scheme retrieval.")
    parser.add_argument("--split", choices=SPLITS, default="dev", help="query set to evaluate")
    args = parser.parse_args()

    queries = load_queries(args.split)
    # Embed once and reuse across setups; only the search parameters change.
    query_embeddings = BgeM3Embedder().embed([query.query for query in queries])

    results = {}
    with closing(QdrantClient(path=str(QDRANT_PATH))) as client:
        for setup in build_setups():
            ranks = rank_queries(client, setup, queries, query_embeddings)
            results[setup.name()] = {
                "overall": compute_metrics(ranks),
                "by_style": compute_metrics_by_style(queries, ranks),
                "ranks": {query.id: rank for query, rank in zip(queries, ranks)},
            }

    print_summary(args.split, queries, results)

    report = {
        "date": date.today().isoformat(),
        "split": args.split,
        "embedding_model": EMBEDDING_MODEL,
        "default_dense_weight": DEFAULT_DENSE_WEIGHT,
        "top_k": TOP_K,
        "setups": results,
    }
    results_path = EVAL_DIR / "results" / f"retrieval_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
