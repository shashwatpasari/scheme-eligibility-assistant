"""Re-ranking of search results with a cross-encoder.

Search turns the question and each scheme into separate vectors and compares them,
which is fast but loses detail. A cross-encoder reads the question and one scheme
together and scores how well they match, which is slower but more accurate. So
search finds candidates and the re-ranker puts them in a better order.

Any model with a `score(query, texts)` method can be used as the re-ranker.

Usage (requires a built index):
    uv run python -m scheme_assistant.rerank "widow with no income" --state Ladakh
"""

import argparse
from contextlib import closing
from typing import Protocol

from qdrant_client import QdrantClient

from scheme_assistant.search import (
    QDRANT_PATH,
    BgeM3Embedder,
    SearchHit,
    load_scheme,
    scheme_text,
    search,
)

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 20  # search results to re-rank; best MRR on dev among 10, 20 and 50
RERANK_MAX_TOKENS = 512  # 1024 was 2x slower and gave the same top 10 in a spot check
RERANK_BATCH_SIZE = 8  # fastest batch size on an Apple M4 in a timing check


class Reranker(Protocol):
    """Anything that can score how well texts match a query."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text; higher means more relevant."""
        ...


class BgeReranker:
    """Cross-encoder re-ranker using bge-reranker-v2-m3 (multilingual, runs locally)."""

    def __init__(self) -> None:
        # Imported lazily: loading the library and model takes several seconds.
        from FlagEmbedding import FlagReranker

        # Half precision is ~1.5x faster on Apple chips and gave the same order.
        self.model = FlagReranker(RERANKER_MODEL, use_fp16=True)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against the query.

        Args:
            query: The user's question.
            texts: Scheme texts to score.

        Returns:
            One score per text, in the same order. Higher means more relevant.
        """
        if not texts:
            return []
        scores = self.model.compute_score(
            [(query, text) for text in texts],
            max_length=RERANK_MAX_TOKENS,
            batch_size=RERANK_BATCH_SIZE,
        )
        return [float(score) for score in scores]


def rerank(reranker: Reranker, query: str, hits: list[SearchHit], limit: int) -> list[SearchHit]:
    """Re-order search hits by re-ranker score and keep the best.

    Args:
        reranker: Model that scores (query, scheme text) pairs.
        query: The user's question.
        hits: Search results to re-order.
        limit: Maximum number of hits to return.

    Returns:
        Hits ordered by re-ranker score, best first, with that score attached.
    """
    texts = [scheme_text(load_scheme(hit.scheme_id)) for hit in hits]
    scores = reranker.score(query, texts)
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [SearchHit(hit.scheme_id, hit.name, score) for hit, score in ranked[:limit]]


def search_and_rerank(
    client: QdrantClient,
    embedder: BgeM3Embedder,
    reranker: Reranker,
    query: str,
    user_state: str | None = None,
    limit: int = 10,
    candidates: int = RERANK_CANDIDATES,
) -> list[SearchHit]:
    """Search for candidate schemes, then re-rank them.

    Args:
        client: Qdrant client holding the scheme index.
        embedder: Embedder used to encode the query.
        reranker: Model used to re-order the candidates.
        query: The user's question.
        user_state: If given, only central schemes and this state's schemes are searched.
        limit: Number of hits to return.
        candidates: Number of search results to re-rank; should be at least `limit`.

    Returns:
        The best `limit` hits after re-ranking.
    """
    hits = search(client, embedder, query, user_state=user_state, limit=candidates)
    return rerank(reranker, query, hits, limit)


# --- Command line -----------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Search, then re-rank, and show both orders.")
    parser.add_argument("query", help="search query")
    parser.add_argument("--state", help="restrict results to central schemes and this state")
    parser.add_argument("--limit", type=int, default=10, help="number of results")
    parser.add_argument(
        "--candidates", type=int, default=RERANK_CANDIDATES, help="search results to re-rank"
    )
    args = parser.parse_args()

    embedder = BgeM3Embedder()
    with closing(QdrantClient(path=str(QDRANT_PATH))) as client:
        hits = search(client, embedder, args.query, user_state=args.state, limit=args.candidates)
    search_rank = {hit.scheme_id: rank for rank, hit in enumerate(hits, start=1)}
    reranked = rerank(BgeReranker(), args.query, hits, args.limit)

    print(f"{'new':>3} {'was':>3}  {'score':>6}  scheme")
    for rank, hit in enumerate(reranked, start=1):
        print(f"{rank:3} {search_rank[hit.scheme_id]:3}  {hit.score:6.2f}  {hit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
