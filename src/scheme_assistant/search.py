"""Hybrid scheme retrieval backed by Qdrant and bge-m3.

Retrieval modes:
    dense   Semantic similarity over bge-m3 dense embeddings (cosine).
    sparse  Lexical matching over bge-m3 sparse weights (dot product).
    hybrid  Both, merged with weighted Reciprocal Rank Fusion (RRF).

Results can be restricted to central schemes plus the user's state.

Usage:
    uv run python -m scheme_assistant.search --build
    uv run python -m scheme_assistant.search "widow with no income" --state Ladakh
    uv run python -m scheme_assistant.search "widow with no income" --dense-weight 0.5
"""

import argparse
from contextlib import closing
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from scheme_assistant.build_schemes import SCHEMES_DIR
from scheme_assistant.hf_dataset import PROJECT_ROOT
from scheme_assistant.models import Scheme

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
COLLECTION_NAME = "schemes"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
MODES = ["dense", "sparse", "hybrid"]

CANDIDATE_LIMIT = 50  # results taken from each retriever before fusion
DEFAULT_DENSE_WEIGHT = 0.7  # chosen on the dev query set; sparse gets the remaining 0.3
RRF_K = 1  # rank offset; 1 matches Qdrant's built-in RRF (1st place earns 1/2)

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_TOKENS = 2048  # truncates only the ~10 longest schemes


@dataclass
class Embedding:
    """Dense and sparse representations of one text."""

    dense: list[float]
    sparse: dict[int, float]  # token id -> weight


@dataclass
class SearchHit:
    """A single retrieved scheme."""

    scheme_id: str
    name: str
    score: float


class BgeM3Embedder:
    """Encodes text with bge-m3, producing dense and sparse vectors in one pass."""

    def __init__(self) -> None:
        # Imported lazily: loading the library and model takes several seconds.
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=False)

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Encode a batch of texts.

        Args:
            texts: Texts to encode.

        Returns:
            One Embedding per input text, in the same order.
        """
        output = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            max_length=EMBEDDING_MAX_TOKENS,
            return_dense=True,
            return_sparse=True,
        )

        embeddings = []
        for dense_vector, token_weights in zip(output["dense_vecs"], output["lexical_weights"]):
            sparse_weights = {}
            for token_id, weight in token_weights.items():
                sparse_weights[int(token_id)] = float(weight)
            embeddings.append(Embedding(dense=dense_vector.tolist(), sparse=sparse_weights))
        return embeddings


# --- Indexing ---------------------------------------------------------------


def load_schemes() -> list[Scheme]:
    """Load all scheme files from the data directory, sorted by id."""
    schemes = []
    for path in sorted(SCHEMES_DIR.glob("*.json")):
        schemes.append(Scheme.model_validate_json(path.read_text(encoding="utf-8")))
    return schemes


def load_scheme(scheme_id: str) -> Scheme:
    """Load one scheme by id."""
    path = SCHEMES_DIR / f"{scheme_id}.json"
    return Scheme.model_validate_json(path.read_text(encoding="utf-8"))


def scheme_text(scheme: Scheme) -> str:
    """Build the text that represents a scheme in the index.

    Only fields describing who the scheme targets and what it offers are used;
    documents and application steps add noise without helping retrieval.
    """
    return "\n\n".join([scheme.name, scheme.details, scheme.eligibility, scheme.benefits])


def to_sparse_vector(weights: dict[int, float]) -> models.SparseVector:
    """Convert token weights to a Qdrant sparse vector."""
    return models.SparseVector(indices=list(weights.keys()), values=list(weights.values()))


def build_index(client: QdrantClient, embedder: BgeM3Embedder, schemes: list[Scheme]) -> int:
    """Embed schemes and write them to a freshly created collection.

    Any existing collection is dropped first, so rebuilding never leaves stale points.

    Args:
        client: Qdrant client to write to.
        embedder: Encoder used for the scheme texts.
        schemes: Schemes to index.

    Returns:
        Number of schemes indexed.
    """
    embeddings = embedder.embed([scheme_text(scheme) for scheme in schemes])

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=len(embeddings[0].dense),
                distance=models.Distance.COSINE,
            )
        },
        # Qdrant always scores sparse vectors by dot product.
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
    )

    points = []
    for index, (scheme, embedding) in enumerate(zip(schemes, embeddings)):
        points.append(
            models.PointStruct(
                id=index,  # Qdrant requires integer or UUID ids; the scheme id is in the payload
                vector={
                    DENSE_VECTOR_NAME: embedding.dense,
                    SPARSE_VECTOR_NAME: to_sparse_vector(embedding.sparse),
                },
                payload={
                    "scheme_id": scheme.id,
                    "name": scheme.name,
                    "level": scheme.level,
                    "state": scheme.state,
                },
            )
        )

    client.upload_points(COLLECTION_NAME, points=points)
    return len(points)


# --- Retrieval --------------------------------------------------------------


def build_state_filter(user_state: str | None) -> models.Filter | None:
    """Restrict results to central schemes and schemes of the user's state.

    Args:
        user_state: The user's state or UT, or None if unknown.

    Returns:
        A Qdrant filter, or None when no state is given (search everything).
    """
    if user_state is None:
        return None

    is_central = models.FieldCondition(key="level", match=models.MatchValue(value="central"))
    is_user_state = models.FieldCondition(key="state", match=models.MatchValue(value=user_state))
    return models.Filter(should=[is_central, is_user_state])  # should = OR


def query_vector(
    client: QdrantClient,
    vector: list[float] | models.SparseVector,
    vector_name: str,
    query_filter: models.Filter | None,
    limit: int,
) -> list[SearchHit]:
    """Run a single dense or sparse query against the collection."""
    response = client.query_points(
        COLLECTION_NAME,
        query=vector,
        using=vector_name,
        query_filter=query_filter,
        limit=limit,
    )

    hits = []
    for point in response.points:
        if point.payload is None:
            continue
        hits.append(
            SearchHit(
                scheme_id=point.payload["scheme_id"],
                name=point.payload["name"],
                score=point.score,
            )
        )
    return hits


def rrf_scores(hits: list[SearchHit], weight: float) -> dict[str, float]:
    """Score each scheme by its rank in one result list: weight / (RRF_K + rank)."""
    scores = {}
    for rank, hit in enumerate(hits, start=1):
        scores[hit.scheme_id] = weight / (RRF_K + rank)
    return scores


def weighted_rrf(
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    dense_weight: float,
    limit: int,
) -> list[SearchHit]:
    """Merge dense and sparse results with weighted Reciprocal Rank Fusion.

    A scheme's fused score is the sum of its rank scores from both lists. Ranks
    are used instead of raw similarity scores because dense and sparse scores
    are on different scales.

    Args:
        dense_hits: Dense results, best first.
        sparse_hits: Sparse results, best first.
        dense_weight: Weight of the dense list, from 0 to 1. Sparse gets 1 - dense_weight.
        limit: Maximum number of results.

    Returns:
        Fused hits, best first.

    Raises:
        ValueError: If dense_weight is outside 0 to 1.
    """
    if not 0 <= dense_weight <= 1:
        raise ValueError(f"dense_weight must be between 0 and 1, got {dense_weight}")

    dense_scores = rrf_scores(dense_hits, dense_weight)
    sparse_scores = rrf_scores(sparse_hits, 1 - dense_weight)

    names = {}
    for hit in dense_hits + sparse_hits:
        names[hit.scheme_id] = hit.name

    fused_scores = {}
    for scheme_id in names:
        fused_scores[scheme_id] = dense_scores.get(scheme_id, 0.0) + sparse_scores.get(
            scheme_id, 0.0
        )

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [
        SearchHit(scheme_id, names[scheme_id], fused_scores[scheme_id])
        for scheme_id in ranked_ids[:limit]
    ]


def search_by_embedding(
    client: QdrantClient,
    embedding: Embedding,
    mode: str = "hybrid",
    user_state: str | None = None,
    limit: int = 10,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
) -> list[SearchHit]:
    """Retrieve schemes for a pre-computed query embedding.

    Args:
        client: Qdrant client holding the scheme collection.
        embedding: Embedding of the user's query.
        mode: One of "dense", "sparse" or "hybrid".
        user_state: If given, only central schemes and this state's schemes are returned.
        limit: Maximum number of results.
        dense_weight: Hybrid mode only; weight of dense results in the fusion.

    Returns:
        Hits ordered from most to least relevant.

    Raises:
        ValueError: If the mode or dense_weight is invalid.
    """
    query_filter = build_state_filter(user_state)
    dense_vector = embedding.dense
    sparse_vector = to_sparse_vector(embedding.sparse)

    if mode == "dense":
        return query_vector(client, dense_vector, DENSE_VECTOR_NAME, query_filter, limit)
    if mode == "sparse":
        return query_vector(client, sparse_vector, SPARSE_VECTOR_NAME, query_filter, limit)
    if mode == "hybrid":
        dense_hits = query_vector(
            client, dense_vector, DENSE_VECTOR_NAME, query_filter, CANDIDATE_LIMIT
        )
        sparse_hits = query_vector(
            client, sparse_vector, SPARSE_VECTOR_NAME, query_filter, CANDIDATE_LIMIT
        )
        return weighted_rrf(dense_hits, sparse_hits, dense_weight, limit)

    raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")


def search(
    client: QdrantClient,
    embedder: BgeM3Embedder,
    query: str,
    mode: str = "hybrid",
    user_state: str | None = None,
    limit: int = 10,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
) -> list[SearchHit]:
    """Embed a query and retrieve matching schemes.

    See search_by_embedding for the remaining arguments.
    """
    query_embedding = embedder.embed([query])[0]
    return search_by_embedding(client, query_embedding, mode, user_state, limit, dense_weight)


# --- Command line -----------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or query the scheme search index.")
    parser.add_argument("query", nargs="?", help="search query")
    parser.add_argument("--build", action="store_true", help="(re)build the index")
    parser.add_argument("--state", help="restrict results to central schemes and this state")
    parser.add_argument("--mode", choices=MODES, default="hybrid", help="retrieval mode")
    parser.add_argument("--limit", type=int, default=10, help="number of results")
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help=f"hybrid only: weight of dense results, 0 to 1 (default {DEFAULT_DENSE_WEIGHT})",
    )
    args = parser.parse_args()

    if not args.build and not args.query:
        parser.error("provide a query, or --build to build the index")

    embedder = BgeM3Embedder()
    with closing(QdrantClient(path=str(QDRANT_PATH))) as client:
        if args.build:
            indexed_count = build_index(client, embedder, load_schemes())
            print(f"Indexed {indexed_count} schemes into {QDRANT_PATH.relative_to(PROJECT_ROOT)}/")

        if args.query:
            hits = search(
                client,
                embedder,
                args.query,
                mode=args.mode,
                user_state=args.state,
                limit=args.limit,
                dense_weight=args.dense_weight,
            )
            for rank, hit in enumerate(hits, start=1):
                print(f"{rank:2}. {hit.scheme_id:20} {hit.score:.3f}  {hit.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
