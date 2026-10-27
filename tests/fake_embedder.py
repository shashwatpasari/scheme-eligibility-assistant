"""Deterministic bag-of-words embedder used in place of bge-m3 in tests."""

import re
import zlib

from scheme_assistant.search import Embedding

VECTOR_SIZE = 256


class FakeEmbedder:
    """Hashes words into a fixed-size count vector.

    Texts that share words get similar dense and sparse vectors, which is enough
    to exercise indexing, filtering and fusion without loading a model.
    """

    def embed(self, texts: list[str]) -> list[Embedding]:
        embeddings = []
        for text in texts:
            word_counts: dict[int, float] = {}
            for word in re.findall(r"\w+", text.lower()):
                bucket = zlib.crc32(word.encode()) % VECTOR_SIZE
                word_counts[bucket] = word_counts.get(bucket, 0.0) + 1.0

            dense_vector = [word_counts.get(bucket, 0.0) for bucket in range(VECTOR_SIZE)]
            if not any(dense_vector):
                dense_vector[0] = 1.0  # cosine distance is undefined for a zero vector

            embeddings.append(Embedding(dense=dense_vector, sparse=word_counts))
        return embeddings
