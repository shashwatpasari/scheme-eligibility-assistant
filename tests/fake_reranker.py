"""Simple re-rankers used in place of bge-reranker in tests."""

import re


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class WordOverlapReranker:
    """Scores each text by the number of query words it contains."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(len(words(query) & words(text))) for text in texts]


class ReverseReranker:
    """Gives the last text the highest score, so it reverses the search order."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(index) for index in range(len(texts))]
