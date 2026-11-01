"""Adapter for retained retrieval rankings from the original experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from clinical_re_bilingual.retrieval.base import BaseRetriever
from clinical_re_bilingual.schema import RelationExample, RetrievalRanking


class PrecomputedRetriever(BaseRetriever):
    """Serve validated historical rankings without regenerating them."""

    def __init__(self, name: str, rankings: Mapping[str, Sequence[str]]) -> None:
        if not name:
            raise ValueError("retriever name must not be empty")
        self.name = name
        self._rankings = {key: list(value) for key, value in rankings.items()}
        self._training_ids: set[str] = set()
        self._language = None

    def fit(self, examples: Sequence[RelationExample]) -> PrecomputedRetriever:
        if not examples:
            raise ValueError("at least one training example is required")
        self._training_ids = {example.example_id for example in examples}
        if len(self._training_ids) != len(examples):
            raise ValueError("training example IDs must be unique")
        languages = {example.language for example in examples}
        if len(languages) != 1:
            raise ValueError("rankings must be fitted separately for each language")
        self._language = examples[0].language
        for query_id, ranking in self._rankings.items():
            if len(ranking) != len(set(ranking)):
                raise ValueError(f"ranking for {query_id!r} contains duplicate IDs")
            unknown = set(ranking) - self._training_ids
            if unknown:
                raise ValueError(f"ranking for {query_id!r} contains unknown training IDs")
        return self

    def rank(self, query: RelationExample) -> RetrievalRanking:
        if self._language is None:
            raise RuntimeError("fit must be called before rank")
        if query.language != self._language:
            raise ValueError("query language does not match the fitted ranking index")
        if query.example_id not in self._rankings:
            raise KeyError(f"no retained ranking for query {query.example_id!r}")
        return RetrievalRanking(
            test_example_id=query.example_id,
            retriever=self.name,
            language=query.language,
            ranked_train_ids=self._rankings[query.example_id],
        )
