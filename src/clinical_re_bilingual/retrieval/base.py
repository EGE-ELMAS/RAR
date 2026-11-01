"""Interface implemented by every demonstration retriever."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from clinical_re_bilingual.schema import RelationExample, RetrievalRanking


class BaseRetriever(ABC):
    name: str

    @abstractmethod
    def fit(self, examples: Sequence[RelationExample]) -> BaseRetriever:
        """Index the authorized training examples."""

    @abstractmethod
    def rank(self, query: RelationExample) -> RetrievalRanking:
        """Return every candidate in descending relevance order."""

    def retrieve(self, query: RelationExample, k: int) -> list[str]:
        return self.rank(query).top_k(k)
