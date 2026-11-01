"""PURE/FT-RR entity-aware demonstration retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from clinical_re_bilingual.retrieval.base import BaseRetriever
from clinical_re_bilingual.retrieval.encoders import RelationEncoder
from clinical_re_bilingual.schema import RelationExample, RetrievalRanking


class PURERetriever(BaseRetriever):
    """Rank concatenated subject/object marker representations by cosine similarity."""

    name = "pure_rr"

    def __init__(self, encoder: RelationEncoder, candidate_pool_size: int = 20) -> None:
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1")
        self.encoder = encoder
        self.candidate_pool_size = candidate_pool_size
        self._examples: list[RelationExample] = []
        self._embeddings: np.ndarray | None = None

    def fit(self, examples: Sequence[RelationExample]) -> PURERetriever:
        if not examples:
            raise ValueError("PURE retrieval requires at least one training example")
        ids = [example.example_id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("training example IDs must be unique")
        languages = {example.language for example in examples}
        if len(languages) != 1:
            raise ValueError("PURE must be fitted separately for each language")
        self._examples = list(examples)
        self._embeddings = self.encoder.encode(self._examples)
        if self._embeddings.shape[0] != len(self._examples):
            raise ValueError("encoder output count does not match training examples")
        return self

    def rank(self, query: RelationExample) -> RetrievalRanking:
        if self._embeddings is None or not self._examples:
            raise RuntimeError("fit must be called before rank")
        if query.language != self._examples[0].language:
            raise ValueError("query language does not match the fitted PURE index")
        query_embedding = self.encoder.encode([query])
        similarities = cosine_similarity(query_embedding, self._embeddings)[0]
        order = np.argsort(-similarities, kind="stable")[: self.candidate_pool_size]
        return RetrievalRanking(
            test_example_id=query.example_id,
            retriever=self.name,
            language=query.language,
            ranked_train_ids=[self._examples[index].example_id for index in order],
            scores=[float(similarities[index]) for index in order],
        )
