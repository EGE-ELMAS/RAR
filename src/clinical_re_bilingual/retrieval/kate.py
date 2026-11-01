"""KNN-Augmented in-context Example selection (KATE)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from clinical_re_bilingual.retrieval.base import BaseRetriever
from clinical_re_bilingual.retrieval.encoders import TextEncoder
from clinical_re_bilingual.schema import RelationExample, RetrievalRanking


class KATERetriever(BaseRetriever):
    """Rank sentences by cosine similarity of first-token embeddings.

    Stable sorting resolves exact similarity ties by training-data order.
    """

    name = "kate"

    def __init__(self, encoder: TextEncoder, candidate_pool_size: int = 20) -> None:
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1")
        self.encoder = encoder
        self.candidate_pool_size = candidate_pool_size
        self._examples: list[RelationExample] = []
        self._embeddings: np.ndarray | None = None

    def fit(self, examples: Sequence[RelationExample]) -> KATERetriever:
        if not examples:
            raise ValueError("KATE requires at least one training example")
        ids = [example.example_id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("training example IDs must be unique")
        languages = {example.language for example in examples}
        if len(languages) != 1:
            raise ValueError("KATE must be fitted separately for each language")
        self._examples = list(examples)
        self._embeddings = self.encoder.encode([example.sentence for example in examples])
        if self._embeddings.shape[0] != len(self._examples):
            raise ValueError("encoder output count does not match training examples")
        return self

    def rank(self, query: RelationExample) -> RetrievalRanking:
        if self._embeddings is None or not self._examples:
            raise RuntimeError("fit must be called before rank")
        if query.language != self._examples[0].language:
            raise ValueError("query language does not match the fitted KATE index")
        query_embedding = self.encoder.encode([query.sentence])
        similarities = cosine_similarity(query_embedding, self._embeddings)[0]
        order = np.argsort(-similarities, kind="stable")
        order = order[: self.candidate_pool_size]
        return RetrievalRanking(
            test_example_id=query.example_id,
            retriever=self.name,
            language=query.language,
            ranked_train_ids=[self._examples[index].example_id for index in order],
            scores=[float(similarities[index]) for index in order],
        )
