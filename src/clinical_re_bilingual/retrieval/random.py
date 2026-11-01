"""Deterministic random demonstration ranking."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from clinical_re_bilingual.retrieval.base import BaseRetriever
from clinical_re_bilingual.schema import RelationExample, RetrievalRanking


class RandomRetriever(BaseRetriever):
    name = "random"

    def __init__(self, seed: int = 42, candidate_pool_size: int = 20) -> None:
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1")
        self.seed = seed
        self.candidate_pool_size = candidate_pool_size
        self._examples: list[RelationExample] = []

    def fit(self, examples: Sequence[RelationExample]) -> RandomRetriever:
        if not examples:
            raise ValueError("random retrieval requires at least one training example")
        ids = [example.example_id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("training example IDs must be unique")
        languages = {example.language for example in examples}
        if len(languages) != 1:
            raise ValueError("random retrieval must be fitted separately for each language")
        self._examples = list(examples)
        return self

    def rank(self, query: RelationExample) -> RetrievalRanking:
        if not self._examples:
            raise RuntimeError("fit must be called before rank")
        if query.language != self._examples[0].language:
            raise ValueError("query language does not match the fitted random index")
        digest = hashlib.sha256(f"{self.seed}:{query.example_id}".encode()).digest()
        query_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        order = list(range(len(self._examples)))
        random.Random(query_seed).shuffle(order)
        order = order[: self.candidate_pool_size]
        return RetrievalRanking(
            test_example_id=query.example_id,
            retriever=self.name,
            language=query.language,
            ranked_train_ids=[self._examples[index].example_id for index in order],
        )
