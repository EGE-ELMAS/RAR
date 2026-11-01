"""Relation-aware contrastive pair construction and demonstration retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from clinical_re_bilingual.retrieval.base import BaseRetriever
from clinical_re_bilingual.retrieval.encoders import TextEncoder
from clinical_re_bilingual.schema import RelationExample, RetrievalRanking


@dataclass(frozen=True)
class ContrastiveTriple:
    """A sentence anchor with one relation-aware positive and hard negative."""

    anchor_id: str
    positive_id: str
    negative_id: str
    anchor: str
    positive: str
    negative: str


def _validate_weights(weights: Sequence[float], name: str) -> None:
    if any(weight < 0 for weight in weights):
        raise ValueError(f"{name} weights must be non-negative")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"{name} weights must sum to one")


def _component_embeddings(
    examples: Sequence[RelationExample], encoder: TextEncoder, *, include_labels: bool
) -> tuple[NDArray[np.float32], ...]:
    sentence = encoder.encode([example.sentence for example in examples])
    subject = encoder.encode([example.subject for example in examples])
    object_ = encoder.encode([example.object for example in examples])
    matrices = [sentence, subject, object_]
    if include_labels:
        matrices.append(encoder.encode([example.label.value for example in examples]))
    if any(matrix.ndim != 2 or matrix.shape[0] != len(examples) for matrix in matrices):
        raise ValueError("encoder returned an invalid component embedding matrix")
    return tuple(matrices)


def _weighted_similarity(
    left: tuple[NDArray[np.float32], ...],
    right: tuple[NDArray[np.float32], ...],
    *,
    sentence_weight: float,
    entity_weight: float,
    subject_weight: float,
    object_weight: float,
    relation_weight: float = 0.0,
) -> NDArray[np.float64]:
    _validate_weights((sentence_weight, entity_weight, relation_weight), "component")
    _validate_weights((subject_weight, object_weight), "entity")
    score = sentence_weight * cosine_similarity(left[0], right[0])
    entity_score = subject_weight * cosine_similarity(
        left[1], right[1]
    ) + object_weight * cosine_similarity(left[2], right[2])
    score += entity_weight * entity_score
    if relation_weight:
        if len(left) < 4 or len(right) < 4:
            raise ValueError("relation embeddings are required when relation_weight is non-zero")
        score += relation_weight * cosine_similarity(left[3], right[3])
    return score


class RARPairBuilder:
    """Build supervised SimCSE triples using Equation 7 similarity."""

    def __init__(
        self,
        encoder: TextEncoder,
        *,
        positive_neighbors: int = 1,
        negative_neighbors: int = 1,
        sentence_weight: float = 1 / 3,
        entity_weight: float = 1 / 3,
        relation_weight: float = 1 / 3,
        subject_weight: float = 1 / 2,
        object_weight: float = 1 / 2,
    ) -> None:
        if positive_neighbors < 1 or negative_neighbors < 1:
            raise ValueError("neighbor counts must be at least one")
        if positive_neighbors != negative_neighbors:
            raise ValueError("positive and negative neighbor counts must be equal")
        _validate_weights((sentence_weight, entity_weight, relation_weight), "component")
        _validate_weights((subject_weight, object_weight), "entity")
        self.encoder = encoder
        self.positive_neighbors = positive_neighbors
        self.negative_neighbors = negative_neighbors
        self.sentence_weight = sentence_weight
        self.entity_weight = entity_weight
        self.relation_weight = relation_weight
        self.subject_weight = subject_weight
        self.object_weight = object_weight

    def build(self, examples: Sequence[RelationExample]) -> list[ContrastiveTriple]:
        required = max(self.positive_neighbors, self.negative_neighbors) + 1
        if len(examples) < required:
            raise ValueError(f"at least {required} examples are required")
        ids = [example.example_id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("training example IDs must be unique")
        if len({example.language for example in examples}) != 1:
            raise ValueError("contrastive triples must be built separately for each language")

        embeddings = _component_embeddings(examples, self.encoder, include_labels=True)
        scores = _weighted_similarity(
            embeddings,
            embeddings,
            sentence_weight=self.sentence_weight,
            entity_weight=self.entity_weight,
            relation_weight=self.relation_weight,
            subject_weight=self.subject_weight,
            object_weight=self.object_weight,
        )
        np.fill_diagonal(scores, np.nan)
        triples: list[ContrastiveTriple] = []
        for anchor_index, anchor in enumerate(examples):
            candidates = [index for index in range(len(examples)) if index != anchor_index]
            descending = sorted(candidates, key=lambda index: (-scores[anchor_index, index], index))
            ascending = sorted(candidates, key=lambda index: (scores[anchor_index, index], index))
            positives = descending[: self.positive_neighbors]
            negatives = ascending[: self.negative_neighbors]
            for positive_index, negative_index in zip(positives, negatives, strict=True):
                triples.append(
                    ContrastiveTriple(
                        anchor_id=anchor.example_id,
                        positive_id=examples[positive_index].example_id,
                        negative_id=examples[negative_index].example_id,
                        anchor=anchor.sentence,
                        positive=examples[positive_index].sentence,
                        negative=examples[negative_index].sentence,
                    )
                )
        return triples


class RARRetriever(BaseRetriever):
    """Rank examples using fine-tuned sentence and entity embeddings."""

    name = "rar"

    def __init__(
        self,
        encoder: TextEncoder,
        *,
        candidate_pool_size: int = 20,
        sentence_weight: float = 1 / 2,
        entity_weight: float = 1 / 2,
        subject_weight: float = 1 / 2,
        object_weight: float = 1 / 2,
    ) -> None:
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least one")
        _validate_weights((sentence_weight, entity_weight), "component")
        _validate_weights((subject_weight, object_weight), "entity")
        self.encoder = encoder
        self.candidate_pool_size = candidate_pool_size
        self.sentence_weight = sentence_weight
        self.entity_weight = entity_weight
        self.subject_weight = subject_weight
        self.object_weight = object_weight
        self._examples: list[RelationExample] = []
        self._embeddings: tuple[NDArray[np.float32], ...] | None = None

    def fit(self, examples: Sequence[RelationExample]) -> RARRetriever:
        if not examples:
            raise ValueError("RAR retrieval requires at least one training example")
        ids = [example.example_id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("training example IDs must be unique")
        if len({example.language for example in examples}) != 1:
            raise ValueError("RAR must be fitted separately for each language")
        self._examples = list(examples)
        self._embeddings = _component_embeddings(self._examples, self.encoder, include_labels=False)
        return self

    def rank(self, query: RelationExample) -> RetrievalRanking:
        if self._embeddings is None or not self._examples:
            raise RuntimeError("fit must be called before rank")
        if query.language != self._examples[0].language:
            raise ValueError("query language does not match the fitted RAR index")
        query_embeddings = _component_embeddings([query], self.encoder, include_labels=False)
        scores = _weighted_similarity(
            query_embeddings,
            self._embeddings,
            sentence_weight=self.sentence_weight,
            entity_weight=self.entity_weight,
            subject_weight=self.subject_weight,
            object_weight=self.object_weight,
        )[0]
        order = np.argsort(-scores, kind="stable")[: self.candidate_pool_size]
        return RetrievalRanking(
            test_example_id=query.example_id,
            retriever=self.name,
            language=query.language,
            ranked_train_ids=[self._examples[index].example_id for index in order],
            scores=[float(scores[index]) for index in order],
        )


def relation_knn_micro_f1(
    train_examples: Sequence[RelationExample],
    validation_examples: Sequence[RelationExample],
    encoder: TextEncoder,
) -> float:
    """Evaluate an encoder by labels of the nearest RAR training examples."""

    if not validation_examples:
        raise ValueError("at least one validation example is required")
    retriever = RARRetriever(encoder, candidate_pool_size=1).fit(train_examples)
    labels_by_id = {example.example_id: example.label.value for example in train_examples}
    predicted = [
        labels_by_id[retriever.rank(item).ranked_train_ids[0]] for item in validation_examples
    ]
    gold = [item.label.value for item in validation_examples]
    return float(f1_score(gold, predicted, average="micro"))
