from collections.abc import Sequence

import numpy as np
import pytest

from clinical_re_bilingual.retrieval.rar import (
    RARPairBuilder,
    RARRetriever,
    relation_knn_micro_f1,
)
from clinical_re_bilingual.schema import Language, RelationExample, RelationLabel


class FakeEncoder:
    vectors = {
        "anchor": [1.0, 0.0],
        "positive": [0.9, 0.1],
        "negative": [-1.0, 0.0],
        "subject-a": [1.0, 0.0],
        "subject-b": [0.9, 0.1],
        "subject-c": [-1.0, 0.0],
        "object-a": [0.0, 1.0],
        "object-b": [0.1, 0.9],
        "object-c": [0.0, -1.0],
        "TrAP": [1.0, 0.0],
        "TrIP": [-1.0, 0.0],
    }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def item(
    example_id: str,
    sentence: str,
    subject: str,
    object_: str,
    label: RelationLabel,
) -> RelationExample:
    return RelationExample(
        example_id=example_id,
        language=Language.ENGLISH,
        sentence=sentence,
        subject=subject,
        subject_type="treatment",
        object=object_,
        object_type="problem",
        label=label,
    )


def training_examples() -> list[RelationExample]:
    return [
        item("a", "anchor", "subject-a", "object-a", RelationLabel.TrAP),
        item("b", "positive", "subject-b", "object-b", RelationLabel.TrAP),
        item("c", "negative", "subject-c", "object-c", RelationLabel.TrIP),
    ]


def test_pair_builder_excludes_anchor_and_selects_extremes() -> None:
    triples = RARPairBuilder(FakeEncoder()).build(training_examples())
    first = triples[0]
    assert first.anchor_id == "a"
    assert first.positive_id == "b"
    assert first.negative_id == "c"
    assert all(triple.anchor_id != triple.positive_id for triple in triples)
    assert all(triple.anchor_id != triple.negative_id for triple in triples)


def test_rar_uses_sentence_and_entity_components() -> None:
    ranking = (
        RARRetriever(FakeEncoder())
        .fit(training_examples())
        .rank(item("q", "anchor", "subject-a", "object-a", RelationLabel.TrAP))
    )
    assert ranking.ranked_train_ids == ["a", "b", "c"]
    assert ranking.top_k(2) == ["a", "b"]


def test_rar_retains_twenty_candidates_by_default() -> None:
    examples = [
        item(f"a-{index}", "anchor", "subject-a", "object-a", RelationLabel.TrAP)
        for index in range(25)
    ]
    ranking = (
        RARRetriever(FakeEncoder())
        .fit(examples)
        .rank(examples[0].model_copy(update={"example_id": "query"}))
    )
    assert len(ranking.ranked_train_ids) == 20


def test_relation_validation_uses_nearest_neighbor_labels() -> None:
    validation = [
        item("q1", "anchor", "subject-a", "object-a", RelationLabel.TrAP),
        item("q2", "negative", "subject-c", "object-c", RelationLabel.TrIP),
    ]
    assert relation_knn_micro_f1(training_examples(), validation, FakeEncoder()) == 1.0


def test_pair_builder_rejects_mixed_languages() -> None:
    examples = training_examples()
    examples[1] = examples[1].model_copy(update={"language": Language.TURKISH})
    with pytest.raises(ValueError, match="separately"):
        RARPairBuilder(FakeEncoder()).build(examples)
