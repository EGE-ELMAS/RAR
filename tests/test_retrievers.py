from collections.abc import Sequence

import numpy as np

from clinical_re_bilingual.retrieval.kate import KATERetriever
from clinical_re_bilingual.retrieval.precomputed import PrecomputedRetriever
from clinical_re_bilingual.retrieval.random import RandomRetriever
from clinical_re_bilingual.schema import Language, RelationExample, RelationLabel


def example(example_id: str, sentence: str) -> RelationExample:
    return RelationExample(
        example_id=example_id,
        language=Language.ENGLISH,
        sentence=sentence,
        subject="synthetic treatment",
        subject_type="treatment",
        object="synthetic problem",
        object_type="problem",
        label=RelationLabel.TrAP,
    )


class FakeEncoder:
    vectors = {
        "closest": [1.0, 0.0],
        "second": [0.8, 0.2],
        "opposite": [-1.0, 0.0],
        "query": [1.0, 0.0],
        "tie-a": [0.0, 1.0],
        "tie-b": [0.0, 1.0],
        "tie-query": [0.0, 1.0],
    }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_kate_ranks_by_cosine_similarity() -> None:
    training = [
        example("train-1", "opposite"),
        example("train-2", "second"),
        example("train-3", "closest"),
    ]
    ranking = KATERetriever(FakeEncoder()).fit(training).rank(example("test-1", "query"))
    assert ranking.ranked_train_ids == ["train-3", "train-2", "train-1"]
    assert ranking.top_k(2) == ["train-3", "train-2"]


def test_kate_ties_preserve_training_order() -> None:
    training = [example("train-a", "tie-a"), example("train-b", "tie-b")]
    ranking = KATERetriever(FakeEncoder()).fit(training).rank(example("test-1", "tie-query"))
    assert ranking.ranked_train_ids == ["train-a", "train-b"]


def test_kate_keeps_only_configured_candidate_pool() -> None:
    training = [
        example("train-1", "opposite"),
        example("train-2", "second"),
        example("train-3", "closest"),
    ]
    ranking = (
        KATERetriever(FakeEncoder(), candidate_pool_size=2)
        .fit(training)
        .rank(example("test-1", "query"))
    )
    assert ranking.ranked_train_ids == ["train-3", "train-2"]


def test_random_ranking_is_repeatable_per_query() -> None:
    training = [example(f"train-{index}", "closest") for index in range(10)]
    retriever = RandomRetriever(seed=42).fit(training)
    first = retriever.rank(example("test-1", "query"))
    second = retriever.rank(example("test-1", "query"))
    assert first == second
    assert set(first.ranked_train_ids) == {item.example_id for item in training}


def test_random_default_candidate_pool_is_twenty() -> None:
    training = [example(f"train-{index}", "closest") for index in range(30)]
    ranking = RandomRetriever(seed=42).fit(training).rank(example("test-1", "query"))
    assert len(ranking.ranked_train_ids) == 20


def test_precomputed_ranking_preserves_historical_order() -> None:
    training = [example("train-1", "closest"), example("train-2", "second")]
    retriever = PrecomputedRetriever("kate", {"test-1": ["train-2", "train-1"]}).fit(training)
    assert retriever.retrieve(example("test-1", "query"), 2) == ["train-2", "train-1"]
