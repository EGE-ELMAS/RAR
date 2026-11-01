from collections.abc import Sequence

import numpy as np

from clinical_re_bilingual.retrieval.encoders import BertEntityMarkerEncoder
from clinical_re_bilingual.retrieval.pure import PURERetriever
from clinical_re_bilingual.schema import Language, RelationExample, RelationLabel


def example(example_id: str, sentence: str, *, with_spans: bool = False) -> RelationExample:
    kwargs = {}
    if with_spans:
        kwargs = {
            "subject_start": 0,
            "subject_end": 0,
            "object_start": 2,
            "object_end": 2,
        }
    return RelationExample(
        example_id=example_id,
        language=Language.ENGLISH,
        sentence=sentence,
        subject="therapy",
        subject_type="treatment",
        object="condition",
        object_type="problem",
        label=RelationLabel.TrAP,
        **kwargs,
    )


class FakeRelationEncoder:
    vectors = {
        "closest": [1.0, 0.0, 1.0, 0.0],
        "second": [0.8, 0.2, 0.8, 0.2],
        "opposite": [-1.0, 0.0, -1.0, 0.0],
        "query": [1.0, 0.0, 1.0, 0.0],
    }

    def encode(self, examples: Sequence[RelationExample]) -> np.ndarray:
        return np.asarray([self.vectors[item.sentence] for item in examples], dtype=np.float32)


class FakeTokenizer:
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token_id = 0

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return list(range(1, len(tokens) + 1))


def test_pure_ranks_concatenated_entity_representations() -> None:
    training = [
        example("train-1", "opposite"),
        example("train-2", "second"),
        example("train-3", "closest"),
    ]
    ranking = PURERetriever(FakeRelationEncoder()).fit(training).rank(example("test-1", "query"))
    assert ranking.ranked_train_ids == ["train-3", "train-2", "train-1"]


def test_entity_markers_follow_original_token_offsets() -> None:
    encoder = BertEntityMarkerEncoder("unused", max_length=12)
    encoder._tokenizer = FakeTokenizer()
    record = example("train-1", "therapy treats condition", with_spans=True)
    input_ids, attention_mask, subject_index, object_index = encoder._feature(record)
    assert subject_index == 1
    assert object_index == 5
    assert len(input_ids) == 12
    assert sum(attention_mask) == 9


def test_pure_requires_entity_offsets() -> None:
    encoder = BertEntityMarkerEncoder("unused")
    encoder._tokenizer = FakeTokenizer()
    try:
        encoder._feature(example("train-1", "therapy treats condition"))
    except ValueError as error:
        assert "requires entity offsets" in str(error)
    else:
        raise AssertionError("missing offsets were not rejected")
