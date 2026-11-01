from collections.abc import Sequence

import numpy as np

from clinical_re_bilingual.retrieval.rar import RARPairBuilder
from clinical_re_bilingual.schema import Language, RelationExample, RelationLabel


class OrderedEncoder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [[float(index + 1), 1.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )


def test_pair_builder_uses_five_matched_positive_negative_pairs() -> None:
    examples = [
        RelationExample(
            example_id=f"train-{index}",
            language=Language.ENGLISH,
            sentence=f"sentence-{index}",
            subject=f"subject-{index}",
            subject_type="treatment",
            object=f"object-{index}",
            object_type="problem",
            label=RelationLabel.TrAP,
        )
        for index in range(7)
    ]
    triples = RARPairBuilder(OrderedEncoder(), positive_neighbors=5, negative_neighbors=5).build(
        examples
    )
    assert len(triples) == len(examples) * 5
    assert all(triple.anchor_id != triple.positive_id for triple in triples)
    assert all(triple.anchor_id != triple.negative_id for triple in triples)
