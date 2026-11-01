from datetime import date

import pytest

from clinical_re_bilingual.prompting.builder import PromptStrategy
from clinical_re_bilingual.schema import (
    Language,
    RelationExample,
    RelationLabel,
    RetrievalRanking,
)
from clinical_re_bilingual.workflows.inference import predict_dataset


def example(example_id: str) -> RelationExample:
    return RelationExample(
        example_id=example_id,
        language=Language.ENGLISH,
        sentence="synthetic sentence",
        subject="therapy",
        subject_type="treatment",
        object="condition",
        object_type="problem",
        label=RelationLabel.TrAP,
    )


class FakeClient:
    model_name = "fake"

    def generate(self, prompt: str, *, temperature: float) -> str:
        return "TrAP"


def test_dataset_predictions_preserve_test_order() -> None:
    training = [example(f"train-{index}") for index in range(20)]
    test = [example("test-2"), example("test-1")]
    ranked_ids = [item.example_id for item in training]
    rankings = [
        RetrievalRanking(
            test_example_id=item.example_id,
            retriever="rar",
            language=Language.ENGLISH,
            ranked_train_ids=ranked_ids,
        )
        for item in reversed(test)
    ]
    records = predict_dataset(
        client=FakeClient(),
        training_examples=training,
        test_examples=test,
        rankings=rankings,
        num_shots=5,
        strategy=PromptStrategy.STANDARD,
        temperature=0.0,
        run_date=date(2026, 10, 29),
    )
    assert [record.example_id for record in records] == ["test-2", "test-1"]


def test_dataset_predictions_reject_missing_ranking() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        predict_dataset(
            client=FakeClient(),
            training_examples=[example(f"train-{index}") for index in range(20)],
            test_examples=[example("test-1")],
            rankings=[],
            num_shots=5,
            strategy=PromptStrategy.STANDARD,
            temperature=0.0,
            run_date=date(2026, 10, 29),
        )


def test_zero_shot_static_cot_does_not_require_rankings() -> None:
    records = predict_dataset(
        client=FakeClient(),
        training_examples=[],
        test_examples=[example("test-1")],
        rankings=None,
        num_shots=0,
        strategy=PromptStrategy.ZS_STATIC_COT,
        temperature=0.0,
        run_date=date(2026, 10, 29),
    )
    assert records[0].retriever == "none"
    assert records[0].num_shots == 0
    assert records[0].prompt_strategy == "zs_static_cot"
