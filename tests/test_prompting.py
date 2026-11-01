from datetime import date

import pytest

from clinical_re_bilingual.inference.runner import ExperimentRunner
from clinical_re_bilingual.prompting import (
    PromptBuilder,
    PromptStrategy,
    demonstrations_from_ranking,
    parse_relation_label,
)
from clinical_re_bilingual.prompting.rationales import rationale_request
from clinical_re_bilingual.schema import (
    Language,
    RelationExample,
    RelationLabel,
    RetrievalRanking,
)


def example(example_id: str, label: RelationLabel = RelationLabel.TrAP) -> RelationExample:
    return RelationExample(
        example_id=example_id,
        language=Language.ENGLISH,
        sentence=f"synthetic context {example_id}",
        subject="therapy",
        subject_type="treatment",
        object="condition",
        object_type="problem",
        label=label,
    )


def ranking() -> RetrievalRanking:
    return RetrievalRanking(
        test_example_id="test",
        retriever="rar",
        language=Language.ENGLISH,
        ranked_train_ids=[f"train-{index}" for index in range(20)],
        scores=[float(20 - index) for index in range(20)],
    )


def test_first_k_ranked_examples_are_used_in_order() -> None:
    training = [example(f"train-{index}") for index in reversed(range(20))]
    selected = demonstrations_from_ranking(ranking(), training, 5)
    assert [item.example_id for item in selected] == [f"train-{index}" for index in range(5)]


def test_standard_prompt_preserves_subject_object_order() -> None:
    prompt = PromptBuilder(Language.ENGLISH, PromptStrategy.STANDARD).build(
        example("test"), [example("train-0")]
    )
    assert 'relation between "therapy" and "condition"' in prompt
    assert "Response: TREATMENT IS ADMINISTERED FOR MEDICAL PROBLEM" in prompt
    assert prompt.endswith("Response:")


def test_output_format_cot_contains_entity_types_and_text() -> None:
    prompt = PromptBuilder(Language.ENGLISH, PromptStrategy.OUTPUT_FORMAT_COT).build(
        example("test"), [example("train-0")]
    )
    assert "treatment [therapy] IS ADMINISTERED FOR problem [condition]" in prompt


def test_zero_shot_static_cot_uses_reasoning_instruction_without_examples() -> None:
    prompt = PromptBuilder(Language.ENGLISH, PromptStrategy.ZS_STATIC_COT).build(
        example("test"), []
    )
    assert "Let's think step by step" in prompt
    assert "synthetic context train" not in prompt
    assert prompt.endswith("Response:")


def test_few_shot_static_cot_requires_fixed_rationale() -> None:
    with pytest.raises(ValueError, match="requires a rationale"):
        PromptBuilder(Language.ENGLISH, PromptStrategy.FS_STATIC_COT).build(
            example("test"), [example("train-0")]
        )


def test_dynamic_reasoning_requires_rationale() -> None:
    with pytest.raises(ValueError, match="requires a rationale"):
        PromptBuilder(Language.ENGLISH, PromptStrategy.GOLD_LABEL_COT).build(
            example("test"), [example("train-0")]
        )


def test_gold_label_rationale_request_contains_known_relation() -> None:
    request = rationale_request(example("train-0"), PromptStrategy.GOLD_LABEL_COT)
    assert "TREATMENT IS ADMINISTERED FOR MEDICAL PROBLEM" in request


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("TrAP", RelationLabel.TrAP),
        ("TREATMENT IS ADMINISTERED FOR MEDICAL PROBLEM", RelationLabel.TrAP),
        ("PİUTdv", RelationLabel.TrAP),
        ("TstOÇP", RelationLabel.TeRP),
    ],
)
def test_bilingual_response_parser(response: str, expected: RelationLabel) -> None:
    assert parse_relation_label(response) is expected


class FakeClient:
    model_name = "fake-model"

    def generate(self, prompt: str, *, temperature: float) -> str:
        assert "synthetic context test" in prompt
        assert temperature == 0.0
        return "TrAP"


def test_runner_returns_public_safe_prediction_record() -> None:
    training = [example(f"train-{index}") for index in range(20)]
    record = ExperimentRunner(FakeClient()).predict(
        example("test"),
        training,
        ranking(),
        num_shots=5,
        strategy=PromptStrategy.STANDARD,
        run_date=date(2026, 10, 29),
    )
    assert record.predicted_label is RelationLabel.TrAP
    assert "sentence" not in record.model_dump()
