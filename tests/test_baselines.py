import json
from datetime import date
from pathlib import Path

from clinical_re_bilingual.evaluation.io import write_evaluation_outputs
from clinical_re_bilingual.evaluation.metrics import confusion_matrix_records
from clinical_re_bilingual.models.config import (
    read_baseline_checkpoints,
    read_baselines_config,
)
from clinical_re_bilingual.models.features import PureFeatureBuilder
from clinical_re_bilingual.schema import (
    Language,
    PredictionRecord,
    RelationExample,
    RelationLabel,
)


class FakeTokenizer:
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token_id = 0

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return list(range(1, len(tokens) + 1))


def relation_example() -> RelationExample:
    return RelationExample(
        example_id="example-1",
        language=Language.ENGLISH,
        sentence="therapy treats condition",
        subject="therapy",
        subject_type="treatment",
        object="condition",
        object_type="problem",
        label=RelationLabel.TrAP,
        subject_start=0,
        subject_end=0,
        object_start=2,
        object_end=2,
    )


def test_baseline_configuration_matches_experiment() -> None:
    root = Path(__file__).parents[1]
    config = read_baselines_config(root / "config/baselines.yaml")
    models = read_baseline_checkpoints(root / "config/models.yaml")
    assert config.bert.epochs == config.pure.epochs == 10
    assert config.bert.learning_rate == 2e-5
    assert config.pure.learning_rate == 1e-5
    assert config.bert.max_length == 128
    assert config.pure.max_length == 228
    assert models.pure.tr == "dbmdz/bert-base-turkish-uncased"


def test_pure_features_use_entity_start_markers() -> None:
    feature = PureFeatureBuilder(FakeTokenizer(), max_length=12).build(relation_example())
    assert feature.subject_index == 1
    assert feature.object_index == 5
    assert sum(feature.attention_mask) == 9
    assert len(feature.input_ids) == 12


def prediction(predicted: RelationLabel) -> PredictionRecord:
    return PredictionRecord(
        example_id=f"example-{predicted.value}",
        gold_label=RelationLabel.TrAP,
        predicted_label=predicted,
        model="synthetic",
        language=Language.ENGLISH,
        retriever="none",
        num_shots=0,
        prompt_strategy="fine_tuned",
        temperature=0.0,
        run_date=date(2026, 10, 30),
    )


def test_confusion_matrix_has_fixed_label_order() -> None:
    result = confusion_matrix_records([prediction(RelationLabel.TrAP)])
    assert result["labels"] == sorted(label.value for label in RelationLabel)
    assert sum(sum(row) for row in result["matrix"]) == 1


def test_evaluation_outputs_contain_no_clinical_text(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    confusion_path = tmp_path / "confusion.csv"
    write_evaluation_outputs([prediction(RelationLabel.TrAP)], metrics_path, confusion_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["micro_f1"] == 1.0
    assert "sentence" not in metrics_path.read_text(encoding="utf-8")
    assert "gold\\predicted" in confusion_path.read_text(encoding="utf-8")
