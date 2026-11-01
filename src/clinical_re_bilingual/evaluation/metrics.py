"""Metrics computed from public-safe label records."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from clinical_re_bilingual.schema import PredictionRecord, RelationLabel


def classification_metrics(records: Sequence[PredictionRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("at least one prediction record is required")
    gold = [record.gold_label.value for record in records]
    predicted = [record.predicted_label.value for record in records]
    labels = [label.value for label in sorted(RelationLabel, key=lambda item: item.value)]
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "micro_f1": float(f1_score(gold, predicted, average="micro")),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, predicted, average="weighted", zero_division=0)),
        "per_relation": classification_report(
            gold,
            predicted,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def confusion_matrix_records(records: Sequence[PredictionRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("at least one prediction record is required")
    labels = [label.value for label in sorted(RelationLabel, key=lambda item: item.value)]
    gold = [record.gold_label.value for record in records]
    predicted = [record.predicted_label.value for record in records]
    matrix = confusion_matrix(gold, predicted, labels=labels)
    return {"labels": labels, "matrix": matrix.tolist()}
