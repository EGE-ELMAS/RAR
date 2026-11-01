"""Aggregate a directory of public-safe prediction files."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from clinical_re_bilingual.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_records,
)
from clinical_re_bilingual.inference.io import read_predictions_jsonl
from clinical_re_bilingual.schema import PredictionRecord, RelationLabel

SUMMARY_FIELDS = [
    "experiment",
    "model",
    "language",
    "retriever",
    "num_shots",
    "prompt_strategy",
    "temperature",
    "run_date",
    "examples",
    "accuracy",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
]


def _constant(records: Sequence[PredictionRecord], field: str):
    values = {getattr(record, field) for record in records}
    if len(values) != 1:
        raise ValueError(f"prediction file has inconsistent {field}: {values}")
    return next(iter(values))


def validate_prediction_set(records: Sequence[PredictionRecord], path: Path) -> None:
    if len(records) != 500:
        raise ValueError(f"{path} contains {len(records)} records; expected 500")
    language = _constant(records, "language")
    expected_ids = [f"{language.value}_test_{index:04d}" for index in range(1, 501)]
    actual_ids = [record.example_id for record in records]
    if actual_ids != expected_ids:
        raise ValueError(f"{path} does not contain the ordered official test IDs")


def summarize_file(path: Path) -> tuple[dict[str, object], list[dict[str, object]], dict]:
    records = read_predictions_jsonl(path)
    validate_prediction_set(records, path)
    metrics = classification_metrics(records)
    per_relation = metrics["per_relation"]
    assert isinstance(per_relation, dict)
    weighted = per_relation["weighted avg"]
    assert isinstance(weighted, dict)
    row = {
        "experiment": path.name,
        "model": _constant(records, "model"),
        "language": _constant(records, "language").value,
        "retriever": _constant(records, "retriever"),
        "num_shots": _constant(records, "num_shots"),
        "prompt_strategy": _constant(records, "prompt_strategy"),
        "temperature": _constant(records, "temperature"),
        "run_date": _constant(records, "run_date").isoformat(),
        "examples": len(records),
        "accuracy": metrics["accuracy"],
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": weighted["f1-score"],
    }
    relation_rows = []
    for label in RelationLabel:
        values = per_relation[label.value]
        assert isinstance(values, dict)
        relation_rows.append(
            {
                "experiment": path.name,
                "relation": label.value,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
            }
        )
    return row, relation_rows, confusion_matrix_records(records)


def write_results_summary(predictions_dir: Path, output_dir: Path) -> int:
    paths = sorted(predictions_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no JSONL prediction files found in {predictions_dir}")
    summary_rows = []
    relation_rows = []
    matrices = {}
    gold_by_language: dict[str, list[str]] = {}
    for path in paths:
        records = read_predictions_jsonl(path)
        validate_prediction_set(records, path)
        language = records[0].language.value
        gold = [record.gold_label.value for record in records]
        if language in gold_by_language and gold_by_language[language] != gold:
            raise ValueError(f"{path} has an inconsistent {language} gold-label sequence")
        gold_by_language.setdefault(language, gold)
        row, per_relation, matrix = summarize_file(path)
        summary_rows.append(row)
        relation_rows.extend(per_relation)
        matrices[path.name] = matrix
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "all_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    relation_fields = ["experiment", "relation", "precision", "recall", "f1", "support"]
    with (output_dir / "per_relation_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=relation_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(relation_rows)
    (output_dir / "confusion_matrices.json").write_text(
        json.dumps(matrices, indent=2) + "\n", encoding="utf-8"
    )
    return len(summary_rows)
