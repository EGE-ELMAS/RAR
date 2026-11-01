"""Public-safe metric and confusion-matrix output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from clinical_re_bilingual.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_records,
)
from clinical_re_bilingual.schema import PredictionRecord


def write_evaluation_outputs(
    records: list[PredictionRecord], metrics_path: Path, confusion_path: Path
) -> None:
    metrics = classification_metrics(records)
    matrix = confusion_matrix_records(records)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    labels = matrix["labels"]
    rows = matrix["matrix"]
    assert isinstance(labels, list) and isinstance(rows, list)
    with confusion_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gold\\predicted", *labels])
        for label, row in zip(labels, rows, strict=True):
            writer.writerow([label, *row])
