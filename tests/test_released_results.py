import csv
import json
from pathlib import Path

import pytest

from clinical_re_bilingual.evaluation.aggregate import write_results_summary
from clinical_re_bilingual.inference.io import read_predictions_jsonl

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results" / "predictions"
SUMMARY = ROOT / "results" / "summary"


def test_release_contains_only_complete_label_only_experiments() -> None:
    paths = sorted(PREDICTIONS.glob("*.jsonl"))
    assert len(paths) == 84
    for path in paths:
        records = read_predictions_jsonl(path)
        assert len(records) == 500
        assert [record.example_id for record in records] == [
            f"{records[0].language.value}_test_{index:04d}" for index in range(1, 501)
        ]
        assert {record.num_shots for record in records} <= {0, 5}


def test_committed_summaries_are_reproducible(tmp_path: Path) -> None:
    assert write_results_summary(PREDICTIONS, tmp_path) == 84
    generated_matrices = json.loads((tmp_path / "confusion_matrices.json").read_text())
    committed_matrices = json.loads((SUMMARY / "confusion_matrices.json").read_text())
    assert generated_matrices == committed_matrices

    for name, numeric_fields in [
        (
            "all_results.csv",
            {"temperature", "accuracy", "micro_f1", "macro_f1", "weighted_f1"},
        ),
        ("per_relation_results.csv", {"precision", "recall", "f1"}),
    ]:
        with (tmp_path / name).open(encoding="utf-8", newline="") as stream:
            generated = list(csv.DictReader(stream))
        with (SUMMARY / name).open(encoding="utf-8", newline="") as stream:
            committed = list(csv.DictReader(stream))
        assert len(generated) == len(committed)
        for generated_row, committed_row in zip(generated, committed, strict=True):
            assert generated_row.keys() == committed_row.keys()
            for field in generated_row:
                if field in numeric_fields:
                    assert float(generated_row[field]) == pytest.approx(float(committed_row[field]))
                else:
                    assert generated_row[field] == committed_row[field]
