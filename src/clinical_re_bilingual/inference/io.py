"""Public-safe prediction serialization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from clinical_re_bilingual.schema import PredictionRecord


def write_predictions_jsonl(records: Iterable[PredictionRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")
            count += 1
    if count == 0:
        path.unlink(missing_ok=True)
        raise ValueError("refusing to write an empty prediction file")
    return count


def read_predictions_jsonl(path: Path) -> list[PredictionRecord]:
    records = [
        PredictionRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"prediction file is empty: {path}")
    ids = [record.example_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction file contains duplicate example IDs")
    return records
