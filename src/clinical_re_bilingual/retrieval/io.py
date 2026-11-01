"""Public-safe serialization for retrieval rankings."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from clinical_re_bilingual.schema import RetrievalRanking


def write_rankings_jsonl(rankings: Iterable[RetrievalRanking], path: Path) -> int:
    """Write ID-only rankings and return the number of records written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for ranking in rankings:
            stream.write(ranking.model_dump_json() + "\n")
            count += 1
    if count == 0:
        path.unlink(missing_ok=True)
        raise ValueError("refusing to write an empty ranking file")
    return count
