from pathlib import Path

import pytest

from clinical_re_bilingual.data.i2b2 import (
    I2B2Layout,
    I2B2ParseError,
    parse_concept_line,
    parse_i2b2_directory,
    parse_relation_line,
)
from clinical_re_bilingual.schema import Language, RelationLabel


def test_parse_concept_line() -> None:
    concept = parse_concept_line('c="synthetic finding" 1:1 1:2||t="problem"')
    assert concept.text == "synthetic finding"
    assert concept.start_token == 1
    assert concept.end_token == 2
    assert concept.entity_type == "problem"


def test_parse_relation_line_preserves_direction() -> None:
    relation = parse_relation_line(
        'c="synthetic therapy" 1:0 1:1||r="TrAP"||c="synthetic condition" 1:4 1:5'
    )
    assert relation.subject.text == "synthetic therapy"
    assert relation.object.text == "synthetic condition"
    assert relation.label is RelationLabel.TrAP


def test_cross_sentence_relation_is_rejected() -> None:
    with pytest.raises(I2B2ParseError, match="cross-sentence"):
        parse_relation_line('c="a" 1:0 1:0||r="PIP"||c="b" 2:0 2:0')


def test_parse_synthetic_directory(tmp_path: Path) -> None:
    concepts = tmp_path / "concepts"
    relations = tmp_path / "relations"
    texts = tmp_path / "texts"
    for directory in (concepts, relations, texts):
        directory.mkdir()

    (concepts / "record-1.con").write_text(
        'c="synthetic therapy" 1:0 1:1||t="treatment"\n'
        'c="synthetic condition" 1:4 1:5||t="problem"\n',
        encoding="utf-8",
    )
    (relations / "record-1.rel").write_text(
        'c="synthetic therapy" 1:0 1:1||r="TrAP"||c="synthetic condition" 1:4 1:5\n',
        encoding="utf-8",
    )
    (texts / "record-1.txt").write_text(
        "synthetic therapy was used for synthetic condition .\n", encoding="utf-8"
    )

    examples = parse_i2b2_directory(I2B2Layout(concepts, relations, texts), Language.ENGLISH)
    assert len(examples) == 1
    assert examples[0].relation_type == "TrAP"
    assert examples[0].e1_type == "treatment"
    assert examples[0].e2_type == "problem"
    assert examples[0].example_id.startswith("en_")
