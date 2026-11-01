"""Parser for the original 2010 i2b2/VA concept, relation, and text files.

The parser operates on locally held corpus files obtained under the applicable
data use agreement. Text-bearing derived artifacts are excluded from version
control.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from clinical_re_bilingual.schema import Language, RelationExample, RelationLabel

_CONCEPT_PATTERN = re.compile(
    r'^c="(?P<text>.*)"\s+(?P<start_line>\d+):(?P<start_token>\d+)\s+'
    r'(?P<end_line>\d+):(?P<end_token>\d+)\|\|t="(?P<entity_type>[^"]+)"$'
)
_RELATION_PART_PATTERN = re.compile(
    r'^c="(?P<text>.*)"\s+(?P<start_line>\d+):(?P<start_token>\d+)\s+'
    r"(?P<end_line>\d+):(?P<end_token>\d+)$"
)
_RELATION_LABEL_PATTERN = re.compile(r'^r="(?P<label>[^"]+)"$')


class I2B2ParseError(ValueError):
    """Raised when an annotation cannot be parsed without guessing."""


@dataclass(frozen=True)
class Span:
    text: str
    line: int
    start_token: int
    end_token: int


@dataclass(frozen=True)
class Concept(Span):
    entity_type: str


@dataclass(frozen=True)
class ParsedRelation:
    subject: Span
    label: RelationLabel
    object: Span


@dataclass(frozen=True)
class I2B2Layout:
    concepts_dir: Path
    relations_dir: Path
    text_dir: Path

    def validate(self) -> None:
        for name, directory in (
            ("concepts", self.concepts_dir),
            ("relations", self.relations_dir),
            ("text", self.text_dir),
        ):
            if not directory.is_dir():
                raise FileNotFoundError(f"{name} directory does not exist: {directory}")


@dataclass(frozen=True)
class ParsedExample:
    example_id: str
    language: str
    source_document_id: str
    source_line: int
    original_sentence: str
    e1: str
    e1_type: str
    e1_from: int
    e1_to: int
    e2: str
    e2_type: str
    e2_from: int
    e2_to: int
    relation_type: str

    def to_relation_example(self) -> RelationExample:
        return RelationExample(
            example_id=self.example_id,
            language=Language(self.language),
            sentence=self.original_sentence,
            subject=self.e1,
            subject_type=self.e1_type,
            object=self.e2,
            object_type=self.e2_type,
            label=RelationLabel(self.relation_type),
            subject_start=self.e1_from,
            subject_end=self.e1_to,
            object_start=self.e2_from,
            object_end=self.e2_to,
        )


def parse_concept_line(line: str) -> Concept:
    match = _CONCEPT_PATTERN.fullmatch(line.strip())
    if match is None:
        raise I2B2ParseError(f"invalid concept annotation: {line!r}")
    values = match.groupdict()
    start_line = int(values["start_line"])
    end_line = int(values["end_line"])
    if start_line != end_line:
        raise I2B2ParseError("multi-line concepts are not supported by the sentence-level task")
    return Concept(
        text=values["text"],
        line=start_line,
        start_token=int(values["start_token"]),
        end_token=int(values["end_token"]),
        entity_type=values["entity_type"],
    )


def _parse_relation_span(value: str) -> Span:
    match = _RELATION_PART_PATTERN.fullmatch(value.strip())
    if match is None:
        raise I2B2ParseError(f"invalid relation concept: {value!r}")
    fields = match.groupdict()
    start_line = int(fields["start_line"])
    end_line = int(fields["end_line"])
    if start_line != end_line:
        raise I2B2ParseError("multi-line relation arguments are not supported")
    return Span(
        text=fields["text"],
        line=start_line,
        start_token=int(fields["start_token"]),
        end_token=int(fields["end_token"]),
    )


def parse_relation_line(line: str) -> ParsedRelation:
    parts = line.strip().split("||")
    if len(parts) != 3:
        raise I2B2ParseError(f"relation annotation must contain three parts: {line!r}")
    subject = _parse_relation_span(parts[0])
    label_match = _RELATION_LABEL_PATTERN.fullmatch(parts[1].strip())
    if label_match is None:
        raise I2B2ParseError(f"invalid relation label: {parts[1]!r}")
    try:
        label = RelationLabel(label_match.group("label"))
    except ValueError as error:
        raise I2B2ParseError(f"unsupported relation label: {parts[1]!r}") from error
    object_ = _parse_relation_span(parts[2])
    if subject.line != object_.line:
        raise I2B2ParseError("cross-sentence relations are outside this task definition")
    return ParsedRelation(subject=subject, label=label, object=object_)


def _concept_key(span: Span) -> tuple[int, int, int]:
    return span.line, span.start_token, span.end_token


def _example_id(language: Language, document_id: str, relation_index: int) -> str:
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:12]
    return f"{language.value}_{digest}_{relation_index:04d}"


def parse_document(
    concept_path: Path,
    relation_path: Path,
    text_path: Path,
    language: Language,
) -> list[ParsedExample]:
    lines = text_path.read_text(encoding="utf-8").splitlines()
    concepts: dict[tuple[int, int, int], Concept] = {}
    for raw_line in concept_path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            concept = parse_concept_line(raw_line)
            key = _concept_key(concept)
            if key in concepts and concepts[key] != concept:
                raise I2B2ParseError(f"conflicting concepts at {key} in {concept_path}")
            concepts[key] = concept

    examples: list[ParsedExample] = []
    relation_lines = relation_path.read_text(encoding="utf-8").splitlines()
    for relation_index, raw_line in enumerate(relation_lines):
        if not raw_line.strip():
            continue
        relation = parse_relation_line(raw_line)
        line_index = relation.subject.line - 1
        if line_index < 0 or line_index >= len(lines):
            raise I2B2ParseError(
                f"relation references missing line {relation.subject.line} in {text_path}"
            )
        subject_concept = concepts.get(_concept_key(relation.subject))
        object_concept = concepts.get(_concept_key(relation.object))
        if subject_concept is None or object_concept is None:
            raise I2B2ParseError(
                f"relation argument has no matching concept annotation in {concept_path}"
            )
        if (
            subject_concept.text != relation.subject.text
            or object_concept.text != relation.object.text
        ):
            raise I2B2ParseError("relation and concept text disagree for the same token span")

        document_id = relation_path.stem
        example = ParsedExample(
            example_id=_example_id(language, document_id, relation_index),
            language=language.value,
            source_document_id=document_id,
            source_line=relation.subject.line,
            original_sentence=lines[line_index].strip(),
            e1=relation.subject.text,
            e1_type=subject_concept.entity_type,
            e1_from=relation.subject.start_token,
            e1_to=relation.subject.end_token,
            e2=relation.object.text,
            e2_type=object_concept.entity_type,
            e2_from=relation.object.start_token,
            e2_to=relation.object.end_token,
            relation_type=relation.label.value,
        )
        example.to_relation_example()
        examples.append(example)
    return examples


def parse_i2b2_directory(layout: I2B2Layout, language: Language) -> list[ParsedExample]:
    layout.validate()
    examples: list[ParsedExample] = []
    for relation_path in sorted(path for path in layout.relations_dir.iterdir() if path.is_file()):
        document_id = relation_path.stem
        concept_path = layout.concepts_dir / f"{document_id}.con"
        text_path = layout.text_dir / f"{document_id}.txt"
        if not concept_path.is_file() or not text_path.is_file():
            raise FileNotFoundError(
                f"missing concept or text file for relation document {document_id!r}"
            )
        examples.extend(parse_document(concept_path, relation_path, text_path, language))
    return examples


def write_tsv(examples: list[ParsedExample], output_path: Path) -> None:
    """Write protected parsed data to an explicitly supplied local path."""
    if not examples:
        raise ValueError("refusing to write an empty parsed dataset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(examples[0])), delimiter="\t")
        writer.writeheader()
        writer.writerows(asdict(example) for example in examples)


def read_relation_examples_tsv(path: Path) -> list[RelationExample]:
    """Read protected parsed records while exposing only validated model records."""
    required = {
        "example_id",
        "language",
        "original_sentence",
        "e1",
        "e1_type",
        "e1_from",
        "e1_to",
        "e2",
        "e2_type",
        "e2_from",
        "e2_to",
        "relation_type",
    }
    examples: list[RelationExample] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"parsed dataset is missing columns: {sorted(missing)}")
        for row in reader:
            examples.append(
                RelationExample(
                    example_id=row["example_id"],
                    language=Language(row["language"]),
                    sentence=row["original_sentence"],
                    subject=row["e1"],
                    subject_type=row["e1_type"],
                    object=row["e2"],
                    object_type=row["e2_type"],
                    label=RelationLabel(row["relation_type"]),
                    subject_start=int(row["e1_from"]),
                    subject_end=int(row["e1_to"]),
                    object_start=int(row["e2_from"]),
                    object_end=int(row["e2_to"]),
                )
            )
    if not examples:
        raise ValueError(f"parsed dataset is empty: {path}")
    return examples
