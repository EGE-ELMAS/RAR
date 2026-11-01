"""Public-safe split and bilingual-alignment manifests.

Manifests contain opaque example IDs, labels, and reproducibility metadata only.
They must never contain clinical sentences or entity text.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.model_selection import train_test_split

from clinical_re_bilingual.schema import Language, RelationExample, RelationLabel


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    dataset_name: str = Field(min_length=1)
    seed: int
    dev_fraction: float = Field(gt=0.0, lt=1.0)
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_ids: list[str]
    dev_ids: list[str]
    test_ids: list[str]
    label_counts: dict[str, dict[RelationLabel, int]]

    @model_validator(mode="after")
    def validate_partitions(self) -> SplitManifest:
        partitions = {
            "train": self.train_ids,
            "dev": self.dev_ids,
            "test": self.test_ids,
        }
        for name, ids in partitions.items():
            if not ids:
                raise ValueError(f"{name} partition must not be empty")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} partition contains duplicate IDs")
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = set(partitions[left]) & set(partitions[right])
            if overlap:
                raise ValueError(f"{left}/{right} leakage detected: {sorted(overlap)[:3]}")
        expected_sizes = {name: len(ids) for name, ids in partitions.items()}
        for name, expected_size in expected_sizes.items():
            observed_size = sum(self.label_counts.get(name, {}).values())
            if observed_size != expected_size:
                raise ValueError(
                    f"{name} label counts total {observed_size}, expected {expected_size}"
                )
        return self

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


class AlignmentPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alignment_id: str = Field(pattern=r"^pair_\d{6}$")
    english_id: str = Field(min_length=1)
    turkish_id: str = Field(min_length=1)
    label: RelationLabel


class AlignmentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    dataset_name: str = Field(min_length=1)
    pairs: list[AlignmentPair]

    @model_validator(mode="after")
    def validate_one_to_one(self) -> AlignmentManifest:
        if not self.pairs:
            raise ValueError("alignment manifest must contain at least one pair")
        alignment_ids = [pair.alignment_id for pair in self.pairs]
        english_ids = [pair.english_id for pair in self.pairs]
        turkish_ids = [pair.turkish_id for pair in self.pairs]
        for name, values in (
            ("alignment", alignment_ids),
            ("English", english_ids),
            ("Turkish", turkish_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} IDs must be one-to-one")
        return self

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _validate_unique_examples(
    examples: list[RelationExample], expected_language: Language | None = None
) -> None:
    if not examples:
        raise ValueError("example collection must not be empty")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example IDs must be unique")
    if expected_language is not None:
        wrong = [
            example.example_id for example in examples if example.language != expected_language
        ]
        if wrong:
            raise ValueError(
                f"expected {expected_language.value} examples; mismatched IDs: {wrong[:3]}"
            )


def dataset_fingerprint(examples: list[RelationExample]) -> str:
    """Hash safe identifiers and labels without hashing protected text."""
    _validate_unique_examples(examples)
    canonical = "\n".join(
        f"{example.example_id}\t{example.language.value}\t{example.label.value}"
        for example in sorted(examples, key=lambda item: item.example_id)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _label_counts(examples: list[RelationExample]) -> dict[RelationLabel, int]:
    counts = Counter(example.label for example in examples)
    return {label: counts[label] for label in sorted(counts, key=lambda item: item.value)}


def create_split_manifest(
    training_pool: list[RelationExample],
    official_test: list[RelationExample],
    *,
    dataset_name: str,
    dev_fraction: float,
    seed: int,
) -> SplitManifest:
    """Split only the training pool; preserve the official test set unchanged."""
    _validate_unique_examples(training_pool)
    _validate_unique_examples(official_test)
    pool_ids = {example.example_id for example in training_pool}
    test_ids = {example.example_id for example in official_test}
    overlap = pool_ids & test_ids
    if overlap:
        raise ValueError(f"training/test leakage detected: {sorted(overlap)[:3]}")

    label_frequencies = Counter(example.label for example in training_pool)
    sparse_labels = [label.value for label, count in label_frequencies.items() if count < 2]
    if sparse_labels:
        raise ValueError(
            "stratified validation requires at least two training examples per label; "
            f"insufficient labels: {sorted(sparse_labels)}"
        )

    indices = list(range(len(training_pool)))
    labels = [example.label.value for example in training_pool]
    train_indices, dev_indices = train_test_split(
        indices,
        test_size=dev_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_index_set = set(train_indices)
    dev_index_set = set(dev_indices)
    train_examples = [
        example for index, example in enumerate(training_pool) if index in train_index_set
    ]
    dev_examples = [
        example for index, example in enumerate(training_pool) if index in dev_index_set
    ]

    return SplitManifest(
        dataset_name=dataset_name,
        seed=seed,
        dev_fraction=dev_fraction,
        dataset_fingerprint=dataset_fingerprint(training_pool + official_test),
        train_ids=[example.example_id for example in train_examples],
        dev_ids=[example.example_id for example in dev_examples],
        test_ids=[example.example_id for example in official_test],
        label_counts={
            "train": _label_counts(train_examples),
            "dev": _label_counts(dev_examples),
            "test": _label_counts(official_test),
        },
    )


def create_positional_alignment(
    english: list[RelationExample],
    turkish: list[RelationExample],
    *,
    dataset_name: str,
) -> AlignmentManifest:
    """Create a one-to-one manifest after verifying aligned labels and languages."""
    _validate_unique_examples(english, Language.ENGLISH)
    _validate_unique_examples(turkish, Language.TURKISH)
    if len(english) != len(turkish):
        raise ValueError(
            f"alignment length mismatch: {len(english)} English vs {len(turkish)} Turkish"
        )
    pairs: list[AlignmentPair] = []
    for index, (english_example, turkish_example) in enumerate(zip(english, turkish, strict=True)):
        if english_example.label != turkish_example.label:
            raise ValueError(
                f"label mismatch at position {index}: "
                f"{english_example.label.value} vs {turkish_example.label.value}"
            )
        pairs.append(
            AlignmentPair(
                alignment_id=f"pair_{index:06d}",
                english_id=english_example.example_id,
                turkish_id=turkish_example.example_id,
                label=english_example.label,
            )
        )
    return AlignmentManifest(dataset_name=dataset_name, pairs=pairs)


def read_manifest(path: Path) -> SplitManifest:
    return SplitManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
