from pathlib import Path

import pytest

from clinical_re_bilingual.data.manifests import (
    SplitManifest,
    create_positional_alignment,
    create_split_manifest,
)
from clinical_re_bilingual.schema import Language, RelationExample, RelationLabel


def make_example(index: int, language: Language, label: RelationLabel) -> RelationExample:
    return RelationExample(
        example_id=f"{language.value}_{index:04d}",
        language=language,
        sentence=f"synthetic sentence {index}",
        subject="synthetic subject",
        subject_type="treatment",
        object="synthetic object",
        object_type="problem",
        label=label,
    )


def balanced_examples(language: Language, copies_per_label: int, offset: int = 0):
    examples = []
    index = offset
    for label in RelationLabel:
        for _ in range(copies_per_label):
            examples.append(make_example(index, language, label))
            index += 1
    return examples


def test_split_manifest_is_deterministic_and_preserves_test_order() -> None:
    pool = balanced_examples(Language.ENGLISH, copies_per_label=4)
    test = balanced_examples(Language.ENGLISH, copies_per_label=1, offset=1000)
    first = create_split_manifest(
        pool, test, dataset_name="synthetic-en", dev_fraction=0.25, seed=42
    )
    second = create_split_manifest(
        pool, test, dataset_name="synthetic-en", dev_fraction=0.25, seed=42
    )
    assert first == second
    assert first.test_ids == [example.example_id for example in test]
    assert not (set(first.train_ids) & set(first.dev_ids))


def test_manifest_rejects_partition_leakage() -> None:
    with pytest.raises(ValueError, match="leakage"):
        SplitManifest(
            dataset_name="synthetic",
            seed=42,
            dev_fraction=0.2,
            dataset_fingerprint="0" * 64,
            train_ids=["shared"],
            dev_ids=["shared"],
            test_ids=["test"],
            label_counts={
                "train": {RelationLabel.TrAP: 1},
                "dev": {RelationLabel.TrAP: 1},
                "test": {RelationLabel.TrAP: 1},
            },
        )


def test_manifest_json_contains_no_clinical_text(tmp_path: Path) -> None:
    pool = balanced_examples(Language.ENGLISH, copies_per_label=4)
    test = balanced_examples(Language.ENGLISH, copies_per_label=1, offset=1000)
    manifest = create_split_manifest(
        pool, test, dataset_name="synthetic-en", dev_fraction=0.25, seed=42
    )
    path = tmp_path / "split.json"
    manifest.write_json(path)
    serialized = path.read_text(encoding="utf-8")
    assert "synthetic sentence" not in serialized
    assert "synthetic subject" not in serialized


def test_bilingual_alignment_requires_matching_labels() -> None:
    english = [make_example(1, Language.ENGLISH, RelationLabel.TrAP)]
    turkish = [make_example(1, Language.TURKISH, RelationLabel.TrCP)]
    with pytest.raises(ValueError, match="label mismatch"):
        create_positional_alignment(english, turkish, dataset_name="synthetic")


def test_bilingual_alignment_is_one_to_one() -> None:
    english = balanced_examples(Language.ENGLISH, copies_per_label=1)
    turkish = balanced_examples(Language.TURKISH, copies_per_label=1)
    manifest = create_positional_alignment(english, turkish, dataset_name="synthetic")
    assert len(manifest.pairs) == len(RelationLabel)
    assert manifest.pairs[0].alignment_id == "pair_000000"
