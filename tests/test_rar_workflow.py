import json
from pathlib import Path

import pytest

from clinical_re_bilingual.config import read_models_config, read_rar_config
from clinical_re_bilingual.data.manifests import create_split_manifest
from clinical_re_bilingual.retrieval.io import write_rankings_jsonl
from clinical_re_bilingual.schema import (
    Language,
    RelationExample,
    RelationLabel,
    RetrievalRanking,
)
from clinical_re_bilingual.workflows.rar import validate_workflow_inputs


def example(example_id: str, label: RelationLabel, *, test: bool = False) -> RelationExample:
    return RelationExample(
        example_id=example_id,
        language=Language.ENGLISH,
        sentence=f"synthetic {'test' if test else 'training'} sentence",
        subject="synthetic subject",
        subject_type="treatment",
        object="synthetic object",
        object_type="problem",
        label=label,
    )


def balanced_pool() -> list[RelationExample]:
    return [
        example(f"train-{label.value}-{index}", label)
        for label in RelationLabel
        for index in range(4)
    ]


def test_rar_and_model_configuration_loads() -> None:
    root = Path(__file__).parents[1]
    rar = read_rar_config(root / "config/rar.yaml")
    models = read_models_config(root / "config/models.yaml")
    assert rar.train.epochs == 3
    assert rar.pair_generation.positive_neighbors == 5
    assert rar.pair_generation.negative_neighbors == 5
    assert rar.retrieval.candidate_pool_size == 20
    assert models.retrievers.rar.en == "roberta-large"
    assert models.retrievers.rar.tr == "TURKCELL/roberta-base-turkish-uncased"


def test_workflow_partitions_match_manifest() -> None:
    pool = balanced_pool()
    test = [example(f"test-{label.value}", label, test=True) for label in RelationLabel]
    manifest = create_split_manifest(
        pool, test, dataset_name="synthetic", dev_fraction=0.25, seed=42
    )
    train, validation = validate_workflow_inputs(pool, test, manifest, Language.ENGLISH)
    assert [item.example_id for item in train] == manifest.train_ids
    assert [item.example_id for item in validation] == manifest.dev_ids


def test_workflow_rejects_changed_data() -> None:
    pool = balanced_pool()
    test = [example(f"test-{label.value}", label, test=True) for label in RelationLabel]
    manifest = create_split_manifest(
        pool, test, dataset_name="synthetic", dev_fraction=0.25, seed=42
    )
    changed = pool.copy()
    changed[0] = changed[0].model_copy(update={"label": RelationLabel.PIP})
    with pytest.raises(ValueError, match="fingerprint"):
        validate_workflow_inputs(changed, test, manifest, Language.ENGLISH)


def test_ranking_output_contains_ids_but_no_text(tmp_path: Path) -> None:
    ranking = RetrievalRanking(
        test_example_id="test-1",
        retriever="rar",
        language=Language.ENGLISH,
        ranked_train_ids=["train-1", "train-2"],
        scores=[0.9, 0.8],
    )
    path = tmp_path / "rankings.jsonl"
    assert write_rankings_jsonl([ranking], path) == 1
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["ranked_train_ids"] == ["train-1", "train-2"]
    assert "sentence" not in record
