"""End-to-end training and indexing workflow for RAR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clinical_re_bilingual.config import read_models_config, read_rar_config
from clinical_re_bilingual.data.i2b2 import read_relation_examples_tsv
from clinical_re_bilingual.data.manifests import (
    SplitManifest,
    dataset_fingerprint,
    read_manifest,
)
from clinical_re_bilingual.retrieval.encoders import HuggingFaceCLSEncoder
from clinical_re_bilingual.retrieval.io import write_rankings_jsonl
from clinical_re_bilingual.retrieval.rar import RARPairBuilder, RARRetriever
from clinical_re_bilingual.retrieval.simcse import SimCSETrainer
from clinical_re_bilingual.schema import Language, RelationExample


@dataclass(frozen=True)
class RARWorkflowResult:
    checkpoint_dir: Path
    rankings_path: Path
    training_examples: int
    validation_examples: int
    indexed_examples: int
    test_examples: int


def select_manifest_partition(
    examples: list[RelationExample], ids: list[str], partition_name: str
) -> list[RelationExample]:
    by_id = {example.example_id: example for example in examples}
    if len(by_id) != len(examples):
        raise ValueError("source examples contain duplicate IDs")
    missing = [example_id for example_id in ids if example_id not in by_id]
    if missing:
        raise ValueError(f"{partition_name} IDs are missing from the dataset: {missing[:3]}")
    return [by_id[example_id] for example_id in ids]


def validate_workflow_inputs(
    training_pool: list[RelationExample],
    official_test: list[RelationExample],
    manifest: SplitManifest,
    language: Language,
) -> tuple[list[RelationExample], list[RelationExample]]:
    all_examples = training_pool + official_test
    wrong_language = [
        example.example_id for example in all_examples if example.language is not language
    ]
    if wrong_language:
        raise ValueError(f"language mismatch for IDs: {wrong_language[:3]}")
    observed_fingerprint = dataset_fingerprint(all_examples)
    if observed_fingerprint != manifest.dataset_fingerprint:
        raise ValueError("dataset fingerprint does not match the split manifest")
    test_ids = [example.example_id for example in official_test]
    if test_ids != manifest.test_ids:
        raise ValueError("official test order does not match the split manifest")
    train = select_manifest_partition(training_pool, manifest.train_ids, "train")
    validation = select_manifest_partition(training_pool, manifest.dev_ids, "dev")
    return train, validation


def run_rar_workflow(
    *,
    training_pool_path: Path,
    official_test_path: Path,
    manifest_path: Path,
    language: Language,
    rar_config_path: Path,
    models_config_path: Path,
    output_dir: Path,
    device: str | None = None,
) -> RARWorkflowResult:
    """Train RAR and generate public-safe top-20 test-to-training rankings."""

    config = read_rar_config(rar_config_path)
    models = read_models_config(models_config_path)
    training_pool = read_relation_examples_tsv(training_pool_path)
    official_test = read_relation_examples_tsv(official_test_path)
    manifest = read_manifest(manifest_path)
    train, validation = validate_workflow_inputs(training_pool, official_test, manifest, language)

    base_model = models.retrievers.rar.for_language(language)
    initial_encoder = HuggingFaceCLSEncoder(
        base_model,
        batch_size=config.train.batch_size,
        device=device,
    )
    pair_config = config.pair_generation
    triples = RARPairBuilder(
        initial_encoder,
        positive_neighbors=pair_config.positive_neighbors,
        negative_neighbors=pair_config.negative_neighbors,
        sentence_weight=pair_config.sentence_weight,
        entity_weight=pair_config.entity_weight,
        relation_weight=pair_config.relation_weight,
        subject_weight=pair_config.subject_weight,
        object_weight=pair_config.object_weight,
    ).build(train)

    checkpoint_dir = output_dir / "checkpoint"
    trainer = SimCSETrainer(
        base_model,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        temperature=config.train.temperature,
        max_length=config.train.max_length,
        seed=config.seed,
        device=device,
    )
    trainer.train(
        triples,
        checkpoint_dir,
        validation_train_examples=train,
        validation_examples=validation,
    )

    trained_encoder = HuggingFaceCLSEncoder(
        str(checkpoint_dir),
        batch_size=config.train.batch_size,
        device=device,
    )
    retrieval_config = config.retrieval
    retriever = RARRetriever(
        trained_encoder,
        candidate_pool_size=retrieval_config.candidate_pool_size,
        sentence_weight=retrieval_config.sentence_weight,
        entity_weight=retrieval_config.entity_weight,
        subject_weight=retrieval_config.subject_weight,
        object_weight=retrieval_config.object_weight,
    ).fit(training_pool)
    rankings_path = output_dir / "rankings.jsonl"
    write_rankings_jsonl((retriever.rank(example) for example in official_test), rankings_path)

    summary = {
        "language": language.value,
        "base_model": base_model,
        "training_examples": len(train),
        "validation_examples": len(validation),
        "indexed_examples": len(training_pool),
        "test_examples": len(official_test),
        "candidate_pool_size": retrieval_config.candidate_pool_size,
        "checkpoint_metric": config.train.checkpoint_metric,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return RARWorkflowResult(
        checkpoint_dir=checkpoint_dir,
        rankings_path=rankings_path,
        training_examples=len(train),
        validation_examples=len(validation),
        indexed_examples=len(training_pool),
        test_examples=len(official_test),
    )
