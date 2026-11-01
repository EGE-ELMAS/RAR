"""Training and evaluation workflow for supervised BERT and PURE baselines."""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import numpy as np
import torch

from clinical_re_bilingual.data.i2b2 import read_relation_examples_tsv
from clinical_re_bilingual.data.manifests import read_manifest
from clinical_re_bilingual.evaluation.io import write_evaluation_outputs
from clinical_re_bilingual.inference.io import write_predictions_jsonl
from clinical_re_bilingual.models.baselines import train_bert_baseline, train_pure_baseline
from clinical_re_bilingual.models.config import (
    read_baseline_checkpoints,
    read_baselines_config,
)
from clinical_re_bilingual.schema import Language, PredictionRecord
from clinical_re_bilingual.workflows.rar import validate_workflow_inputs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_baseline_workflow(
    *,
    baseline: str,
    training_pool_path: Path,
    official_test_path: Path,
    manifest_path: Path,
    language: Language,
    baselines_config_path: Path,
    models_config_path: Path,
    output_dir: Path,
    device: str | None = None,
    run_date: date | None = None,
) -> int:
    if baseline not in {"bert", "pure"}:
        raise ValueError("baseline must be 'bert' or 'pure'")
    config = read_baselines_config(baselines_config_path)
    models = read_baseline_checkpoints(models_config_path)
    training_pool = read_relation_examples_tsv(training_pool_path)
    official_test = read_relation_examples_tsv(official_test_path)
    manifest = read_manifest(manifest_path)
    train, validation = validate_workflow_inputs(training_pool, official_test, manifest, language)
    _set_seed(config.seed)
    checkpoint_dir = output_dir / "checkpoint"
    if baseline == "bert":
        model_name = models.bert.for_language(language)
        result = train_bert_baseline(
            train,
            validation,
            official_test,
            model_name=model_name,
            config=config.bert,
            output_dir=checkpoint_dir,
            device_name=device,
        )
    else:
        model_name = models.pure.for_language(language)
        result = train_pure_baseline(
            train,
            validation,
            official_test,
            model_name=model_name,
            config=config.pure,
            output_dir=checkpoint_dir,
            device_name=device,
        )
    records = [
        PredictionRecord(
            example_id=example.example_id,
            gold_label=example.label,
            predicted_label=predicted,
            model=f"{baseline}:{model_name}",
            language=language,
            retriever="none",
            num_shots=0,
            prompt_strategy="fine_tuned",
            temperature=0.0,
            run_date=run_date or date.today(),
        )
        for example, predicted in zip(official_test, result.predicted_labels, strict=True)
    ]
    predictions_path = output_dir / "predictions.jsonl"
    count = write_predictions_jsonl(records, predictions_path)
    write_evaluation_outputs(
        records,
        output_dir / "metrics.json",
        output_dir / "confusion_matrix.csv",
    )
    return count
