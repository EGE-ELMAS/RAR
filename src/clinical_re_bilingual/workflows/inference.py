"""Provider-independent execution of one relation-extraction experiment."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

from clinical_re_bilingual.data.i2b2 import read_relation_examples_tsv
from clinical_re_bilingual.inference.base import TextGenerationClient
from clinical_re_bilingual.inference.io import write_predictions_jsonl
from clinical_re_bilingual.inference.providers import (
    GeminiClient,
    OpenAICompatibleChatClient,
    OpenAIResponsesClient,
)
from clinical_re_bilingual.inference.runner import ExperimentRunner
from clinical_re_bilingual.prompting.builder import PromptStrategy
from clinical_re_bilingual.schema import PredictionRecord, RelationExample, RetrievalRanking


def create_client(provider: str, model_name: str) -> TextGenerationClient:
    if provider == "openai":
        return OpenAIResponsesClient(model_name)
    if provider == "gemini":
        return GeminiClient(model_name)
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return OpenAICompatibleChatClient(model_name, api_key=api_key)
    raise ValueError(f"unsupported provider: {provider}")


def _read_rankings_jsonl(path: Path) -> list[RetrievalRanking]:
    rankings = [
        RetrievalRanking.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rankings:
        raise ValueError(f"ranking file is empty: {path}")
    ids = [ranking.test_example_id for ranking in rankings]
    if len(ids) != len(set(ids)):
        raise ValueError("ranking file contains duplicate test IDs")
    return rankings


def predict_dataset(
    *,
    client: TextGenerationClient,
    training_examples: Sequence[RelationExample],
    test_examples: Sequence[RelationExample],
    rankings: Sequence[RetrievalRanking] | None,
    num_shots: int,
    strategy: PromptStrategy,
    temperature: float,
    run_date: date,
    rationales: Mapping[str, str] | None = None,
) -> list[PredictionRecord]:
    if strategy is PromptStrategy.ZS_STATIC_COT:
        if num_shots != 0:
            raise ValueError("zero-shot static CoT requires num_shots=0")
        if rankings:
            raise ValueError("zero-shot static CoT does not use rankings")
        rankings_by_id: dict[str, RetrievalRanking] = {}
    else:
        if rankings is None:
            raise ValueError(f"strategy {strategy.value} requires rankings")
        rankings_by_id = {ranking.test_example_id: ranking for ranking in rankings}
        if len(rankings_by_id) != len(rankings):
            raise ValueError("rankings contain duplicate test IDs")
        expected_ids = [example.example_id for example in test_examples]
        if set(rankings_by_id) != set(expected_ids):
            missing = [
                example_id for example_id in expected_ids if example_id not in rankings_by_id
            ]
            extra = [example_id for example_id in rankings_by_id if example_id not in expected_ids]
            raise ValueError(f"ranking/test ID mismatch; missing={missing[:3]}, extra={extra[:3]}")
    runner = ExperimentRunner(client, temperature=temperature)
    return [
        runner.predict(
            example,
            training_examples,
            rankings_by_id.get(example.example_id),
            num_shots=num_shots,
            strategy=strategy,
            run_date=run_date,
            rationales=rationales,
        )
        for example in test_examples
    ]


def run_inference_workflow(
    *,
    training_pool_path: Path,
    official_test_path: Path,
    rankings_path: Path | None,
    provider: str,
    model_name: str,
    num_shots: int,
    strategy: PromptStrategy,
    temperature: float,
    output_path: Path,
    rationales_path: Path | None = None,
    run_date: date | None = None,
) -> int:
    training = read_relation_examples_tsv(training_pool_path)
    test = read_relation_examples_tsv(official_test_path)
    rankings = _read_rankings_jsonl(rankings_path) if rankings_path is not None else None
    rationales = None
    if rationales_path is not None:
        loaded = json.loads(rationales_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in loaded.items()
        ):
            raise ValueError("rationale file must be a JSON object mapping IDs to strings")
        rationales = loaded
    records = predict_dataset(
        client=create_client(provider, model_name),
        training_examples=training,
        test_examples=test,
        rankings=rankings,
        num_shots=num_shots,
        strategy=strategy,
        temperature=temperature,
        run_date=run_date or date.today(),
        rationales=rationales,
    )
    return write_predictions_jsonl(records, output_path)
