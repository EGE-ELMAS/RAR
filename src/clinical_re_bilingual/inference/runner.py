"""Execute relation classification without retaining prompts or raw responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from clinical_re_bilingual.inference.base import TextGenerationClient
from clinical_re_bilingual.prompting.builder import PromptBuilder, PromptStrategy
from clinical_re_bilingual.prompting.labels import parse_relation_label
from clinical_re_bilingual.prompting.selection import demonstrations_from_ranking
from clinical_re_bilingual.schema import PredictionRecord, RelationExample, RetrievalRanking


class ExperimentRunner:
    def __init__(self, client: TextGenerationClient, *, temperature: float = 0.0) -> None:
        self.client = client
        self.temperature = temperature

    def predict(
        self,
        query: RelationExample,
        training_examples: Sequence[RelationExample],
        ranking: RetrievalRanking | None,
        *,
        num_shots: int,
        strategy: PromptStrategy,
        run_date: date,
        rationales: Mapping[str, str] | None = None,
    ) -> PredictionRecord:
        if strategy is PromptStrategy.ZS_STATIC_COT:
            if num_shots != 0:
                raise ValueError("zero-shot static CoT requires num_shots=0")
            if ranking is not None:
                raise ValueError("zero-shot static CoT does not use a ranking")
            demonstrations = []
            retriever = "none"
        else:
            if num_shots < 1:
                raise ValueError(f"strategy {strategy.value} requires at least one shot")
            if ranking is None:
                raise ValueError(f"strategy {strategy.value} requires a ranking")
            if ranking.test_example_id != query.example_id:
                raise ValueError("ranking does not belong to the query example")
            demonstrations = demonstrations_from_ranking(ranking, training_examples, num_shots)
            retriever = ranking.retriever
        prompt = PromptBuilder(query.language, strategy).build(
            query, demonstrations, rationales=rationales
        )
        response = self.client.generate(prompt, temperature=self.temperature)
        predicted = parse_relation_label(response)
        return PredictionRecord(
            example_id=query.example_id,
            gold_label=query.label,
            predicted_label=predicted,
            model=self.client.model_name,
            language=query.language,
            retriever=retriever,
            num_shots=num_shots,
            prompt_strategy=strategy.value,
            temperature=self.temperature,
            run_date=run_date,
        )
