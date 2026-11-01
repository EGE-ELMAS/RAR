from datetime import date

import pytest

from clinical_re_bilingual.schema import (
    Language,
    PredictionRecord,
    RelationLabel,
    RetrievalRanking,
)


def test_top_k_preserves_rank_order() -> None:
    ranking = RetrievalRanking(
        test_example_id="en_test_0001",
        retriever="rar",
        language=Language.ENGLISH,
        ranked_train_ids=["en_train_0003", "en_train_0008", "en_train_0010"],
        scores=[0.9, 0.8, 0.7],
    )
    assert ranking.top_k(2) == ["en_train_0003", "en_train_0008"]


def test_duplicate_ranking_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        RetrievalRanking(
            test_example_id="en_test_0001",
            retriever="rar",
            language=Language.ENGLISH,
            ranked_train_ids=["en_train_0003", "en_train_0003"],
        )


def test_public_prediction_schema() -> None:
    record = PredictionRecord(
        example_id="en_test_0001",
        gold_label=RelationLabel.TrAP,
        predicted_label=RelationLabel.TrAP,
        model="gemini-1.5-flash-latest",
        language=Language.ENGLISH,
        retriever="rar",
        num_shots=5,
        prompt_strategy="standard",
        temperature=0.0,
        run_date=date(2026, 10, 28),
    )
    assert "sentence" not in record.model_dump()
