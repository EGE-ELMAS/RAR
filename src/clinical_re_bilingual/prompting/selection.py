"""Resolve the first k ranked IDs into protected in-memory demonstrations."""

from collections.abc import Sequence

from clinical_re_bilingual.schema import RelationExample, RetrievalRanking


def demonstrations_from_ranking(
    ranking: RetrievalRanking,
    training_examples: Sequence[RelationExample],
    k: int,
) -> list[RelationExample]:
    by_id = {example.example_id: example for example in training_examples}
    if len(by_id) != len(training_examples):
        raise ValueError("training example IDs must be unique")
    selected_ids = ranking.top_k(k)
    missing = [example_id for example_id in selected_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"ranking references missing training IDs: {missing[:3]}")
    return [by_id[example_id] for example_id in selected_ids]
