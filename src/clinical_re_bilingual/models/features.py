"""Stable label mappings and PURE entity-marker features."""

from __future__ import annotations

from dataclasses import dataclass

from clinical_re_bilingual.schema import RelationExample, RelationLabel

RELATION_LABELS = tuple(sorted(RelationLabel, key=lambda label: label.value))
label_to_id = {label: index for index, label in enumerate(RELATION_LABELS)}
id_to_label = {index: label for label, index in label_to_id.items()}


@dataclass(frozen=True)
class PureFeature:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    label_id: int
    subject_index: int
    object_index: int


class PureFeatureBuilder:
    def __init__(self, tokenizer, *, max_length: int = 228) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least two")
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def marker_tokens(entity_types: set[str]) -> list[str]:
        markers = ["<SUBJ_START>", "<SUBJ_END>", "<OBJ_START>", "<OBJ_END>"]
        for entity_type in sorted(entity_types):
            markers.extend(
                [
                    f"<SUBJ_START={entity_type}>",
                    f"<SUBJ_END={entity_type}>",
                    f"<OBJ_START={entity_type}>",
                    f"<OBJ_END={entity_type}>",
                    f"<SUBJ={entity_type}>",
                    f"<OBJ={entity_type}>",
                ]
            )
        return markers

    def build(self, example: RelationExample) -> PureFeature:
        offsets = (
            example.subject_start,
            example.subject_end,
            example.object_start,
            example.object_end,
        )
        if any(value is None for value in offsets):
            raise ValueError(f"PURE requires entity offsets for {example.example_id}")
        subject_start, subject_end, object_start, object_end = offsets
        assert subject_start is not None and subject_end is not None
        assert object_start is not None and object_end is not None
        words = example.sentence.split()
        if subject_end >= len(words) or object_end >= len(words):
            raise ValueError(f"entity offsets exceed sentence length for {example.example_id}")

        tokens = [self.tokenizer.cls_token]
        subject_index = object_index = None
        for index, word in enumerate(words):
            if index == subject_start:
                subject_index = len(tokens)
                tokens.append(f"<SUBJ_START={example.subject_type}>")
            if index == object_start:
                object_index = len(tokens)
                tokens.append(f"<OBJ_START={example.object_type}>")
            tokens.append(word)
            if index == subject_end:
                tokens.append(f"<SUBJ_END={example.subject_type}>")
            if index == object_end:
                tokens.append(f"<OBJ_END={example.object_type}>")
        tokens.append(self.tokenizer.sep_token)
        if subject_index is None or object_index is None:
            raise ValueError(f"entity start offset missing for {example.example_id}")
        if len(tokens) > self.max_length:
            raise ValueError(
                f"entity marker would be truncated for {example.example_id}; "
                f"increase max_length={self.max_length}"
            )
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding)
        attention_mask.extend([0] * padding)
        return PureFeature(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=[0] * self.max_length,
            label_id=label_to_id[example.label],
            subject_index=subject_index,
            object_index=object_index,
        )
