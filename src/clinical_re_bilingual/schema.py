"""Validated records shared across preprocessing, retrieval, and evaluation."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Language(str, Enum):
    ENGLISH = "en"
    TURKISH = "tr"


class RelationLabel(str, Enum):
    TrIP = "TrIP"
    TrWP = "TrWP"
    TrCP = "TrCP"
    TrAP = "TrAP"
    TrNAP = "TrNAP"
    PIP = "PIP"
    TeRP = "TeRP"
    TeCP = "TeCP"


class RelationExample(BaseModel):
    """A protected text-bearing record; never serialize it to public results."""

    model_config = ConfigDict(extra="forbid")

    example_id: str = Field(min_length=1)
    language: Language
    sentence: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    subject_type: str = Field(min_length=1)
    object: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    label: RelationLabel
    subject_start: int | None = Field(default=None, ge=0)
    subject_end: int | None = Field(default=None, ge=0)
    object_start: int | None = Field(default=None, ge=0)
    object_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_entity_spans(self) -> RelationExample:
        spans = (self.subject_start, self.subject_end, self.object_start, self.object_end)
        if all(value is None for value in spans):
            return self
        if any(value is None for value in spans):
            raise ValueError("entity token offsets must be specified together")
        assert self.subject_start is not None and self.subject_end is not None
        assert self.object_start is not None and self.object_end is not None
        if self.subject_start > self.subject_end:
            raise ValueError("subject_start must not exceed subject_end")
        if self.object_start > self.object_end:
            raise ValueError("object_start must not exceed object_end")
        return self


class RetrievalRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_example_id: str = Field(min_length=1)
    retriever: str = Field(min_length=1)
    language: Language
    ranked_train_ids: list[str]
    scores: list[float] | None = None

    @model_validator(mode="after")
    def validate_ranking(self) -> RetrievalRanking:
        if len(self.ranked_train_ids) != len(set(self.ranked_train_ids)):
            raise ValueError("ranked_train_ids must not contain duplicates")
        if self.test_example_id in self.ranked_train_ids:
            raise ValueError("test example leakage detected in retrieval ranking")
        if self.scores is not None:
            if len(self.scores) != len(self.ranked_train_ids):
                raise ValueError("scores and ranked_train_ids must have equal lengths")
            if any(a < b for a, b in zip(self.scores, self.scores[1:], strict=False)):
                raise ValueError("scores must be in descending order")
        return self

    def top_k(self, k: int) -> list[str]:
        if k < 1:
            raise ValueError("k must be at least 1")
        if k > len(self.ranked_train_ids):
            available = len(self.ranked_train_ids)
            raise ValueError(f"requested {k} examples but ranking has only {available}")
        return self.ranked_train_ids[:k]


class PredictionRecord(BaseModel):
    """Public-safe prediction record containing no clinical text."""

    model_config = ConfigDict(extra="forbid")

    example_id: str = Field(min_length=1)
    gold_label: RelationLabel
    predicted_label: RelationLabel
    model: str = Field(min_length=1)
    language: Language
    retriever: str = Field(min_length=1)
    num_shots: int = Field(ge=0)
    prompt_strategy: str = Field(min_length=1)
    temperature: float
    run_date: date
