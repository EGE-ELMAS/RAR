"""Validated configuration loading for reproducible experiments."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from clinical_re_bilingual.schema import Language


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    temperature: float = Field(gt=0)
    max_length: int = Field(ge=2)
    checkpoint_metric: str = Field(pattern=r"^knn_micro_f1$")


class PairGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_neighbors: int = Field(ge=1)
    negative_neighbors: int = Field(ge=1)
    sentence_weight: float = Field(ge=0)
    entity_weight: float = Field(ge=0)
    relation_weight: float = Field(ge=0)
    subject_weight: float = Field(ge=0)
    object_weight: float = Field(ge=0)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_weight: float = Field(ge=0)
    entity_weight: float = Field(ge=0)
    subject_weight: float = Field(ge=0)
    object_weight: float = Field(ge=0)
    candidate_pool_size: int = Field(default=20, ge=1)


class RARConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    train: TrainConfig
    pair_generation: PairGenerationConfig
    retrieval: RetrievalConfig


class LanguageModels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    en: str = Field(min_length=1)
    tr: str = Field(min_length=1)

    def for_language(self, language: Language) -> str:
        return self.en if language is Language.ENGLISH else self.tr


class RetrieverModels(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rar: LanguageModels


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    retrievers: RetrieverModels


def _read_yaml(path: Path) -> object:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if content is None:
        raise ValueError(f"configuration file is empty: {path}")
    return content


def read_rar_config(path: Path) -> RARConfig:
    return RARConfig.model_validate(_read_yaml(path))


def read_models_config(path: Path) -> ModelsConfig:
    return ModelsConfig.model_validate(_read_yaml(path))
