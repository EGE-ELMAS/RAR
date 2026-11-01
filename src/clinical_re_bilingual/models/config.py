"""Validated supervised-baseline configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from clinical_re_bilingual.schema import Language


class BaselineTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(ge=1)
    train_batch_size: int = Field(ge=1)
    eval_batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    max_length: int = Field(ge=2)
    warmup_proportion: float = Field(ge=0, lt=1)
    max_grad_norm: float = Field(gt=0)
    checkpoint_metric: str = Field(pattern=r"^macro_f1$")


class PureTrainConfig(BaselineTrainConfig):
    weight_decay: float = Field(ge=0)


class BaselinesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    bert: BaselineTrainConfig
    pure: PureTrainConfig


class LanguageCheckpoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    en: str = Field(min_length=1)
    tr: str = Field(min_length=1)

    def for_language(self, language: Language) -> str:
        return self.en if language is Language.ENGLISH else self.tr


class BaselineCheckpoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bert: LanguageCheckpoints
    pure: LanguageCheckpoints


class ModelsFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    baselines: BaselineCheckpoints


def _yaml(path: Path) -> object:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        raise ValueError(f"configuration file is empty: {path}")
    return value


def read_baselines_config(path: Path) -> BaselinesConfig:
    return BaselinesConfig.model_validate(_yaml(path))


def read_baseline_checkpoints(path: Path) -> BaselineCheckpoints:
    return ModelsFile.model_validate(_yaml(path)).baselines
