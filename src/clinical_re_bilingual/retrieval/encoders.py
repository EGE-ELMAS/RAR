"""Text encoders used by retrieval methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from clinical_re_bilingual.schema import RelationExample


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one two-dimensional embedding matrix in input order."""


class RelationEncoder(Protocol):
    def encode(self, examples: Sequence[RelationExample]) -> NDArray[np.float32]:
        """Return entity-aware relation representations in input order."""


class HuggingFaceCLSEncoder:
    """Encode sentences using the first hidden state of a Hugging Face model."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device_name = device
        self.cache_dir = cache_dir
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "KATE encoding requires the torch and transformers dependencies"
            ) from error
        device_name = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self._model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self._model.to(self._device)
        self._model.eval()

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("at least one text is required")
        self._load()
        import torch

        matrices: list[NDArray[np.float32]] = []
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                output = self._model(**encoded)
                cls_embeddings = output.last_hidden_state[:, 0, :]
            matrices.append(cls_embeddings.detach().cpu().numpy().astype(np.float32))
        matrix = np.concatenate(matrices, axis=0)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise RuntimeError("encoder returned an invalid embedding matrix")
        return matrix


class BertEntityMarkerEncoder:
    """Concatenate subject and object start-marker hidden states."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int = 16,
        max_length: int = 228,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_name = device
        self.cache_dir = cache_dir
        self._tokenizer = None
        self._model = None
        self._device = None
        self._registered_entity_types: set[str] = set()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "PURE encoding requires the torch and transformers dependencies"
            ) from error
        device_name = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, cache_dir=self.cache_dir
        )
        self._model = AutoModel.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def marker_tokens(entity_types: set[str]) -> list[str]:
        markers = ["<subj_start>", "<subj_end>", "<obj_start>", "<obj_end>"]
        for entity_type in sorted(entity_types):
            markers.extend(
                [
                    f"<subj_start={entity_type}>",
                    f"<subj_end={entity_type}>",
                    f"<obj_start={entity_type}>",
                    f"<obj_end={entity_type}>",
                    f"<subj={entity_type}>",
                    f"<obj={entity_type}>",
                ]
            )
        return markers

    def _register_markers(self, examples: Sequence[RelationExample]) -> None:
        assert self._tokenizer is not None
        assert self._model is not None
        entity_types = {
            value for example in examples for value in (example.subject_type, example.object_type)
        }
        new_types = entity_types - self._registered_entity_types
        if not new_types:
            return
        all_types = self._registered_entity_types | new_types
        self._tokenizer.add_tokens(self.marker_tokens(all_types))
        self._model.resize_token_embeddings(len(self._tokenizer))
        self._registered_entity_types = all_types

    def _feature(self, example: RelationExample) -> tuple[list[int], list[int], int, int]:
        assert self._tokenizer is not None
        offsets = (
            example.subject_start,
            example.subject_end,
            example.object_start,
            example.object_end,
        )
        if any(value is None for value in offsets):
            raise ValueError(f"PURE requires entity offsets for {example.example_id!r}")
        subject_start, subject_end, object_start, object_end = offsets
        assert subject_start is not None and subject_end is not None
        assert object_start is not None and object_end is not None

        words = example.sentence.split()
        if subject_end >= len(words) or object_end >= len(words):
            raise ValueError(f"entity offsets exceed sentence length for {example.example_id!r}")
        tokens = [self._tokenizer.cls_token]
        subject_index = object_index = None
        for index, word in enumerate(words):
            if index == subject_start:
                subject_index = len(tokens)
                tokens.append(f"<subj_start={example.subject_type}>")
            if index == object_start:
                object_index = len(tokens)
                tokens.append(f"<obj_start={example.object_type}>")
            tokens.append(word)
            if index == subject_end:
                tokens.append(f"<subj_end={example.subject_type}>")
            if index == object_end:
                tokens.append(f"<obj_end={example.object_type}>")
        tokens.append(self._tokenizer.sep_token)
        if subject_index is None or object_index is None:
            raise ValueError(f"entity start offset not found for {example.example_id!r}")
        if len(tokens) > self.max_length:
            raise ValueError(
                f"marked sentence exceeds max_length={self.max_length} for {example.example_id!r}"
            )
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding = self.max_length - len(input_ids)
        input_ids.extend([self._tokenizer.pad_token_id] * padding)
        attention_mask.extend([0] * padding)
        return input_ids, attention_mask, subject_index, object_index

    def encode(self, examples: Sequence[RelationExample]) -> NDArray[np.float32]:
        if not examples:
            raise ValueError("at least one relation example is required")
        self._load()
        self._register_markers(examples)
        import torch

        assert self._model is not None
        assert self._device is not None
        features = [self._feature(example) for example in examples]
        matrices: list[NDArray[np.float32]] = []
        for start in range(0, len(features), self.batch_size):
            batch = features[start : start + self.batch_size]
            input_ids = torch.tensor([item[0] for item in batch], device=self._device)
            attention_mask = torch.tensor([item[1] for item in batch], device=self._device)
            subject_indices = torch.tensor([item[2] for item in batch], device=self._device)
            object_indices = torch.tensor([item[3] for item in batch], device=self._device)
            with torch.inference_mode():
                output = self._model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = output.last_hidden_state
                rows = torch.arange(hidden.shape[0], device=self._device)
                subject_states = hidden[rows, subject_indices]
                object_states = hidden[rows, object_indices]
                relation_states = torch.cat((subject_states, object_states), dim=1)
            matrices.append(relation_states.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(matrices, axis=0)
