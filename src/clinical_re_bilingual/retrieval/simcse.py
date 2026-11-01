"""Supervised SimCSE training for the relation-aware retriever."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np

from clinical_re_bilingual.retrieval.rar import ContrastiveTriple, relation_knn_micro_f1
from clinical_re_bilingual.schema import RelationExample


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("SimCSE training requires torch") from error
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimCSETrainer:
    """Fine-tune a Hugging Face encoder with supervised contrastive triples."""

    def __init__(
        self,
        model_name: str,
        *,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        temperature: float = 0.05,
        max_length: int = 256,
        seed: int = 42,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if epochs < 1 or batch_size < 1 or max_length < 2:
            raise ValueError("epochs, batch_size, and max_length must be positive")
        if learning_rate <= 0 or temperature <= 0:
            raise ValueError("learning_rate and temperature must be positive")
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.max_length = max_length
        self.seed = seed
        self.device_name = device
        self.cache_dir = cache_dir

    def train(
        self,
        triples: Sequence[ContrastiveTriple],
        output_dir: str | Path,
        *,
        validation_train_examples: Sequence[RelationExample] | None = None,
        validation_examples: Sequence[RelationExample] | None = None,
    ) -> list[float]:
        """Train and select a checkpoint using relation-label retrieval quality."""

        if not triples:
            raise ValueError("at least one contrastive triple is required")
        if (validation_train_examples is None) != (validation_examples is None):
            raise ValueError("both validation example collections must be provided together")
        try:
            import torch
            import torch.nn.functional as functional
            from torch.utils.data import DataLoader
            from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
        except ImportError as error:
            raise RuntimeError("SimCSE training requires torch and transformers") from error

        set_seed(self.seed)
        device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(device)
        projection = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, model.config.hidden_size),
            torch.nn.Tanh(),
        ).to(device)
        parameters = list(model.parameters()) + list(projection.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)

        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            list(triples),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
            collate_fn=lambda items: items,
        )
        total_steps = len(loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        def embeddings(texts: list[str]):
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            return projection(model(**encoded).last_hidden_state[:, 0, :])

        class EvaluationEncoder:
            def encode(self, texts: Sequence[str]) -> np.ndarray:
                rows: list[np.ndarray] = []
                model.eval()
                with torch.inference_mode():
                    for start in range(0, len(texts), self_batch_size):
                        batch = list(texts[start : start + self_batch_size])
                        encoded = tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=self_max_length,
                            return_tensors="pt",
                        )
                        encoded = {key: value.to(device) for key, value in encoded.items()}
                        cls = model(**encoded).last_hidden_state[:, 0, :]
                        rows.append(cls.detach().cpu().numpy().astype(np.float32))
                return np.concatenate(rows, axis=0)

        self_batch_size = self.batch_size
        self_max_length = self.max_length
        evaluation_encoder = EvaluationEncoder()

        epoch_losses: list[float] = []
        validation_scores: list[float] = []
        best_score = -1.0
        best_epoch: int | None = None
        best_state = None
        model.train()
        projection.train()
        for _ in range(self.epochs):
            running_loss = 0.0
            for batch in loader:
                anchor = embeddings([item.anchor for item in batch])
                positive = embeddings([item.positive for item in batch])
                negative = embeddings([item.negative for item in batch])
                positive_logits = functional.cosine_similarity(
                    anchor.unsqueeze(1), positive.unsqueeze(0), dim=-1
                )
                negative_logits = functional.cosine_similarity(
                    anchor.unsqueeze(1), negative.unsqueeze(0), dim=-1
                )
                logits = torch.cat((positive_logits, negative_logits), dim=1)
                logits = logits / self.temperature
                labels = torch.arange(len(batch), device=device)
                loss = functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.detach().cpu())
            epoch_losses.append(running_loss / len(loader))
            if validation_train_examples is not None and validation_examples is not None:
                score = relation_knn_micro_f1(
                    validation_train_examples, validation_examples, evaluation_encoder
                )
                validation_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_epoch = len(epoch_losses)
                    best_state = deepcopy(model.state_dict())
                model.train()
                projection.train()

        if best_state is not None:
            model.load_state_dict(best_state)

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        metadata = {
            "base_model": self.model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "max_length": self.max_length,
            "seed": self.seed,
            "epoch_losses": epoch_losses,
            "validation_metric": "knn_micro_f1",
            "validation_scores": validation_scores,
            "best_epoch": best_epoch,
            "best_validation_score": best_score if best_epoch is not None else None,
        }
        (destination / "rar_training.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return epoch_losses
