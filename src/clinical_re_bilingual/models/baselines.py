"""Training implementations for BERT and PURE relation classifiers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from clinical_re_bilingual.models.config import BaselineTrainConfig, PureTrainConfig
from clinical_re_bilingual.models.features import (
    RELATION_LABELS,
    PureFeatureBuilder,
    id_to_label,
    label_to_id,
)
from clinical_re_bilingual.schema import RelationExample, RelationLabel


@dataclass(frozen=True)
class BaselineResult:
    predicted_labels: list[RelationLabel]
    best_dev_macro_f1: float


def _device(name: str | None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def _labels(examples: list[RelationExample]) -> torch.Tensor:
    return torch.tensor([label_to_id[example.label] for example in examples], dtype=torch.long)


def _optimizer_and_scheduler(model, loader, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_proportion),
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _macro_f1(gold: list[int], predicted: list[int]) -> float:
    return float(f1_score(gold, predicted, average="macro", zero_division=0))


def train_bert_baseline(
    train: list[RelationExample],
    validation: list[RelationExample],
    test: list[RelationExample],
    *,
    model_name: str,
    config: BaselineTrainConfig,
    output_dir: Path,
    device_name: str | None = None,
) -> BaselineResult:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(RELATION_LABELS),
        label2id={label.value: index for label, index in label_to_id.items()},
        id2label={index: label.value for index, label in id_to_label.items()},
    )
    device = _device(device_name)
    model.to(device)

    def loader(examples: list[RelationExample], *, shuffle: bool) -> DataLoader:
        encoded = tokenizer(
            [example.sentence for example in examples],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], _labels(examples))
        return DataLoader(
            dataset,
            batch_size=config.train_batch_size if shuffle else config.eval_batch_size,
            shuffle=shuffle,
        )

    train_loader = loader(train, shuffle=True)
    validation_loader = loader(validation, shuffle=False)
    test_loader = loader(test, shuffle=False)
    optimizer, scheduler = _optimizer_and_scheduler(model, train_loader, config)

    def evaluate(data_loader: DataLoader) -> tuple[list[int], list[int]]:
        gold: list[int] = []
        predicted: list[int] = []
        model.eval()
        with torch.inference_mode():
            for input_ids, attention_mask, labels in data_loader:
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                )
                gold.extend(labels.tolist())
                predicted.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        return gold, predicted

    best_score = -1.0
    best_state = None
    for _ in range(config.epochs):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
        gold, predicted = evaluate(validation_loader)
        score = _macro_f1(gold, predicted)
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _, predicted = evaluate(test_loader)
    return BaselineResult([id_to_label[index] for index in predicted], best_score)


class PureRelationClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size * 2, num_labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        subject_indices,
        object_indices,
    ):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.encoder.config.model_type == "bert":
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        representation = torch.cat(
            (hidden[rows, subject_indices], hidden[rows, object_indices]), dim=1
        )
        return self.classifier(representation)


def train_pure_baseline(
    train: list[RelationExample],
    validation: list[RelationExample],
    test: list[RelationExample],
    *,
    model_name: str,
    config: PureTrainConfig,
    output_dir: Path,
    device_name: str | None = None,
) -> BaselineResult:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    entity_types = {
        value
        for example in train + validation + test
        for value in (example.subject_type, example.object_type)
    }
    tokenizer.add_special_tokens(
        {"additional_special_tokens": PureFeatureBuilder.marker_tokens(entity_types)}
    )
    model = PureRelationClassifier(model_name, len(RELATION_LABELS))
    model.encoder.resize_token_embeddings(len(tokenizer))
    device = _device(device_name)
    model.to(device)
    feature_builder = PureFeatureBuilder(tokenizer, max_length=config.max_length)

    def loader(examples: list[RelationExample], *, shuffle: bool) -> DataLoader:
        features = [feature_builder.build(example) for example in examples]
        dataset = TensorDataset(
            torch.tensor([item.input_ids for item in features]),
            torch.tensor([item.attention_mask for item in features]),
            torch.tensor([item.token_type_ids for item in features]),
            torch.tensor([item.label_id for item in features]),
            torch.tensor([item.subject_index for item in features]),
            torch.tensor([item.object_index for item in features]),
        )
        return DataLoader(
            dataset,
            batch_size=config.train_batch_size if shuffle else config.eval_batch_size,
            shuffle=shuffle,
        )

    train_loader = loader(train, shuffle=True)
    validation_loader = loader(validation, shuffle=False)
    test_loader = loader(test, shuffle=False)
    no_decay = ("bias", "LayerNorm.weight")
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(value in name for value in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(value in name for value in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_proportion),
        num_training_steps=total_steps,
    )
    loss_function = nn.CrossEntropyLoss()

    def evaluate(data_loader: DataLoader) -> tuple[list[int], list[int]]:
        gold: list[int] = []
        predicted: list[int] = []
        model.eval()
        with torch.inference_mode():
            for batch in data_loader:
                input_ids, attention_mask, token_types, labels, subject, object_ = batch
                logits = model(
                    input_ids.to(device),
                    attention_mask.to(device),
                    token_types.to(device),
                    subject.to(device),
                    object_.to(device),
                )
                gold.extend(labels.tolist())
                predicted.extend(logits.argmax(dim=-1).cpu().tolist())
        return gold, predicted

    best_score = -1.0
    best_state = None
    for _ in range(config.epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, token_types, labels, subject, object_ = batch
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids.to(device),
                attention_mask.to(device),
                token_types.to(device),
                subject.to(device),
                object_.to(device),
            )
            loss_function(logits, labels.to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
        gold, predicted = evaluate(validation_loader)
        score = _macro_f1(gold, predicted)
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir / "encoder")
    torch.save(model.classifier.state_dict(), output_dir / "classifier.pt")
    _, predicted = evaluate(test_loader)
    return BaselineResult([id_to_label[index] for index in predicted], best_score)
