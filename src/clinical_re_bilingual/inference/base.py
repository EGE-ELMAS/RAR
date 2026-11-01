"""Minimal interface shared by hosted language-model providers."""

from typing import Protocol


class TextGenerationClient(Protocol):
    model_name: str

    def generate(self, prompt: str, *, temperature: float) -> str:
        """Return generated text without persisting the protected prompt."""
