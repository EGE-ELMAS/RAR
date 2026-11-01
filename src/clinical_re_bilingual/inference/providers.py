"""Thin hosted-model adapters with lazy optional dependencies."""

from __future__ import annotations

from typing import Any


class OpenAIResponsesClient:
    def __init__(self, model_name: str, *, client: Any | None = None) -> None:
        self.model_name = model_name
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError("OpenAI inference requires the openai extra") from error
            client = OpenAI()
        self._client = client

    def generate(self, prompt: str, *, temperature: float) -> str:
        response = self._client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
            store=False,
        )
        text = response.output_text
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("OpenAI returned an empty text response")
        return text


class OpenAICompatibleChatClient:
    """Chat-completions adapter for DeepSeek and compatible endpoints."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError("compatible chat inference requires the openai extra") from error
            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client

    def generate(self, prompt: str, *, temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
        )
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("chat provider returned an empty text response")
        return text


class GeminiClient:
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        if client is None:
            try:
                from google import genai
            except ImportError as error:
                raise RuntimeError("Gemini inference requires the gemini extra") from error
            client = genai.Client(api_key=api_key)
        self._client = client

    def generate(self, prompt: str, *, temperature: float) -> str:
        try:
            from google.genai import types
        except ImportError as error:
            raise RuntimeError("Gemini inference requires the gemini extra") from error
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        text = response.text
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini returned an empty text response")
        return text
