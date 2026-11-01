from types import SimpleNamespace

from clinical_re_bilingual.inference.providers import (
    OpenAICompatibleChatClient,
    OpenAIResponsesClient,
)


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text="TrAP")


def test_openai_adapter_disables_server_side_storage() -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    adapter = OpenAIResponsesClient("gpt-4o-mini", client=client)
    assert adapter.generate("protected prompt", temperature=0.0) == "TrAP"
    assert responses.kwargs["store"] is False
    assert responses.kwargs["temperature"] == 0.0


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="TeRP")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_compatible_chat_adapter_uses_one_user_message() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    adapter = OpenAICompatibleChatClient("deepseek-chat", client=client)
    assert adapter.generate("protected prompt", temperature=0.0) == "TeRP"
    assert completions.kwargs["messages"] == [{"role": "user", "content": "protected prompt"}]
