import json
from types import SimpleNamespace
from typing import Any

import pytest

from llm.core.interface import (
    AuthenticationError,
    ContextLengthError,
    RateLimitError,
)
from llm.core.types import LLMInput, Message, Role
from llm.providers.openai import OpenAIProvider


class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id: str, function: FakeFunction) -> None:
        self.id = id
        self.function = function


class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[FakeToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage | None, finish_reason: str = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class FakeResponse:
    def __init__(
        self,
        choices: list[FakeChoice],
        usage: FakeUsage | None = None,
        model: str = "gpt-4o-mini"
    ) -> None:
        self.choices = choices
        self.usage = usage
        self.model = model


class FakeCompletions:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self._response = response

    def create(self, **kwargs: Any) -> FakeResponse:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeChat:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.completions = FakeCompletions(response)


class FakeClient:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.chat = FakeChat(response)
        self.api_key = "test-key"


def make_provider(response: FakeResponse | Exception) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test-key")
    provider.client = FakeClient(response)
    return provider


def make_response(
    content: str | None = None,
    tool_calls: list[FakeToolCall] | None = None,
    finish_reason: str = "stop",
    empty_choices: bool = False,
) -> FakeResponse:
    if empty_choices:
        choices = []
    else:
        choices = [FakeChoice(message=FakeMessage(content=content, tool_calls=tool_calls), finish_reason=finish_reason)]

    return FakeResponse(
        choices=choices,
        usage=FakeUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        model="gpt-4o-test",
    )


@pytest.mark.unit
def test_list_models() -> None:
    provider = make_provider(make_response())
    models = provider.list_models()

    assert len(models) == 4
    model_names = [m.name for m in models]
    assert "gpt-4o" in model_names
    assert "gpt-4o-mini" in model_names
    assert "gpt-4-turbo" in model_names
    assert "gpt-3.5-turbo" in model_names


@pytest.mark.unit
def test_get_default_model() -> None:
    provider = make_provider(make_response())
    assert provider.get_default_model() == "gpt-4o-mini"


@pytest.mark.unit
def test_validate_config() -> None:
    provider = make_provider(make_response())
    assert provider.validate_config() is True

    provider.client.api_key = ""
    assert provider.validate_config() is False

    provider.client.api_key = None
    assert provider.validate_config() is False


@pytest.mark.unit
def test_generate_text_only() -> None:
    provider = make_provider(make_response(content="Hello from OpenAI!"))
    output = provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

    assert output.content == "Hello from OpenAI!"
    assert output.tool_calls is None
    assert output.model == "gpt-4o-test"
    assert output.usage is not None
    assert output.usage["prompt_tokens"] == 10
    assert output.usage["completion_tokens"] == 20
    assert output.usage["total_tokens"] == 30
    assert output.stop_reason == "stop"


@pytest.mark.unit
def test_generate_with_tools() -> None:
    provider = make_provider(
        make_response(
            tool_calls=[
                FakeToolCall(id="call_1", function=FakeFunction(name="search", arguments='{"query": "openai"}'))
            ],
            finish_reason="tool_calls",
        )
    )

    output = provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Use tool")]))

    assert output.content == ""
    assert output.tool_calls is not None
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].id == "call_1"
    assert output.tool_calls[0].name == "search"
    assert output.tool_calls[0].arguments == {"query": "openai"}
    assert output.stop_reason == "tool_calls"


@pytest.mark.unit
def test_generate_multiple_tool_calls() -> None:
    provider = make_provider(
        make_response(
            tool_calls=[
                FakeToolCall(id="call_1", function=FakeFunction(name="search", arguments='{"query": "openai"}')),
                FakeToolCall(id="call_2", function=FakeFunction(name="read", arguments='{"path": "file.txt"}')),
            ],
            finish_reason="tool_calls",
        )
    )

    output = provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Use tools")]))

    assert output.content == ""
    assert output.tool_calls is not None
    assert len(output.tool_calls) == 2
    assert output.tool_calls[0].id == "call_1"
    assert output.tool_calls[0].name == "search"
    assert output.tool_calls[0].arguments == {"query": "openai"}
    assert output.tool_calls[1].id == "call_2"
    assert output.tool_calls[1].name == "read"
    assert output.tool_calls[1].arguments == {"path": "file.txt"}
    assert output.stop_reason == "tool_calls"


@pytest.mark.unit
def test_generate_handles_authentication_error() -> None:
    provider = make_provider(Exception("Error 401: Authentication failed"))

    with pytest.raises(AuthenticationError) as exc_info:
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

    assert "401" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


@pytest.mark.unit
def test_generate_handles_rate_limit_error() -> None:
    provider = make_provider(Exception("Error 429: Rate limit exceeded"))

    with pytest.raises(RateLimitError) as exc_info:
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

    assert "429" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


@pytest.mark.unit
def test_generate_handles_context_length_error() -> None:
    provider = make_provider(Exception("Context length exceeded"))

    with pytest.raises(ContextLengthError) as exc_info:
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

    assert "Context length" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


@pytest.mark.unit
def test_generate_empty_response() -> None:
    provider = make_provider(make_response(empty_choices=True))

    with pytest.raises(ValueError) as exc_info:
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

    assert "LLM returned empty or filtered response" in str(exc_info.value)
