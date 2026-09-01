import json
from types import SimpleNamespace
from typing import Any

import pytest

from llm.core.interface import (
    AuthenticationError,
    ContextLengthError,
    RateLimitError,
)
from llm.core.types import LLMInput, Message, Role, ToolDefinition
from llm.providers.constants import EMPTY_FILTERED_RESPONSE_ERROR
from llm.providers.openai import OpenAIProvider

class FakeCompletions:
    def __init__(self, response_or_error: Any) -> None:
        self.response_or_error = response_or_error

    def create(self, **_params: object) -> Any:
        if isinstance(self.response_or_error, Exception):
            raise self.response_or_error
        return self.response_or_error

class FakeChat:
    def __init__(self, response_or_error: Any) -> None:
        self.completions = FakeCompletions(response_or_error)

class FakeClient:
    def __init__(self, response_or_error: Any) -> None:
        self.chat = FakeChat(response_or_error)
        self.api_key = "test-key"

def make_provider(response_or_error: Any) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test-key")
    provider.client = FakeClient(response_or_error)
    return provider

def make_response(
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=usage,
    )

def make_empty_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[],
        model="gpt-4o-mini",
        usage=None,
    )

@pytest.mark.unit
def test_generate_text_only() -> None:
    provider = make_provider(make_response(content="Hello there!"))
    output = provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

    assert output.content == "Hello there!"
    assert output.tool_calls is None
    assert output.stop_reason == "stop"
    assert output.model == "gpt-4o-mini"
    assert output.usage is not None
    assert output.usage["prompt_tokens"] == 10
    assert output.usage["completion_tokens"] == 20
    assert output.usage["total_tokens"] == 30

@pytest.mark.unit
def test_generate_with_tools() -> None:
    provider = make_provider(
        make_response(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(
                        name="get_weather",
                        arguments='{"location": "London"}'
                    )
                )
            ],
            finish_reason="tool_calls"
        )
    )

    tool = ToolDefinition(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}}
    )

    output = provider.generate(LLMInput(
        messages=[Message(role=Role.USER, content="Weather?")],
        tools=[tool]
    ))

    assert output.content == ""
    assert output.tool_calls is not None
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].id == "call_1"
    assert output.tool_calls[0].name == "get_weather"
    assert output.tool_calls[0].arguments == {"location": "London"}

@pytest.mark.unit
def test_generate_empty_response_error() -> None:
    provider = make_provider(make_empty_response())
    with pytest.raises(ValueError, match=EMPTY_FILTERED_RESPONSE_ERROR):
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

@pytest.mark.unit
def test_generate_authentication_error() -> None:
    provider = make_provider(Exception("Error 401: Authentication failed"))
    with pytest.raises(AuthenticationError):
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

@pytest.mark.unit
def test_generate_rate_limit_error() -> None:
    provider = make_provider(Exception("Error 429: Rate limit exceeded"))
    with pytest.raises(RateLimitError):
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

@pytest.mark.unit
def test_generate_context_length_error() -> None:
    provider = make_provider(Exception("Error context length exceeded"))
    with pytest.raises(ContextLengthError):
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

@pytest.mark.unit
def test_generate_generic_error() -> None:
    provider = make_provider(Exception("Unknown error occurred"))
    with pytest.raises(Exception, match="Unknown error occurred"):
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="Hi")]))

@pytest.mark.unit
def test_list_models() -> None:
    provider = OpenAIProvider(api_key="test")
    models = provider.list_models()
    assert len(models) == 4
    model_names = [m.name for m in models]
    assert "gpt-4o" in model_names
    assert "gpt-4o-mini" in model_names
    assert "gpt-4-turbo" in model_names
    assert "gpt-3.5-turbo" in model_names

@pytest.mark.unit
def test_validate_config() -> None:
    provider = OpenAIProvider(api_key="test-key")
    assert provider.validate_config() is True

    provider_no_key = OpenAIProvider(api_key="")
    assert provider_no_key.validate_config() is False

@pytest.mark.unit
def test_get_default_model() -> None:
    provider = OpenAIProvider(api_key="test-key")
    assert provider.get_default_model() == "gpt-4o-mini"
