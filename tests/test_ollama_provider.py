import os
import json
from unittest.mock import patch, MagicMock

import pytest

from llm.core.interface import AuthenticationError, ContextLengthError, RateLimitError
from llm.core.types import LLMInput, Message, ProviderType, Role, ToolCall
from llm.providers.ollama import OllamaProvider



def test_ollama_provider_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    provider = OllamaProvider()

    assert provider.provider_type == ProviderType.OLLAMA
    assert provider.base_url == "http://localhost:11434"
    assert provider.get_default_model() == "llama3.2"
    assert provider.validate_config() is True


def test_ollama_provider_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:72b")

    provider = OllamaProvider()

    assert provider.base_url == "http://ollama:11434"
    assert provider.get_default_model() == "qwen2.5:72b"
    assert provider.validate_config() is True


def test_ollama_provider_validate_config(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    # Even if we explicitly clear it, it defaults to localhost. We can test if we instantiate with None
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    provider = OllamaProvider(base_url="")
    assert provider.validate_config() is False


def test_ollama_provider_list_models():
    provider = OllamaProvider()
    models = provider.list_models()

    assert len(models) == 3
    assert models[0].name == "llama3.2"
    assert models[1].name == "mistral"
    assert models[2].name == "codellama"

class MockResponse:
    def __init__(self, json_data):
        self.json_data = json_data

    def read(self):
        return json.dumps(self.json_data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@patch("urllib.request.urlopen")
def test_generate_success_with_text(mock_urlopen):
    mock_response_data = {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "Hello there!"
        },
        "done_reason": "stop"
    }
    mock_urlopen.return_value = MockResponse(mock_response_data)

    provider = OllamaProvider()
    input_data = LLMInput(
        messages=[Message(role=Role.USER, content="Hi")]
    )

    output = provider.generate(input_data)

    assert output.content == "Hello there!"
    assert output.model == "llama3.2"
    assert output.stop_reason == "stop"
    assert output.tool_calls is None

    # Verify request payload
    req = mock_urlopen.call_args[0][0]
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["model"] == "llama3.2"
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == "Hi"
    assert payload["stream"] is False
    assert "options" not in payload


@patch("urllib.request.urlopen")
def test_generate_with_tool_calls(mock_urlopen):
    mock_response_data = {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "London"}
                    }
                }
            ]
        },
        "done_reason": "tool_calls"
    }
    mock_urlopen.return_value = MockResponse(mock_response_data)

    provider = OllamaProvider()
    input_data = LLMInput(
        messages=[Message(role=Role.USER, content="What is the weather in London?")]
    )

    output = provider.generate(input_data)

    assert output.content == ""
    assert output.stop_reason == "tool_calls"
    assert output.tool_calls is not None
    assert len(output.tool_calls) == 1

    tool_call = output.tool_calls[0]
    assert tool_call.id == "call_abc123"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"location": "London"}


@patch("urllib.request.urlopen")
def test_generate_with_temperature(mock_urlopen):
    mock_response_data = {
        "model": "llama3.2",
        "message": {
            "content": "Warm hello"
        }
    }
    mock_urlopen.return_value = MockResponse(mock_response_data)

    provider = OllamaProvider()
    input_data = LLMInput(
        messages=[Message(role=Role.USER, content="Hi")],
        temperature=0.7
    )

    provider.generate(input_data)

    # Verify request payload includes temperature
    req = mock_urlopen.call_args[0][0]
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["options"]["temperature"] == 0.7

@patch("urllib.request.urlopen")
def test_generate_handles_connection_error(mock_urlopen):
    mock_urlopen.side_effect = Exception("Connection refused")

    provider = OllamaProvider()
    input_data = LLMInput(messages=[Message(role=Role.USER, content="Hi")])

    with pytest.raises(AuthenticationError) as exc_info:
        provider.generate(input_data)

    assert "Ollama connection failed" in str(exc_info.value)
    assert exc_info.value.provider == ProviderType.OLLAMA


@patch("urllib.request.urlopen")
def test_generate_handles_401_error(mock_urlopen):
    mock_urlopen.side_effect = Exception("HTTP Error 401: Unauthorized")

    provider = OllamaProvider()
    input_data = LLMInput(messages=[Message(role=Role.USER, content="Hi")])

    with pytest.raises(AuthenticationError) as exc_info:
        provider.generate(input_data)

    assert "Ollama connection failed" in str(exc_info.value)
    assert exc_info.value.provider == ProviderType.OLLAMA


@patch("urllib.request.urlopen")
def test_generate_handles_rate_limit(mock_urlopen):
    mock_urlopen.side_effect = Exception("HTTP Error 429: Too Many Requests")

    provider = OllamaProvider()
    input_data = LLMInput(messages=[Message(role=Role.USER, content="Hi")])

    with pytest.raises(RateLimitError) as exc_info:
        provider.generate(input_data)

    assert exc_info.value.provider == ProviderType.OLLAMA


@patch("urllib.request.urlopen")
def test_generate_handles_context_length(mock_urlopen):
    mock_urlopen.side_effect = Exception("Context length exceeded")

    provider = OllamaProvider()
    input_data = LLMInput(messages=[Message(role=Role.USER, content="Hi")])

    with pytest.raises(ContextLengthError) as exc_info:
        provider.generate(input_data)

    assert exc_info.value.provider == ProviderType.OLLAMA


@patch("urllib.request.urlopen")
def test_generate_handles_generic_exception(mock_urlopen):
    mock_urlopen.side_effect = Exception("Some other error")

    provider = OllamaProvider()
    input_data = LLMInput(messages=[Message(role=Role.USER, content="Hi")])

    with pytest.raises(Exception, match="Some other error"):
        provider.generate(input_data)
