import json
import urllib.request
from typing import Any

import pytest

from llm.core.interface import AuthenticationError, ContextLengthError, RateLimitError
from llm.core.types import LLMInput, Message, ProviderType, Role
from llm.providers.ollama import OllamaProvider


class MockResponse:
    def __init__(self, data: dict[str, Any]):
        self.data = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read(self):
        return self.data


def test_ollama_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    provider = OllamaProvider()

    assert provider.provider_type == ProviderType.OLLAMA
    assert provider.base_url == "http://localhost:11434"
    assert provider.get_default_model() == "llama3.2"
    assert provider.validate_config() is True


def test_ollama_provider_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")

    provider = OllamaProvider()

    assert provider.base_url == "http://ollama:11434"
    assert provider.get_default_model() == "mistral"


def test_ollama_provider_generate_text(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()

    def mock_urlopen(request, timeout=None):
        assert request.full_url == "http://localhost:11434/api/chat"
        req_body = json.loads(request.data.decode("utf-8"))
        assert req_body["model"] == "llama3.2"
        assert req_body["messages"] == [{"role": "user", "content": "hello"}]

        return MockResponse({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "hi there"},
            "done_reason": "stop"
        })

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    output = provider.generate(
        LLMInput(messages=[Message(role=Role.USER, content="hello")])
    )

    assert output.content == "hi there"
    assert output.model == "llama3.2"
    assert output.tool_calls is None
    assert output.stop_reason == "stop"


def test_ollama_provider_generate_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()

    def mock_urlopen(request, timeout=None):
        return MockResponse({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "San Francisco"}
                        }
                    }
                ]
            },
            "done_reason": "tool_calls"
        })

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    output = provider.generate(
        LLMInput(messages=[Message(role=Role.USER, content="weather in SF?")])
    )

    assert output.content == ""
    assert output.tool_calls is not None
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].id == "call_1"
    assert output.tool_calls[0].name == "get_weather"
    assert output.tool_calls[0].arguments == {"location": "San Francisco"}
    assert output.stop_reason == "tool_calls"


def test_ollama_provider_generate_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()

    def mock_urlopen(request, timeout=None):
        req_body = json.loads(request.data.decode("utf-8"))
        assert req_body["options"]["temperature"] == 0.5
        assert req_body["model"] == "custom-model"

        return MockResponse({
            "model": "custom-model",
            "message": {"role": "assistant", "content": "hi"},
        })

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    output = provider.generate(
        LLMInput(
            messages=[Message(role=Role.USER, content="hi")],
            model="custom-model",
            temperature=0.5
        )
    )

    assert output.content == "hi"


def test_ollama_provider_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()

    def mock_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(AuthenticationError) as exc:
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="hi")]))

    assert "connection failed" in str(exc.value).lower()
    assert exc.value.provider == ProviderType.OLLAMA


def test_ollama_provider_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()

    def mock_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(AuthenticationError) as exc:
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="hi")]))

    assert "401" in str(exc.value)


def test_ollama_provider_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()

    def mock_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(RateLimitError) as exc:
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="hi")]))

    assert "429" in str(exc.value)


def test_ollama_provider_context_length_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()

    def mock_urlopen(request, timeout=None):
        raise Exception("Context length exceeded")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(ContextLengthError) as exc:
        provider.generate(LLMInput(messages=[Message(role=Role.USER, content="hi")]))

    assert "context" in str(exc.value).lower()


def test_ollama_provider_list_models() -> None:
    provider = OllamaProvider()
    models = provider.list_models()

    assert len(models) == 3
    assert models[0].name == "llama3.2"
    assert models[1].name == "mistral"
    assert models[2].name == "codellama"

    # Check that they use Ollama provider
    for model in models:
        assert model.provider == ProviderType.OLLAMA


def test_ollama_provider_validate_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    provider = OllamaProvider(base_url="")

    # Should be False if base_url is empty
    assert provider.validate_config() is False

    provider = OllamaProvider(base_url="http://localhost:11434")
    assert provider.validate_config() is True
