import pytest
from llm.core.types import LLMInput, Message, Role
from llm.providers.openai import OpenAIProvider
from llm.providers.claude import ClaudeProvider
from llm.providers.astraflow import AstraflowProvider
from llm.providers.atlas import AtlasProvider
import asyncio

class MockProviderMixin:
    def generate(self, input):
        from llm.core.types import LLMOutput
        return LLMOutput(content="async mock", model="mock-model", tool_calls=None, stop_reason="end")

@pytest.mark.asyncio
async def test_agenerate():
    class MockOpenAI(MockProviderMixin, OpenAIProvider):
        def __init__(self):
            pass
    class MockClaude(MockProviderMixin, ClaudeProvider):
        def __init__(self):
            pass
    class MockAstra(MockProviderMixin, AstraflowProvider):
        def __init__(self):
            pass
    class MockAtlas(MockProviderMixin, AtlasProvider):
        def __init__(self):
            pass

    providers = [MockOpenAI(), MockClaude(), MockAstra(), MockAtlas()]

    input_data = LLMInput(messages=[Message(role=Role.USER, content="hello")])
    for provider in providers:
        result = await provider.agenerate(input_data)
        assert result.content == "async mock"
