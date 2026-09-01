"""OpenAI provider adapter."""

from __future__ import annotations

import os

from openai import OpenAI

from llm.core.types import ModelInfo, ProviderType
from llm.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    provider_type = ProviderType.OPENAI

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            _enforce_credentials=False,
        )
        self.default_model = "gpt-4o-mini"
        self._models = [
            ModelInfo(
                name="gpt-4o",
                provider=ProviderType.OPENAI,
                supports_tools=True,
                supports_vision=True,
                max_tokens=4096,
                context_window=128000,
            ),
            ModelInfo(
                name="gpt-4o-mini",
                provider=ProviderType.OPENAI,
                supports_tools=True,
                supports_vision=True,
                max_tokens=4096,
                context_window=128000,
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider=ProviderType.OPENAI,
                supports_tools=True,
                supports_vision=True,
                max_tokens=4096,
                context_window=128000,
            ),
            ModelInfo(
                name="gpt-3.5-turbo",
                provider=ProviderType.OPENAI,
                supports_tools=True,
                supports_vision=False,
                max_tokens=4096,
                context_window=16385,
            ),
        ]
