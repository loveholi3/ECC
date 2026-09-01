"""Atlas Cloud OpenAI-compatible provider adapter."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from llm.core.types import LLMInput, ModelInfo, ProviderType
from llm.providers.base_openai import OpenAICompatibleProvider

ATLAS_BASE_URL = "https://api.atlascloud.ai/v1"
DEFAULT_ATLAS_MODEL = "deepseek-ai/deepseek-v4-pro"
# Reasoning models need enough headroom for their thinking budget plus the answer.
DEFAULT_ATLAS_MAX_TOKENS = 512


class AtlasProvider(OpenAICompatibleProvider):
    """Atlas Cloud endpoint using OpenAI-compatible chat completions.

    Atlas Cloud (https://atlascloud.ai) exposes 300+ hosted models behind a
    single OpenAI-compatible API, so it reuses the same chat-completions flow as
    the other OpenAI-compatible adapters in this package.
    """

    provider_type = ProviderType.ATLAS
    # ``.env.example`` documents ATLAS_API_KEY; ATLASCLOUD_API_KEY is the name used
    # by the Atlas Cloud SDK/skill, so accept either for convenience.
    api_key_env = "ATLAS_API_KEY"
    fallback_api_key_env = "ATLASCLOUD_API_KEY"
    base_url_env = "ATLAS_BASE_URL"
    model_env = "ATLAS_MODEL"
    default_base_url = ATLAS_BASE_URL

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get(self.api_key_env)
            or os.environ.get(self.fallback_api_key_env)
            or ""
        )
        self.base_url = base_url or os.environ.get(self.base_url_env, self.default_base_url)
        env_model = os.environ.get(self.model_env)
        self.default_model = default_model or env_model or DEFAULT_ATLAS_MODEL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, _enforce_credentials=False)
        self._models = [
            ModelInfo(
                name=self.default_model,
                provider=self.provider_type,
                supports_tools=True,
                supports_vision=False,
            )
        ]

    def _modify_params(self, params: dict[str, Any], llm_input: LLMInput) -> None:
        # Atlas reasoning models spend tokens on a thinking budget before the
        # answer, so floor max_tokens to avoid truncated/empty completions.
        max_tokens = llm_input.max_tokens
        if max_tokens is None or max_tokens < DEFAULT_ATLAS_MAX_TOKENS:
            max_tokens = DEFAULT_ATLAS_MAX_TOKENS
        params["max_tokens"] = max_tokens
