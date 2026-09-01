"""Astraflow/UModelVerse OpenAI-compatible provider adapters."""

from __future__ import annotations

import os

from openai import OpenAI

from llm.core.types import ModelInfo, ProviderType
from llm.providers.openai_compatible import OpenAICompatibleProvider

ASTRAFLOW_BASE_URL = "https://api.umodelverse.ai/v1"
ASTRAFLOW_CN_BASE_URL = "https://api.modelverse.cn/v1"
DEFAULT_ASTRAFLOW_MODEL = "gpt-4o-mini"


class _AstraflowBaseProvider(OpenAICompatibleProvider):
    provider_type: ProviderType
    api_key_env: str
    base_url_env: str
    model_env: str
    fallback_model_env: str | None = None
    default_base_url: str
    default_model = DEFAULT_ASTRAFLOW_MODEL

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(self.api_key_env) or ""
        self.base_url = base_url or os.environ.get(
            self.base_url_env, self.default_base_url
        )
        env_model = os.environ.get(self.model_env)
        fallback_model = (
            os.environ.get(self.fallback_model_env) if self.fallback_model_env else None
        )
        self.default_model = (
            default_model or env_model or fallback_model or DEFAULT_ASTRAFLOW_MODEL
        )
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, _enforce_credentials=False
        )
        self._models = [
            ModelInfo(
                name=self.default_model,
                provider=self.provider_type,
                supports_tools=True,
                supports_vision=False,
            )
        ]


class AstraflowProvider(_AstraflowBaseProvider):
    """UModelVerse global endpoint using OpenAI-compatible chat completions."""

    provider_type = ProviderType.ASTRAFLOW
    api_key_env = "ASTRAFLOW_API_KEY"
    base_url_env = "ASTRAFLOW_BASE_URL"
    model_env = "ASTRAFLOW_MODEL"
    default_base_url = ASTRAFLOW_BASE_URL


class AstraflowCNProvider(_AstraflowBaseProvider):
    """UModelVerse China endpoint using OpenAI-compatible chat completions."""

    provider_type = ProviderType.ASTRAFLOW_CN
    api_key_env = "ASTRAFLOW_CN_API_KEY"
    base_url_env = "ASTRAFLOW_CN_BASE_URL"
    model_env = "ASTRAFLOW_CN_MODEL"
    fallback_model_env = "ASTRAFLOW_MODEL"
    default_base_url = ASTRAFLOW_CN_BASE_URL
