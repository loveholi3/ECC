"""Base class for OpenAI-compatible LLM providers."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from llm.core.interface import (
    AuthenticationError,
    ContextLengthError,
    LLMProvider,
    RateLimitError,
)
from llm.core.types import LLMInput, LLMOutput, ModelInfo, ProviderType, ToolCall
from llm.providers.constants import EMPTY_FILTERED_RESPONSE_ERROR


def parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    """Parse tool arguments from a raw JSON string safely."""
    if not raw_arguments:
        return {}

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw": raw_arguments}

    if isinstance(arguments, dict):
        return arguments
    return {"value": arguments}


class OpenAICompatibleProvider(LLMProvider):
    """Base class for providers that expose an OpenAI-compatible API."""

    client: OpenAI
    default_model: str
    _models: list[ModelInfo]
    provider_type: ProviderType

    def _prepare_params(self, params: dict[str, Any], llm_input: LLMInput) -> None:
        """Hook for subclasses to modify request parameters."""
        pass

    def generate(self, input: LLMInput) -> LLMOutput:
        try:
            params: dict[str, Any] = {
                "model": input.model or self.default_model,
                "messages": [msg.to_dict() for msg in input.messages],
            }
            if input.temperature != 1.0:
                params["temperature"] = input.temperature
            if input.max_tokens is not None:
                params["max_tokens"] = input.max_tokens
            if input.tools:
                params["tools"] = [tool.to_openai_tool() for tool in input.tools]

            # Let subclasses modify params before making the request
            self._prepare_params(params, input)

            response = self.client.chat.completions.create(**params)
            if not response.choices or response.choices[0].message is None:
                raise ValueError(EMPTY_FILTERED_RESPONSE_ERROR)
            choice = response.choices[0]

            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id or "",
                        name=tc.function.name,
                        arguments=parse_tool_arguments(tc.function.arguments),
                    )
                    for tc in choice.message.tool_calls
                ]

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMOutput(
                content=choice.message.content or "",
                tool_calls=tool_calls,
                model=response.model,
                usage=usage,
                stop_reason=choice.finish_reason,
            )
        except Exception as e:
            msg = str(e)
            if "401" in msg or "authentication" in msg.lower():
                raise AuthenticationError(msg, provider=self.provider_type) from e
            if "429" in msg or "rate_limit" in msg.lower():
                raise RateLimitError(msg, provider=self.provider_type) from e
            if "context" in msg.lower() and "length" in msg.lower():
                raise ContextLengthError(msg, provider=self.provider_type) from e
            raise

    def list_models(self) -> list[ModelInfo]:
        return self._models.copy()

    def validate_config(self) -> bool:
        return bool(self.client.api_key)

    def get_default_model(self) -> str:
        return self.default_model
