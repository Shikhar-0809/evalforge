"""Routes model calls to provider-specific clients based on configuration."""

from __future__ import annotations

import logging

import httpx

from evalforge.llm.anthropic import AnthropicProvider
from evalforge.llm.errors import LLMError
from evalforge.llm.gemini import GeminiProvider
from evalforge.llm.openai import OpenAIProvider

logger = logging.getLogger(__name__)

_ALLOWED_PROVIDERS = frozenset({"anthropic", "openai", "gemini"})


class LLMRouter:
    async def complete(
        self,
        prompt: str,
        provider: str,
        model: str,
        max_tokens: int = 1024,
    ) -> tuple[str, int]:
        if provider not in _ALLOWED_PROVIDERS:
            raise ValueError(provider)
        try:
            if provider == "anthropic":
                return await AnthropicProvider.complete(prompt, model, max_tokens)
            if provider == "openai":
                return await OpenAIProvider.complete(prompt, model, max_tokens)
            return await GeminiProvider.complete(prompt, model, max_tokens)
        except httpx.HTTPError as exc:
            logger.error(
                "LLM HTTP error [%s/%s]: %s",
                provider,
                model,
                exc,
                exc_info=True,
            )
            raise LLMError(str(exc), provider, model) from exc
        except LLMError as exc:
            logger.error(
                "LLM error [%s/%s]: %s",
                provider,
                model,
                exc.message,
                exc_info=True,
            )
            raise LLMError(exc.message, provider, model) from exc
