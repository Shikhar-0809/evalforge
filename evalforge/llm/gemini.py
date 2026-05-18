"""Google Gemini API adapter for EvalForge."""

from __future__ import annotations

import asyncio
import logging
import time

import google.generativeai as genai

from evalforge.config import settings
from evalforge.llm.errors import LLMError

logger = logging.getLogger(__name__)


class GeminiProvider:
    @staticmethod
    async def complete(prompt: str, model: str, max_tokens: int) -> tuple[str, int]:
        if not settings.GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set", "gemini", model)

        def _sync_call() -> str:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            gm = genai.GenerativeModel(model)
            response = gm.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                ),
            )
            text = getattr(response, "text", None)
            if text is not None:
                return text
            if not response.candidates:
                raise RuntimeError("Gemini returned no candidates")
            parts = response.candidates[0].content.parts
            return "".join(getattr(p, "text", "") for p in parts)

        try:
            t0 = time.monotonic()
            text = await asyncio.to_thread(_sync_call)
            latency_ms = int((time.monotonic() - t0) * 1000)
            return text, latency_ms
        except Exception as exc:
            logger.exception("Gemini completion failed")
            raise LLMError(str(exc), "gemini", model) from exc
