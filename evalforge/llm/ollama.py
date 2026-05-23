"""Ollama local LLM adapter for EvalForge."""
from __future__ import annotations
import asyncio
import time
import httpx
from evalforge.llm.errors import LLMError

class OllamaProvider:
    async def complete(self, prompt: str, model: str, max_tokens: int = 1024) -> tuple[str, int]:
        def _sync_call() -> str:
            with httpx.Client(timeout=60) as client:
                response = client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}},
                )
                response.raise_for_status()
                return response.json()["response"]
        try:
            t0 = time.monotonic()
            text = await asyncio.to_thread(_sync_call)
            latency_ms = int((time.monotonic() - t0) * 1000)
            return text, latency_ms
        except Exception as exc:
            raise LLMError(str(exc), provider="ollama", model=model) from exc