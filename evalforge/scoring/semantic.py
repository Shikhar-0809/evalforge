"""Semantic similarity scoring using sentence embeddings."""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _encode_pair_score(self, response: str, reference: str) -> float:
        emb_ref = self._model.encode(reference, convert_to_numpy=True)
        emb_resp = self._model.encode(response, convert_to_numpy=True)
        return self._cosine_similarity(emb_ref, emb_resp)

    async def score(self, response: str, reference: Optional[str]) -> float:
        if reference is None or reference == "":
            return 0.0
        raw = await asyncio.to_thread(self._encode_pair_score, response, reference)
        return max(0.0, raw)
