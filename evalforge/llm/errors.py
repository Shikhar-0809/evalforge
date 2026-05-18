"""Structured errors for LLM provider failures."""


class LLMError(Exception):

    def __init__(self, message: str, provider: str, model: str) -> None:
        self.message = message
        self.provider = provider
        self.model = model
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.provider}/{self.model}] {self.message}"
