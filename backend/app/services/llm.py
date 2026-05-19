"""LLM provider abstraction.

A single `complete(system, user) -> str` interface with two implementations.
The active provider is chosen by `settings.llm_provider`.
"""

import httpx
from langsmith import traceable

from backend.app.config import settings


class LLMError(RuntimeError):
    pass


class LLMProvider:
    name: str

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    name = "ollama"

    @traceable(run_type="llm")
    def complete(self, system: str, user: str) -> str:
        try:
            response = httpx.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                },
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return response.json()["message"]["content"]


class OpenAIProvider(LLMProvider):
    name = "openai"

    @traceable(run_type="llm")
    def complete(self, system: str, user: str) -> str:
        try:
            response = httpx.post(
                f"{settings.openai_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return response.json()["choices"][0]["message"]["content"]


def get_provider() -> LLMProvider:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("llm_provider is 'openai' but OPENAI_API_KEY is not set")
        return OpenAIProvider()
    if settings.llm_provider == "ollama":
        return OllamaProvider()
    raise LLMError(f"Unknown llm_provider: {settings.llm_provider}")
