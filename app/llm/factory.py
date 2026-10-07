from __future__ import annotations

from app.core.config import settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.openai_provider import OpenAIProvider


def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAIProvider()
    if settings.llm_provider == "gemini":
        return GeminiProvider()
    return AnthropicProvider()
