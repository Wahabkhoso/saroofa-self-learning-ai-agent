"""
LLM provider abstraction.

The rest of the app (agent, evaluator, planner) only ever talks to this
interface, never to a specific vendor SDK. Swapping providers means
changing `LLM_PROVIDER` in `.env` - no code changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMImage:
    """A single image attached to a message - base64-encoded, with its
    MIME type. Keeping this as a separate small dataclass (rather than
    overloading `content`) means providers that don't support images yet
    (or a text-only conversation) never have to think about it."""

    base64_data: str
    mime_type: str  # e.g. "image/png", "image/jpeg"


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant" | "system"
    content: str
    image: LLMImage | None = None


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return the response."""
        ...


class LLMProviderError(RuntimeError):
    """Raised when the underlying provider call fails or is misconfigured."""