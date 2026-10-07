from __future__ import annotations

import base64
import time

from app.core.config import settings
from app.llm.base import LLMMessage, LLMProvider, LLMProviderError, LLMResponse

# Google's free-tier models occasionally return 503 UNAVAILABLE under high
# demand - this is transient, not a config problem, and usually clears up
# within a few seconds. Retrying a couple of times with a short backoff
# turns "high demand" moments into a slightly slower successful call
# instead of an immediate failure.
_RETRYABLE_STATUS_MARKERS = ("503", "UNAVAILABLE", "overloaded", "RESOURCE_EXHAUSTED")
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.5


def _is_retryable(error_text: str) -> bool:
    return any(marker in error_text for marker in _RETRYABLE_STATUS_MARKERS)


class GeminiProvider(LLMProvider):
    """Google Gemini provider, via the google-genai SDK.

    Gemini has no separate "system" role - the system prompt is passed
    via `system_instruction` in the generation config instead, and
    "assistant" turns are mapped to Gemini's "model" role.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or settings.gemini_api_key
        self._model = model or settings.model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMProviderError("GEMINI_API_KEY is not set. Add it to your .env file.")
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        from google.genai import types

        contents = []
        for m in messages:
            parts = [types.Part.from_text(text=m.content)]
            if m.image is not None:
                parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(m.image.base64_data), mime_type=m.image.mime_type
                    )
                )
            contents.append(types.Content(role="model" if m.role == "assistant" else "user", parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_output_tokens=max_tokens or settings.llm_max_tokens,
        )

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < _MAX_RETRIES and _is_retryable(str(exc)):
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise LLMProviderError(f"Gemini API call failed: {exc}") from exc
            else:
                text = response.text or ""
                usage = getattr(response, "usage_metadata", None)
                return LLMResponse(
                    content=text,
                    model=self._model,
                    input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                )

        # Unreachable in practice (the loop either returns or raises), but
        # keeps type-checkers happy and fails loudly if it ever is reached.
        raise LLMProviderError(f"Gemini API call failed: {last_error}")