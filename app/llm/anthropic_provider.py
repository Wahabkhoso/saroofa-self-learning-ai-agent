from __future__ import annotations

from app.core.config import settings
from app.llm.base import LLMMessage, LLMProvider, LLMProviderError, LLMResponse


def to_anthropic_message(m: LLMMessage) -> dict:
    """Module-level (not a closure) so it's unit-testable without a live
    API call or client."""
    if m.image is None:
        return {"role": m.role, "content": m.content}
    return {
        "role": m.role,
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": m.image.mime_type,
                    "data": m.image.base64_data,
                },
            },
            {"type": "text", "text": m.content},
        ],
    }


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model or settings.model_name
        self._client = None  # lazy-init so import doesn't require a key present

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMProviderError(
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file."
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
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
        try:
            response = client.messages.create(
                model=self._model,
                system=system or "",
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                messages=[to_anthropic_message(m) for m in messages],
            )
        except Exception as exc:  # noqa: BLE001 - surface as our own error type
            raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            content=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )