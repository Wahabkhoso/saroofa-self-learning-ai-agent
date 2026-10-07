from __future__ import annotations

from app.core.config import settings
from app.llm.base import LLMMessage, LLMProvider, LLMProviderError, LLMResponse


def to_openai_message(m: LLMMessage) -> dict:
    """Module-level (not a closure) so it's unit-testable without a live
    API call or client."""
    if m.image is None:
        return {"role": m.role, "content": m.content}
    return {
        "role": m.role,
        "content": [
            {"type": "text", "text": m.content},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{m.image.mime_type};base64,{m.image.base64_data}"},
            },
        ],
    }


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMProviderError("OPENAI_API_KEY is not set. Add it to your .env file.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
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

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(to_openai_message(m) for m in messages)

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=chat_messages,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"OpenAI API call failed: {exc}") from exc

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )