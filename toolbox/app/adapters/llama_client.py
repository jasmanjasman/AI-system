from typing import Any

import httpx

from app.config import Settings


class LlamaClientError(RuntimeError):
    """The llama.cpp server was unreachable or returned an unusable response."""


class LlamaClient:
    """Thin async client for a local llama.cpp server's OpenAI-compatible API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.llama_base_url,
            timeout=settings.llama_timeout,
        )

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LlamaClientError(f"llama.cpp request failed: {exc}") from exc

        data = response.json()
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaClientError(f"unexpected response shape: {data!r}") from exc

    async def chat(self, prompt: str) -> str:
        message = await self._post_chat(
            {
                "model": self._settings.llama_model,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        content = message.get("content")
        if not isinstance(content, str):
            raise LlamaClientError(f"unexpected response shape: {message!r}")
        return content

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self._post_chat(
            {
                "model": self._settings.llama_model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
        )

    async def aclose(self) -> None:
        await self._client.aclose()
