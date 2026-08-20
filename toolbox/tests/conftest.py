"""Test fixtures: a stubbed llama.cpp so tests never need a model server."""
import json

import pytest
from fastapi.testclient import TestClient

from app.adapters.llama_client import LlamaClientError
from app.main import app as fastapi_app
from app.routes import chat as chat_routes


class FakeLlamaClient:
    """Replays a scripted first reply, then answers 'final answer'."""

    def __init__(self, first_reply, error=None):
        self._first_reply = first_reply
        self._error = error

    async def chat_with_tools(self, messages, tools):
        if self._error is not None:
            raise self._error
        if any(m.get("role") == "tool" for m in messages):
            return {"role": "assistant", "content": "final answer"}
        return self._first_reply

    async def aclose(self):
        pass


@pytest.fixture
def chat_client(monkeypatch):
    """Returns post(first_reply) -> httpx.Response for POST /chat."""

    def post(first_reply, error=None):
        monkeypatch.setattr(
            chat_routes, "get_llama_client", lambda: FakeLlamaClient(first_reply, error)
        )
        with TestClient(fastapi_app) as client:
            return client.post("/chat", json={"prompt": "irrelevant"})

    return post


def tool_call(name, arguments, call_id="call_1"):
    """Build the assistant message a model sends when it wants a tool."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


__all__ = ["FakeLlamaClient", "LlamaClientError", "chat_client", "tool_call"]
