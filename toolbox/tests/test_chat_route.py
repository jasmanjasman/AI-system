"""POST /chat must survive whatever shape the model sends back.

The xfail cases below are known gaps in the route itself, left in place
deliberately. They will flip to XPASS the moment chat.py handles them.
"""
import pytest

from fastapi.testclient import TestClient

from app.adapters.llama_client import LlamaClientError
from app.main import app as fastapi_app
from app.routes import chat as chat_routes
from app.tools.caller import Caller
from app.tools.contracts import ToolProvider, ToolSpec
from app.tools.providers.calculator import CalculatorProvider
from app.tools.registry import ToolRegistry
from conftest import FakeLlamaClient, tool_call


class _EchoProvider(ToolProvider):
    """A provider the route has never heard of, to prove nothing is hardcoded."""

    def __init__(self):
        self.calls = []

    @property
    def namespace(self):
        return "echo"

    def list_tools(self):
        return [ToolSpec(name="say", description="Echo text back.", input_schema={"type": "object"})]

    async def call(self, tool_name, args):
        self.calls.append((tool_name, args))
        return str(args.get("text"))


def test_plain_reply_without_tools(chat_client):
    response = chat_client({"role": "assistant", "content": "hello"})
    assert response.status_code == 200
    assert response.json() == {"answer": "hello", "used_tool": False}


def test_happy_path_runs_the_tool(chat_client):
    response = chat_client(tool_call("math.calculator", {"expression": "23 * 47"}))
    assert response.status_code == 200
    assert response.json() == {"answer": "final answer", "used_tool": True}


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("math.calculator", {"expression": "1/0"}),   # arithmetic error
        ("math.calculator", {"expression": "23 *"}),  # malformed expression
        ("math.calculator", {}),                      # missing argument
        ("math.nope", {"expression": "1+1"}),         # hallucinated tool name
        ("calculator", {"expression": "1+1"}),        # namespace dropped (M6)
    ],
)
def test_malformed_tool_calls_do_not_500(chat_client, name, arguments):
    assert chat_client(tool_call(name, arguments)).status_code == 200


@pytest.mark.xfail(strict=True, reason="chat.py: json.loads(arguments) is unguarded")
def test_unparseable_arguments_json(chat_client):
    assert chat_client(tool_call("math.calculator", "{oops")).status_code == 200


@pytest.mark.xfail(strict=True, reason="chat.py: call['id'] assumes the key exists")
def test_missing_tool_call_id(chat_client):
    reply = tool_call("math.calculator", {"expression": "1+1"})
    del reply["tool_calls"][0]["id"]
    assert chat_client(reply).status_code == 200


@pytest.mark.xfail(strict=True, reason="chat.py: LlamaClientError is never caught")
def test_model_server_down_is_502_not_500(chat_client):
    response = chat_client(None, error=LlamaClientError("connection refused"))
    assert response.status_code == 502


def test_menu_uses_namespaced_names_from_the_registry():
    """The name the model sees is what dispatch() will be handed back."""
    registry = ToolRegistry()
    registry.register(CalculatorProvider())

    menu = chat_routes._build_menu(registry, Caller.anonymous())

    assert [entry["function"]["name"] for entry in menu] == ["math.calculator"]
    assert menu[0]["type"] == "function"
    assert menu[0]["function"]["parameters"] == CalculatorProvider().list_tools()[0].input_schema


def test_route_serves_a_provider_it_was_never_told_about(monkeypatch, chat_client):
    """No 'math' branch left in the route: a brand-new namespace just works."""
    echo = _EchoProvider()
    registry = ToolRegistry()
    registry.register(CalculatorProvider())
    registry.register(echo)
    monkeypatch.setattr(chat_routes, "get_registry", lambda: registry)

    response = chat_client(tool_call("echo.say", {"text": "hi"}))

    assert response.status_code == 200
    assert echo.calls == [("say", {"text": "hi"})]


class _SecretProvider(ToolProvider):
    """A restricted provider, to prove the route honours the boundary."""

    def __init__(self):
        self.calls = []

    @property
    def namespace(self):
        return "secret"

    def list_tools(self):
        return [
            ToolSpec(
                name="read",
                description="Read the secret.",
                input_schema={"type": "object"},
                required_permission="secrets.read",
            )
        ]

    async def call(self, tool_name, args):
        self.calls.append((tool_name, args))
        return "the secret"


def _registry_with_a_secret():
    registry = ToolRegistry()
    registry.register(CalculatorProvider())
    registry.register(_SecretProvider())
    return registry


def test_menu_offers_only_what_the_caller_may_use():
    """No point showing the model a tool whose every call would be refused."""
    registry = _registry_with_a_secret()

    anyone = chat_routes._build_menu(registry, Caller.anonymous())
    holder = chat_routes._build_menu(registry, Caller("agent", frozenset({"secrets.read"})))

    assert [e["function"]["name"] for e in anyone] == ["math.calculator"]
    assert [e["function"]["name"] for e in holder] == ["math.calculator", "secret.read"]


def test_a_tool_the_model_was_not_shown_is_refused_not_run(monkeypatch, chat_client):
    """The menu is a hint; a model can still name a tool it never saw."""
    registry = _registry_with_a_secret()
    monkeypatch.setattr(chat_routes, "get_registry", lambda: registry)
    monkeypatch.setattr(chat_routes, "get_caller", Caller.anonymous)

    response = chat_client(tool_call("secret.read", {}))

    assert response.status_code == 200
    assert registry._providers["secret"].calls == []


def test_the_refusal_does_not_leak_the_permission_name(monkeypatch):
    """What the model is told: the door is shut. Not which key it wants."""
    tool_replies = []

    class Recorder(FakeLlamaClient):
        async def chat_with_tools(self, messages, tools):
            tool_replies.extend(m for m in messages if m.get("role") == "tool")
            return await super().chat_with_tools(messages, tools)

    monkeypatch.setattr(chat_routes, "get_registry", _registry_with_a_secret)
    monkeypatch.setattr(chat_routes, "get_caller", Caller.anonymous)
    monkeypatch.setattr(
        chat_routes, "get_llama_client", lambda: Recorder(tool_call("secret.read", {}))
    )

    with TestClient(fastapi_app) as client:
        client.post("/chat", json={"prompt": "irrelevant"})

    (refusal,) = tool_replies
    assert refusal["content"] == "permission denied: secret.read"
    assert "secrets.read" not in refusal["content"]


def test_a_permitted_call_runs_through_the_route(monkeypatch, chat_client):
    registry = _registry_with_a_secret()
    monkeypatch.setattr(chat_routes, "get_registry", lambda: registry)
    monkeypatch.setattr(chat_routes, "get_caller", lambda: Caller("agent", frozenset({"secrets.read"})))

    response = chat_client(tool_call("secret.read", {}))

    assert response.status_code == 200
    assert registry._providers["secret"].calls == [("read", {})]
