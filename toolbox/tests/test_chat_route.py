"""POST /chat must survive whatever shape the model sends back.

The xfail cases below are known gaps in the route itself, left in place
deliberately. They will flip to XPASS the moment chat.py handles them.
"""
import pytest

from app.adapters.llama_client import LlamaClientError
from app.routes import chat as chat_routes
from app.tools.contracts import ToolProvider, ToolSpec
from app.tools.providers.calculator import CalculatorProvider
from app.tools.registry import ToolRegistry
from conftest import tool_call


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

    menu = chat_routes._build_menu(registry)

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
