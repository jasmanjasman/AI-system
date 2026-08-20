"""POST /chat must survive whatever shape the model sends back.

The xfail cases below are known gaps in the route itself, left in place
deliberately. They will flip to XPASS the moment chat.py handles them.
"""
import pytest

from app.adapters.llama_client import LlamaClientError
from conftest import tool_call


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
