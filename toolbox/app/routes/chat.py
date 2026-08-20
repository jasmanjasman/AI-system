import ast
import json
import operator
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import get_llama_client

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Stops `9 ** 9 ** 9` from tying up the worker on a single request.
_MAX_EXPONENT = 64


def _eval_node(node: ast.AST) -> int | float:
    """Evaluate one arithmetic node, refusing anything that is not arithmetic."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"not a number: {node.value!r}")
        return node.value

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported sign: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent too large: {right}")
        return op(_eval_node(node.left), right)

    raise ValueError(f"unsupported expression: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression without handing the model the runtime.

    Errors come back as text rather than raising, so the model can see what it
    got wrong and retry instead of the request failing.
    """
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except ZeroDivisionError:
        return "error: division by zero"
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError) as exc:
        return f"error: {exc}"

CALCULATOR_TOOL = {
    "type": "function",
    "function":{
        "name": "calculator",
        "description": "Evaluate an arithmetic expression like '23 * 47'.",
        "parameters":{
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
}

@router.post("/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    client = get_llama_client()

    messages: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
    tools = [CALCULATOR_TOOL]

    assistant_msg = await client.chat_with_tools(messages, tools)

    tool_calls = assistant_msg.get("tool_calls")
    if not tool_calls:
        return {"answer": assistant_msg.get("content", ""), "used_tool": False}

    messages.append(assistant_msg)
    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = json.loads(call["function"]["arguments"])

        if fn_name == "calculator":
            result = calculator(fn_args["expression"])

        else:
            result = f"unknown tool: {fn_name}"

        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })

    final_msg = await client.chat_with_tools(messages, tools)
    return {"answer": final_msg.get("content", ""), "used_tool": True}