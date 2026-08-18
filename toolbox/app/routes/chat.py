import ast
import json
import operator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.adapters.llama_client import LlamaClient, LlamaClientError
from app.dependencies import get_llama_client

router = APIRouter()

# A tool call is model-controlled input, so the expression never reaches eval().
# Only these node types and operators survive the walk; anything else is rejected.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Bounds a ** b before it is evaluated; 2 ** 10**9 would otherwise hang the worker.
_MAX_EXPONENT = 1000


class CalculatorError(ValueError):
    """The expression was not a plain arithmetic expression we are willing to evaluate."""


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"unsupported constant: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if op is operator.pow and abs(right) > _MAX_EXPONENT:
            raise CalculatorError(f"exponent too large: {right}")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unsupported operator: {type(node.op).__name__}")
        return op(_evaluate(node.operand))

    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Raises CalculatorError on anything else."""
    # Models write "2^16" for exponentiation, but Python parses ^ as bitwise XOR,
    # which would silently answer 18. Normalise to ** rather than allowing BitXor.
    expression = expression.replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse {expression!r}") from exc

    try:
        return str(_evaluate(tree.body))
    except CalculatorError:
        raise
    except (ArithmeticError, ValueError) as exc:
        raise CalculatorError(f"could not evaluate {expression!r}: {exc}") from exc


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression like '23 * 47'. "
            "Supports + - * / // % and ** (or ^) for powers."
        ),
        "parameters": {
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

TOOLS = [CALCULATOR_TOOL]

# The model may chain calls (compute, then reuse the result). Bounded so a model
# that keeps calling tools cannot loop forever.
MAX_TOOL_ROUNDS = 5


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)


def _run_tool(name: str, raw_args: str) -> str:
    if name != "calculator":
        return f"unknown tool: {name}"
    try:
        args = json.loads(raw_args)
        return calculator(args["expression"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return f"invalid arguments for {name}: {exc}"
    except CalculatorError as exc:
        # Handed back to the model rather than raised: it can retry or explain.
        return f"error: {exc}"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    client: LlamaClient = Depends(get_llama_client),
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
    used_tool = False

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            assistant_msg = await client.chat_with_tools(messages, TOOLS)

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                return {"answer": assistant_msg.get("content") or "", "used_tool": used_tool}

            used_tool = True
            messages.append(assistant_msg)
            for call in tool_calls:
                result = _run_tool(call["function"]["name"], call["function"]["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )
    except LlamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raise HTTPException(
        status_code=504,
        detail=f"model kept calling tools after {MAX_TOOL_ROUNDS} rounds",
    )
