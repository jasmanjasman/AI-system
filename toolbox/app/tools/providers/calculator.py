# app/tools/providers/calculator.py  — THE FIRST ADAPTER
from __future__ import annotations

import ast
import operator
from typing import Any

from app.tools.contracts import ToolProvider, ToolSpec

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


class CalculatorProvider(ToolProvider):
    """Wraps arithmetic as a ToolProvider. This is the concrete adapter that
    plugs into the port. The route no longer knows 'calculator' exists by name —
    it only knows 'some ToolProvider'."""

    @property
    def namespace(self) -> str:
        return "math"

    def list_tools(self) -> list[ToolSpec]:
        # The menu entry now lives WITH the tool, not scattered in the route.
        # One source of truth: describe + implement in the same class.
        return [
            ToolSpec(
                name="calculator",
                description="Evaluate an arithmetic expression like '23 * 47'.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression to evaluate.",
                        }
                    },
                    "required": ["expression"],
                },
            )
        ]

    async def call(self, tool_name: str, args: dict[str, Any]) -> str:
        """Run one of this provider's tools.

        Nothing here raises. Bad input — an unknown tool name, a missing or
        non-string expression — is answered with text so the model can see what
        it got wrong and retry, instead of the whole request failing.
        """
        # A provider may own several tools; dispatch on the bare name internally.
        if tool_name != "calculator":
            return f"error: {self.namespace} has no tool '{tool_name}'"

        expression = args.get("expression")
        if not isinstance(expression, str):
            return f"error: 'expression' must be a string, got {expression!r}"
        return self._calculator(expression)

    @staticmethod
    def _calculator(expression: str) -> str:
        """Evaluate an arithmetic expression without handing the model the runtime."""
        try:
            return str(_eval_node(ast.parse(expression, mode="eval").body))
        except ZeroDivisionError:
            return "error: division by zero"
        except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError) as exc:
            return f"error: {exc}"
