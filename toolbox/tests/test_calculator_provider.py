"""The provider must never execute anything but arithmetic, and never raise."""
import asyncio

import pytest

from app.tools.providers.calculator import CalculatorProvider


@pytest.fixture
def provider():
    return CalculatorProvider()


def call(provider, args, tool_name="calculator"):
    return asyncio.run(provider.call(tool_name, args))


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("23 * 47", "1081"),
        ("2 + 3 * 4", "14"),
        ("-5 + 2", "-3"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("2 ** 10", "1024"),
        ("1 / 4", "0.25"),
    ],
)
def test_evaluates_arithmetic(provider, expression, expected):
    assert call(provider, {"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "[x for x in range(3)]",
        "print(1)",
        "1 if True else 2",
    ],
)
def test_refuses_everything_that_is_not_arithmetic(provider, expression):
    """Regression guard for the eval() hole: these must be rejected, not run."""
    assert call(provider, {"expression": expression}).startswith("error:")


def test_rejects_huge_exponent_instead_of_hanging(provider):
    assert call(provider, {"expression": "9 ** 9 ** 9"}).startswith("error:")


@pytest.mark.parametrize(
    "args, tool_name",
    [
        ({"expression": "1/0"}, "calculator"),
        ({"expression": "23 *"}, "calculator"),
        ({}, "calculator"),
        ({"expression": 23}, "calculator"),
        ({"expression": None}, "calculator"),
        ({"expression": "1+1"}, "calc"),
    ],
)
def test_bad_input_returns_text_and_never_raises(provider, args, tool_name):
    """The route calls this unguarded, so anything that raises here is a 500."""
    assert call(provider, args, tool_name).startswith("error:")


def test_lists_its_tool(provider):
    specs = provider.list_tools()
    assert [s.name for s in specs] == ["calculator"]
    assert provider.namespace == "math"
