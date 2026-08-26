"""The registry replaces the route's namespace if-ladder, so it owns two jobs:
build the menu across every provider, and route one qualified name to one
provider — raising a typed error instead of a KeyError when nothing matches.
"""
import asyncio

import pytest

from app.tools.contracts import ToolProvider, ToolSpec
from app.tools.providers.calculator import CalculatorProvider
from app.tools.registry import ToolRegistry, UnknownToolError


def spec(name):
    return ToolSpec(name=name, description=f"the {name} tool", input_schema={"type": "object"})


class FakeProvider(ToolProvider):
    """Records what it was asked to run, so dispatch routing is observable."""

    def __init__(self, namespace, tool_names):
        self._namespace = namespace
        self._tool_names = tool_names
        self.calls = []

    @property
    def namespace(self):
        return self._namespace

    def list_tools(self):
        return [spec(n) for n in self._tool_names]

    async def call(self, tool_name, args):
        self.calls.append((tool_name, args))
        return f"{self._namespace}:{tool_name}"


@pytest.fixture
def registry():
    return ToolRegistry()


def dispatch(registry, name, args=None):
    return asyncio.run(registry.dispatch(name, args or {}))


def test_empty_registry_has_an_empty_menu(registry):
    assert registry.list_all() == []


def test_menu_spans_every_provider(registry):
    registry.register(FakeProvider("math", ["calculator"]))
    registry.register(FakeProvider("web", ["search", "fetch"]))

    assert [qname for qname, _ in registry.list_all()] == [
        "math.calculator",
        "web.search",
        "web.fetch",
    ]


def test_menu_carries_the_provider_spec(registry):
    registry.register(CalculatorProvider())

    (qname, tool_spec), = registry.list_all()
    assert qname == "math.calculator"
    assert tool_spec == CalculatorProvider().list_tools()[0]  # the provider's own spec


def test_rejects_a_second_provider_on_the_same_namespace(registry):
    registry.register(FakeProvider("math", ["calculator"]))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeProvider("math", ["statistics"]))


def test_rejects_a_provider_that_names_two_tools_the_same(registry):
    with pytest.raises(ValueError, match="duplicate tool"):
        registry.register(FakeProvider("math", ["calculator", "calculator"]))


def test_a_rejected_provider_leaves_nothing_behind(registry):
    with pytest.raises(ValueError):
        registry.register(FakeProvider("math", ["calculator", "calculator"]))

    assert registry.list_all() == []
    with pytest.raises(UnknownToolError):
        dispatch(registry, "math.calculator")


def test_dispatch_routes_to_the_owning_provider(registry):
    math = FakeProvider("math", ["calculator"])
    web = FakeProvider("web", ["search"])
    registry.register(math)
    registry.register(web)

    assert dispatch(registry, "web.search", {"q": "hi"}) == "web:search"
    assert web.calls == [("search", {"q": "hi"})]
    assert math.calls == []


def test_dispatch_strips_only_the_first_dot(registry):
    """A tool name is namespace + first dot + the rest, whatever the rest is."""
    provider = FakeProvider("math", ["stats.mean"])
    registry.register(provider)

    assert dispatch(registry, "math.stats.mean") == "math:stats.mean"
    assert provider.calls == [("stats.mean", {})]


@pytest.mark.parametrize(
    "name",
    [
        "calculator",       # namespace dropped
        "nope.calculator",  # unknown namespace
        "math.nope",        # known namespace, unknown tool
        "math.",            # empty tool name
        "",                 # nothing at all
    ],
)
def test_unknown_names_raise_the_typed_error(registry, name):
    registry.register(FakeProvider("math", ["calculator"]))

    with pytest.raises(UnknownToolError):
        dispatch(registry, name)


def test_unknown_error_carries_the_name_the_model_used(registry):
    with pytest.raises(UnknownToolError) as excinfo:
        dispatch(registry, "math.nope")

    assert "math.nope" in str(excinfo.value)


def test_dispatch_does_not_swallow_the_provider_reply(registry):
    """Provider-level errors are text, not exceptions — they must pass through."""
    registry.register(CalculatorProvider())

    assert dispatch(registry, "math.calculator", {"expression": "1/0"}).startswith("error:")
    assert dispatch(registry, "math.calculator", {"expression": "23 * 47"}) == "1081"
