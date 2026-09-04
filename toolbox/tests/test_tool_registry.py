"""The registry replaces the route's namespace if-ladder, so it owns two jobs:
build the menu across every provider, and route one qualified name to one
provider — raising a typed error instead of a KeyError when nothing matches.
"""
import asyncio

import pytest

from app.tools.caller import Caller
from app.tools.contracts import ToolProvider, ToolSpec
from app.tools.providers.calculator import CalculatorProvider
from app.tools.registry import (
    PermissionDeniedError,
    ToolRegistry,
    ToolValidationError,
    UnknownToolError,
)

# Most tools here declare no required_permission, so the plainest possible
# caller can reach them. Permission tests below opt in explicitly.
ANYONE = Caller.anonymous()


def spec(name, schema=None, permission=None):
    return ToolSpec(
        name=name,
        description=f"the {name} tool",
        input_schema=schema or {"type": "object"},
        required_permission=permission,
    )


class FakeProvider(ToolProvider):
    """Records what it was asked to run, so dispatch routing is observable."""

    def __init__(self, namespace, tool_names, schema=None, permission=None):
        self._namespace = namespace
        self._tool_names = tool_names
        self._schema = schema
        self._permission = permission
        self.calls = []

    @property
    def namespace(self):
        return self._namespace

    def list_tools(self):
        return [spec(n, self._schema, self._permission) for n in self._tool_names]

    async def call(self, tool_name, args):
        self.calls.append((tool_name, args))
        return f"{self._namespace}:{tool_name}"


@pytest.fixture
def registry():
    return ToolRegistry()


def dispatch(registry, name, args=None, caller=ANYONE):
    return asyncio.run(registry.dispatch(caller, name, args or {}))


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


# --- argument validation --------------------------------------------------
# A tool publishes an input_schema; the registry holds the model to it, so a
# provider only ever sees arguments that match what it advertised.

CALC_ARGS = {"expression": "23 * 47"}
CALC_SCHEMA = CalculatorProvider().list_tools()[0].input_schema


def test_valid_arguments_reach_the_provider(registry):
    registry.register(CalculatorProvider())

    assert dispatch(registry, "math.calculator", CALC_ARGS) == "1081"


def test_wrong_argument_type_is_rejected_before_the_provider_runs(registry):
    provider = FakeProvider("math", ["calculator"], schema=CALC_SCHEMA)
    registry.register(provider)

    with pytest.raises(ToolValidationError):
        dispatch(registry, "math.calculator", {"expression": 5})

    assert provider.calls == []  # the provider never saw it


def test_missing_required_argument_is_rejected(registry):
    registry.register(CalculatorProvider())

    with pytest.raises(ToolValidationError, match="required property"):
        dispatch(registry, "math.calculator", {})


def test_validation_error_explains_what_was_wrong(registry):
    """The message is fed to the model verbatim, so it has to be readable."""
    registry.register(CalculatorProvider())

    with pytest.raises(ToolValidationError) as excinfo:
        dispatch(registry, "math.calculator", {"expression": 5})

    assert "not of type 'string'" in str(excinfo.value)


def test_permissive_schema_accepts_anything(registry):
    """A provider that declares no properties opts out of validation."""
    provider = FakeProvider("echo", ["say"])
    registry.register(provider)

    assert dispatch(registry, "echo.say", {"anything": [1, 2, 3]}) == "echo:say"


def test_unknown_tool_is_checked_before_validation(registry):
    """An unknown name must not be reported as a schema problem."""
    registry.register(CalculatorProvider())

    with pytest.raises(UnknownToolError):
        dispatch(registry, "math.nope", {"expression": 5})


# --- schema checking at registration --------------------------------------
# A provider that publishes a schema which is not valid JSON Schema is a
# programming error, not something a model can recover from. Catching it in
# register() turns a 500 on some later request into a failure at startup.

BROKEN_SCHEMA = {"type": "not-a-real-type"}


def test_a_broken_schema_is_rejected_at_registration(registry):
    with pytest.raises(ValueError, match="invalid input_schema"):
        registry.register(FakeProvider("bad", ["tool"], schema=BROKEN_SCHEMA))


def test_the_rejection_names_the_offending_tool(registry):
    with pytest.raises(ValueError) as excinfo:
        registry.register(FakeProvider("bad", ["tool"], schema=BROKEN_SCHEMA))

    assert "bad.tool" in str(excinfo.value)


def test_a_broken_schema_leaves_nothing_behind(registry):
    """Same rollback guarantee as a duplicate name: all or nothing."""
    with pytest.raises(ValueError):
        registry.register(FakeProvider("bad", ["ok", "broken"], schema=BROKEN_SCHEMA))

    assert registry.list_all() == []
    registry.register(FakeProvider("bad", ["ok"]))  # namespace is not burned


def test_dispatch_never_raises_a_raw_schema_error(registry):
    """Whatever reaches dispatch is a schema the registry already checked."""
    registry.register(CalculatorProvider())

    with pytest.raises(ToolValidationError):
        dispatch(registry, "math.calculator", {"expression": 5})


# --- error messages the model has to act on -------------------------------
# The message goes back to the model verbatim, so it must say WHICH argument
# was wrong, not just that something was.

MULTI_ARG_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {"type": "string"},
        "precision": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["expression"],
}


@pytest.fixture
def multi(registry):
    registry.register(FakeProvider("math", ["calc"], schema=MULTI_ARG_SCHEMA))
    return registry


def test_the_message_names_the_offending_argument(multi):
    with pytest.raises(ToolValidationError) as excinfo:
        dispatch(multi, "math.calc", {"expression": "1+1", "precision": "two"})

    assert str(excinfo.value) == "precision: 'two' is not of type 'integer'"


def test_the_message_points_into_a_nested_value(multi):
    with pytest.raises(ToolValidationError) as excinfo:
        dispatch(multi, "math.calc", {"expression": "1+1", "tags": ["ok", 7]})

    assert str(excinfo.value) == "tags[1]: 7 is not of type 'string'"


def test_a_missing_property_is_not_given_a_path(multi):
    """The message already names the field; a path prefix would just repeat it."""
    with pytest.raises(ToolValidationError) as excinfo:
        dispatch(multi, "math.calc", {})

    assert str(excinfo.value) == "'expression' is a required property"


# --- permissions ------------------------------------------------------------
# A tool declares required_permission; the caller carries what it has. The
# menu is filtered as a courtesy, but dispatch is the boundary that decides.

READER = Caller(user_id="reader", permissions=frozenset({"files.read"}))


@pytest.fixture
def mixed_registry():
    """One public tool and one that requires 'files.read'."""
    registry = ToolRegistry()
    registry.register(FakeProvider("math", ["calculator"]))
    registry.register(FakeProvider("fs", ["read"], permission="files.read"))
    return registry


def test_a_tool_is_public_unless_it_asks_for_a_permission():
    """The ToolSpec default is None, so providers opt IN to being restricted."""
    assert spec("calculator").required_permission is None
    assert ANYONE.has(spec("calculator").required_permission)


def test_menu_hides_what_the_caller_may_not_use(mixed_registry):
    assert [q for q, _ in mixed_registry.list_for(ANYONE)] == ["math.calculator"]


def test_menu_shows_the_restricted_tool_to_a_caller_who_holds_it(mixed_registry):
    assert [q for q, _ in mixed_registry.list_for(READER)] == ["math.calculator", "fs.read"]


def test_an_unrelated_permission_does_not_open_the_door(mixed_registry):
    writer = Caller(user_id="writer", permissions=frozenset({"files.write"}))

    assert [q for q, _ in mixed_registry.list_for(writer)] == ["math.calculator"]
    with pytest.raises(PermissionDeniedError):
        dispatch(mixed_registry, "fs.read", caller=writer)


def test_list_all_is_unfiltered(mixed_registry):
    """Diagnostics still need to see every tool, permissions notwithstanding."""
    assert [q for q, _ in mixed_registry.list_all()] == ["math.calculator", "fs.read"]


def test_dispatch_runs_the_tool_for_a_caller_who_holds_the_permission(mixed_registry):
    assert dispatch(mixed_registry, "fs.read", caller=READER) == "fs:read"


def test_dispatch_refuses_a_caller_who_does_not(mixed_registry):
    with pytest.raises(PermissionDeniedError):
        dispatch(mixed_registry, "fs.read", caller=ANYONE)


def test_a_denied_call_never_reaches_the_provider(registry):
    """The point of the whole exercise: the tool body must not run."""
    provider = FakeProvider("fs", ["write"], permission="files.write")
    registry.register(provider)

    with pytest.raises(PermissionDeniedError):
        dispatch(registry, "fs.write", {"path": "/etc/passwd"}, caller=READER)

    assert provider.calls == []


def test_a_public_tool_still_works_for_a_caller_with_no_permissions(mixed_registry):
    assert dispatch(mixed_registry, "math.calculator", caller=ANYONE) == "math:calculator"


def test_permission_is_checked_before_the_arguments_are(mixed_registry):
    """A caller who may not use a tool gets a flat refusal — not a schema
    critique that maps out the arguments of a tool they cannot reach."""
    registry = ToolRegistry()
    registry.register(FakeProvider("fs", ["read"], schema=CALC_SCHEMA, permission="files.read"))

    with pytest.raises(PermissionDeniedError):
        dispatch(registry, "fs.read", {"expression": 5}, caller=ANYONE)


def test_an_unknown_tool_is_still_unknown_not_denied(mixed_registry):
    """Order matters the other way too: 'no such tool' outranks 'not allowed'."""
    with pytest.raises(UnknownToolError):
        dispatch(mixed_registry, "fs.nope", caller=ANYONE)


def test_the_denial_names_the_caller_and_the_permission(mixed_registry):
    """Not for the model — for whoever reads the log after a refusal."""
    with pytest.raises(PermissionDeniedError) as excinfo:
        dispatch(mixed_registry, "fs.read", caller=ANYONE)

    message = str(excinfo.value)
    assert "anonymous" in message
    assert "files.read" in message
    assert "fs.read" in message
