from __future__ import annotations

import logging
from typing import Any

import jsonschema
from jsonschema.validators import validator_for

from app.tools.caller import Caller
from app.tools.contracts import ToolProvider, ToolSpec

logger = logging.getLogger(__name__)


class UnknownToolError(Exception):
    """No provider/tool matches the qualified name.

    A typed error so the route can answer the model in text instead of
    letting a raw KeyError become a 500.
    """


class ToolValidationError(Exception):
    """The model's arguments do not satisfy the tool's input_schema.

    Separate from UnknownToolError because the model should react
    differently: an unknown tool means 'pick a real tool', a validation
    error means 'call the same tool again with corrected arguments'.
    """

class PermissionDeniedError(Exception):
    """The caller lacks the permission the tool requires.

    Separate from the other two again because the model should react
    differently once more: retrying will not help, and neither will
    fixing the arguments. The tool is simply not available to this caller.
    """



def _describe(exc: jsonschema.ValidationError) -> str:
    """Render a validation failure as something a model can act on.

    exc.message alone says "5 is not of type 'string'" without naming the
    argument, which is unusable the moment a tool has more than one. Prefix
    the path to the offending value; a missing-property error already names
    the field itself and has no path, so it is left alone.
    """
    path = "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in exc.absolute_path
    ).lstrip(".")
    return f"{path}: {exc.message}" if path else exc.message


class ToolRegistry:
    """Owns every provider and routes a qualified name to the right one.

    The route no longer knows which providers exist: it asks the registry for
    the menu, and hands whatever the model called straight back to dispatch().
    """

    def __init__(self) -> None:
        self._providers: dict[str, ToolProvider] = {}
        self._specs: dict[str, ToolSpec] = {}
        self._validators: dict[str, Any] = {}

    def register(self, provider: ToolProvider) -> None:
        ns = provider.namespace
        if ns in self._providers:
            raise ValueError(f"namespace '{ns}' already registered")

        # Collect first, commit second: a provider that names two tools the
        # same — or publishes a schema that is not valid JSON Schema — must
        # leave the registry exactly as it was.
        specs: dict[str, ToolSpec] = {}
        validators: dict[str, Any] = {}
        for spec in provider.list_tools():
            qname = f"{ns}.{spec.name}"
            if qname in self._specs or qname in specs:
                raise ValueError(f"duplicate tool '{qname}'")
            specs[qname] = spec
            validators[qname] = self._compile(qname, spec)

        self._providers[ns] = provider
        self._specs.update(specs)
        self._validators.update(validators)
        logger.info(
            "provider registered",
            extra={"namespace": ns, "tool_count": len(specs)},
        )

    @staticmethod
    def _compile(qualified_name: str, spec: ToolSpec) -> Any:
        """Check a tool's schema once and keep the validator built from it.

        Checking here means a provider that publishes a broken schema fails
        at startup, where the traceback names it, instead of on whichever
        request first happens to call that tool. Keeping the validator also
        takes schema compilation off the per-call path.
        """
        cls = validator_for(spec.input_schema)
        try:
            cls.check_schema(spec.input_schema)
        except jsonschema.SchemaError as exc:
            raise ValueError(
                f"tool '{qualified_name}' has an invalid input_schema: {exc.message}"
            ) from exc
        return cls(spec.input_schema)

    def list_all(self) -> list[tuple[str, ToolSpec]]:
        """Every registered tool as (qualified_name, spec), in registration order.

        The unfiltered view: use it for diagnostics and admin listings, not
        for building a menu — see list_for().
        """
        return list(self._specs.items())

    def list_for(self, caller: Caller) -> list[tuple[str, ToolSpec]]:
        """Only the tools this caller may use.

        Filtering the menu is a courtesy to the model, not the security
        boundary: it stops the model proposing tools that can only be
        refused. dispatch() re-checks, because a model can name a tool it
        was never shown.
        """
        return [
            (qname, spec)
            for qname, spec in self._specs.items()
            if caller.has(spec.required_permission)
        ]

    def _validate(self, qualified_name: str, args: dict[str, Any]) -> None:
        """Check the model's arguments against the tool's declared schema.

        Raises ToolValidationError so the route can answer in text: a model
        that sent the wrong type can read why and retry, which is the whole
        point of declaring input_schema in the first place.
        """
        try:
            self._validators[qualified_name].validate(args)
        except jsonschema.ValidationError as exc:
            raise ToolValidationError(_describe(exc)) from exc

    async def dispatch(self, caller: Caller, qualified_name: str, args: dict[str, Any]) -> str:
        namespace, _, bare_name = qualified_name.partition(".")

        # One membership test covers all three ways to be unknown: no dot at
        # all, an unknown namespace, or a known namespace with no such tool.
        if qualified_name not in self._specs:
            logger.warning("unknown tool dispatch", extra={"tool": qualified_name})
            raise UnknownToolError(qualified_name)

        spec = self._specs[qualified_name]

        # Permission BEFORE validation: a caller who may not use this tool
        # gets one flat refusal, not a schema critique that maps out the
        # arguments of a tool they cannot reach. This is the boundary — the
        # filtered menu upstream is only a hint.
        if not caller.has(spec.required_permission):
            logger.warning(
                "permission denied",
                extra={
                    "tool": qualified_name,
                    "user_id": caller.user_id,
                    "required": spec.required_permission,
                },
            )
            raise PermissionDeniedError(
                f"caller '{caller.user_id}' lacks "
                f"'{spec.required_permission}' for {qualified_name}"
            )

        # Validate BEFORE the provider runs: a provider should only ever see
        # arguments that match the schema it published.
        self._validate(qualified_name, args)

        return await self._providers[namespace].call(bare_name, args)
