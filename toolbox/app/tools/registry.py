from __future__ import annotations

import logging
from typing import Any

import jsonschema

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


class ToolRegistry:
    """Owns every provider and routes a qualified name to the right one.

    The route no longer knows which providers exist: it asks the registry for
    the menu, and hands whatever the model called straight back to dispatch().
    """

    def __init__(self) -> None:
        self._providers: dict[str, ToolProvider] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(self, provider: ToolProvider) -> None:
        ns = provider.namespace
        if ns in self._providers:
            raise ValueError(f"namespace '{ns}' already registered")

        # Collect first, commit second: a provider that names two tools the
        # same must leave the registry exactly as it was.
        specs: dict[str, ToolSpec] = {}
        for spec in provider.list_tools():
            qname = f"{ns}.{spec.name}"
            if qname in self._specs or qname in specs:
                raise ValueError(f"duplicate tool '{qname}'")
            specs[qname] = spec

        self._providers[ns] = provider
        self._specs.update(specs)
        logger.info(
            "provider registered",
            extra={"namespace": ns, "tool_count": len(specs)},
        )

    def list_all(self) -> list[tuple[str, ToolSpec]]:
        """Every tool as (qualified_name, spec), in registration order."""
        return list(self._specs.items())

    @staticmethod
    def _validate(spec: ToolSpec, args: dict[str, Any]) -> None:
        """Check the model's arguments against the tool's declared schema.

        Raises ToolValidationError so the route can answer in text: a model
        that sent the wrong type can read why and retry, which is the whole
        point of declaring input_schema in the first place.
        """
        try:
            jsonschema.validate(instance=args, schema=spec.input_schema)
        except jsonschema.ValidationError as exc:
            raise ToolValidationError(exc.message) from exc

    async def dispatch(self, qualified_name: str, args: dict[str, Any]) -> str:
        namespace, _, bare_name = qualified_name.partition(".")

        # One membership test covers all three ways to be unknown: no dot at
        # all, an unknown namespace, or a known namespace with no such tool.
        if qualified_name not in self._specs:
            logger.warning("unknown tool dispatch", extra={"tool": qualified_name})
            raise UnknownToolError(qualified_name)

        # Validate BEFORE the provider runs: a provider should only ever see
        # arguments that match the schema it published.
        self._validate(self._specs[qualified_name], args)

        return await self._providers[namespace].call(bare_name, args)
