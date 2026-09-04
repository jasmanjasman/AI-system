from functools import lru_cache

from app.adapters.llama_client import LlamaClient
from app.config import get_settings
from app.tools.caller import Caller
from app.tools.providers.calculator import CalculatorProvider
from app.tools.registry import ToolRegistry


@lru_cache
def get_llama_client() -> LlamaClient:
    return LlamaClient(get_settings())


@lru_cache
def get_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorProvider())
    return registry


@lru_cache
def get_caller() -> Caller:
    """The caller every request runs as.

    A single configured identity for now — the one place authentication
    plugs in later. Note what it does NOT do: read permissions off the
    request. A client that can name its own permissions has none.
    """
    settings = get_settings()
    granted = {p.strip() for p in settings.caller_permissions.split(",") if p.strip()}
    return Caller(user_id=settings.caller_id, permissions=frozenset(granted))
