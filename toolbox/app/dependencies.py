from functools import lru_cache

from app.adapters.llama_client import LlamaClient
from app.config import get_settings
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
