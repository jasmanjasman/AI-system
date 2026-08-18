from functools import lru_cache

from app.adapters.llama_client import LlamaClient
from app.config import get_settings


@lru_cache
def get_llama_client() -> LlamaClient:
    return LlamaClient(get_settings())
