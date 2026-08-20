from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

class ToolProvider(ABC):
    @property
    @abstractmethod
    def namespace(self) -> str:
        ...

    @abstractmethod
    def list_tools(self) -> list[ToolSpec]:
        ...

    @abstractmethod
    async def call(self, tool_name: str, args: dict[str, Any]) -> str:
        """Run one of this provider's tools and return its result as text.

        Never raises: anything the caller got wrong — an unknown tool name,
        a missing or malformed argument — comes back as an "error: ..." string
        so the model can see it and retry.
        """
        ...

