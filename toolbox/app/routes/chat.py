import json
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import get_llama_client, get_registry
from app.tools.contracts import ToolSpec
from app.tools.registry import ToolRegistry, ToolValidationError, UnknownToolError

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str


def _spec_to_openai(qualified_name: str, spec: ToolSpec) -> dict[str, Any]:
    """Turn a ToolSpec into the OpenAI function-calling menu entry.
    NOTE the namespaced name 'math.calculator' — this is the F1 naming scheme.
    (The charset/dot issue this creates is a deliberate lesson at M6.)"""
    return {
        "type": "function",
        "function": {
            "name": qualified_name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _build_menu(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Menu built from the REGISTRY — spans ALL providers automatically.
    Add a provider in dependencies.py and it shows up here for free."""
    return [_spec_to_openai(qname, spec) for qname, spec in registry.list_all()]


@router.post("/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    client = get_llama_client()
    registry = get_registry()

    messages: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
    tools = _build_menu(registry)

    assistant_msg = await client.chat_with_tools(messages, tools)

    tool_calls = assistant_msg.get("tool_calls")
    if not tool_calls:
        return {"answer": assistant_msg.get("content", ""), "used_tool": False}

    messages.append(assistant_msg)
    for call in tool_calls:
        qualified = call["function"]["name"]           # "math.calculator"
        fn_args = json.loads(call["function"]["arguments"])

        # NO namespace if-branch anymore. Hand the whole thing to the Registry.
        try:
            result = await registry.dispatch(qualified, fn_args)
        except UnknownToolError:
            result = f"unknown tool: {qualified}"      # tell the LLM, let it recover
        except ToolValidationError as exc:
            # Hand the specific schema failure back: a model that reads
            # "5 is not of type 'string'" can retry the same tool correctly.
            result = f"invalid arguments for {qualified}: {exc}"

        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })

    final_msg = await client.chat_with_tools(messages, tools)
    return {"answer": final_msg.get("content", ""), "used_tool": True}
