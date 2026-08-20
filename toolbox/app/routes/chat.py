import json
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import get_calculator_provider, get_llama_client
from app.tools.contracts import ToolProvider, ToolSpec

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str


def _spec_to_openai(namespace: str, spec: ToolSpec) -> dict[str, Any]:
    """Turn a ToolSpec into the OpenAI function-calling menu entry.
    NOTE the namespaced name 'math.calculator' — this is the F1 naming scheme.
    (The charset/dot issue this creates is a deliberate lesson at M6.)"""
    return {
        "type": "function",
        "function": {
            "name": f"{namespace}.{spec.name}",
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _build_menu(provider: ToolProvider) -> list[dict[str, Any]]:
    return [_spec_to_openai(provider.namespace, s) for s in provider.list_tools()]


@router.post("/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    client = get_llama_client()
    provider = get_calculator_provider()

    messages: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
    tools = _build_menu(provider)

    assistant_msg = await client.chat_with_tools(messages, tools)

    tool_calls = assistant_msg.get("tool_calls")
    if not tool_calls:
        return {"answer": assistant_msg.get("content", ""), "used_tool": False}

    messages.append(assistant_msg)
    for call in tool_calls:
        qualified = call["function"]["name"]
        fn_args = json.loads(call["function"]["arguments"])

        namespace, _, bare_name = qualified.partition(".")

        if namespace == provider.namespace:
            result = await provider.call(bare_name, fn_args)
        else:
            result = f"unknown namespace: {namespace}"

        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })

    final_msg = await client.chat_with_tools(messages, tools)
    return {"answer": final_msg.get("content", ""), "used_tool": True}
