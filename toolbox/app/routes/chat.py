import json
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import get_llama_client

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str


def calculator(expression: str) -> str:
    result = eval(expression)
    return str(result)

CALCULATOR_TOOL = {
    "type": "function",
    "function":{
        "name": "calculator",
        "description": "Evaluate an arithmetic expression like '23 * 47'.",
        "parameters":{
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
}

@router.post("/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    client = get_llama_client()

    messages: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
    tools = [CALCULATOR_TOOL]

    assistant_msg = await client.chat_with_tools(messages, tools)

    tool_calls = assistant_msg.get("tool_calls")
    if not tool_calls:
        return {"answer": assistant_msg.get("content", ""), "used_tool": False}

    messages.append(assistant_msg)
    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = json.loads(call["function"]["arguments"])

        if fn_name == "calculator":
            result = calculator(fn_args["expression"])

        else:
            result = f"unknown tool: {fn_name}"

        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })

    final_msg = await client.chat_with_tools(messages, tools)
    return {"answer": final_msg.get("content", ""), "used_tool": True}