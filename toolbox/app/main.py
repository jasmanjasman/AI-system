from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.adapters.llama_client import LlamaClient, LlamaClientError
from app.dependencies import get_llama_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = get_llama_client()
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="toolbox", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/raw")
async def chat_raw(
    request: ChatRequest,
    client: LlamaClient = Depends(get_llama_client),
) -> dict[str, str]:
    try:
        answer = await client.chat(request.prompt)
    except LlamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"answer": answer}
