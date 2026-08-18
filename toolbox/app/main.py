from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dependencies import get_llama_client
from app.routes import chat as chat_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = get_llama_client()
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="toolbox", lifespan=lifespan)
app.include_router(chat_routes.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
