from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.chatbot import RAGChatbot


app = FastAPI(
    title="Duzce University RAG Chatbot API",
    version="1.0.0",
    description="Streamlit arayuzundeki RAG motorunu HTTP uzerinden sunar.",
)

_bot_lock = Lock()
_bot_instance: RAGChatbot | None = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Ogrencinin sorusu")
    program_scope: str | None = Field(default=None, description="Opsiyonel program/birim kapsami")


class SourceItem(BaseModel):
    kategori: str
    baslik: str
    url: str


class ChatResponse(BaseModel):
    query: str
    cevap: str
    kaynaklar: List[SourceItem]


def get_bot(program_scope: str | None = None) -> RAGChatbot:
    global _bot_instance
    with _bot_lock:
        if _bot_instance is None or (program_scope and _bot_instance.program_scope != program_scope):
            _bot_instance = RAGChatbot(program_scope=program_scope or "")
        return _bot_instance


@app.get("/health")
def health() -> Dict[str, Any]:
    bot = get_bot()
    return {
        "status": "ok",
        "vector_count": bot.vector_count,
        "chunk_count": len(bot.chunks),
        "memory_turns": len(bot.memory) // 2,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        bot = get_bot(payload.program_scope)
        result = bot.chat(payload.query)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reset-memory")
def reset_memory() -> Dict[str, str]:
    bot = get_bot()
    bot.clear_memory()
    return {"status": "ok"}
