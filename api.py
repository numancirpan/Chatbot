from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.chatbot import RAGChatbot


app = FastAPI(
    title="Duzce University RAG Chatbot API",
    version="1.0.0",
    description="Streamlit arayuzundeki RAG motorunu HTTP uzerinden sunar.",
)

_bot_lock = Lock()
_chat_lock = Lock()
_bot_instance: RAGChatbot | None = None
_session_bots: dict[str, RAGChatbot] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Ogrencinin sorusu")
    program_scope: str | None = Field(default=None, description="Opsiyonel program/birim kapsami")
    session_id: str | None = Field(default=None, description="Opsiyonel sohbet/session kimligi")


class ResetMemoryRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Opsiyonel sohbet/session kimligi")


class SourceItem(BaseModel):
    kategori: str
    baslik: str
    url: str


class ChatResponse(BaseModel):
    query: str
    cevap: str
    kaynaklar: List[SourceItem]


def _normalize_session_id(session_id: str | None) -> str | None:
    if not session_id:
        return None
    normalized = session_id.strip()
    if not normalized:
        return None
    if len(normalized) > 128:
        raise HTTPException(status_code=400, detail="session_id en fazla 128 karakter olabilir")
    return normalized


def _apply_program_scope(bot: RAGChatbot, program_scope: str | None) -> None:
    if program_scope is None:
        return
    normalized_scope = program_scope.strip()
    bot.program_scope = normalized_scope
    if not bot.conversation_state.get("program_scope"):
        bot.conversation_state["program_scope"] = normalized_scope


def get_bot(program_scope: str | None = None, session_id: str | None = None) -> RAGChatbot:
    global _bot_instance
    normalized_session_id = _normalize_session_id(session_id)
    with _bot_lock:
        if normalized_session_id:
            bot = _session_bots.get(normalized_session_id)
            if bot is None:
                bot = RAGChatbot(program_scope=program_scope or "")
                _session_bots[normalized_session_id] = bot
            else:
                _apply_program_scope(bot, program_scope)
            return bot

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
        "memory_turns": len(bot.message_history.messages) // 2,
        "session_count": len(_session_bots),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        bot = get_bot(payload.program_scope, payload.session_id)
        with _chat_lock:
            result = bot.chat(payload.query)
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reset-memory")
def reset_memory(payload: ResetMemoryRequest | None = None) -> Dict[str, str]:
    session_id = payload.session_id if payload else None
    bot = get_bot(session_id=session_id)
    with _chat_lock:
        bot.clear_memory()
    return {"status": "ok"}
