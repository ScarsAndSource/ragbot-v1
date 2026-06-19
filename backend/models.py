"""
models.py — RAGbot v2
Phase 4: added LLMOutput (structured output schema), ChatResponseV2 (lead_intent, model_used)
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., min_length=8, max_length=64)


class ChatResponse(BaseModel):
    """v1 shape — kept for backward compatibility. /chat now returns ChatResponseV2."""
    reply: str
    lead_triggered: bool = False
    source_chunks: Optional[list[str]] = None


# ── Phase 4 additions ─────────────────────────────────────────────────────────

class LLMOutput(BaseModel):
    """
    Structured output schema for the answer LLM (non-streaming path).
    Groq JSON mode guarantees this shape; _parse_llm_output falls back
    to (raw_text, True) if JSON parse fails.
    """
    answer: str
    source_sufficient: bool = True


class ChatResponseV2(BaseModel):
    """
    Extended chat response returned by /chat (Phase 4+) and /chat/stream
    fallback. Superset of ChatResponse — existing clients parsing reply,
    lead_triggered, source_chunks are unaffected.
    """
    reply: str
    lead_triggered: bool = False
    source_chunks: Optional[list[str]] = None
    lead_intent: Optional[list[str]] = None   # classifier intent_signals
    model_used: str = ""                       # "llama-3.3-70b-versatile" | fallback | "cache"


# ── Unchanged from v1 ─────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    pages_processed: int


class LeadCapture(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    phone: str = Field(..., min_length=7, max_length=20)
    session_id: str
    query: Optional[str] = None


class SessionValidateResponse(BaseModel):
    valid: bool
    session_id: str