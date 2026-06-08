from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., min_length=8, max_length=64)


class ChatResponse(BaseModel):
    reply: str
    lead_triggered: bool = False
    # source_chunks now returned so frontend can render Source Disclosure
    source_chunks: Optional[list[str]] = None


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
