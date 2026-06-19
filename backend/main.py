"""
main.py — RAGbot v2
Phase 1 changes from v1:
  [P1-1] REMOVED: chromadb telemetry suppression
  [P1-2] ADDED: Google credentials written from GOOGLE_CREDENTIALS_JSON env var
  [P1-3] ADDED: Langfuse init at startup + health trace
  [P1-4] ADDED: Upstash Redis session store with in-memory fallback
  [P1-5] ADDED: Qdrant keepalive in APScheduler
  [P1-6] ADDED: Upstash ping in /health
Phase 2 changes:
  [P2-1] ADDED: SHA256 dedup check in /upload — same doc → 200, skip re-ingest
  [P2-2] ADDED: delete old Qdrant collection on re-upload of different doc
  [P2-3] ADDED: Supabase document registration after ingest
  [P2-4] UPDATED: /upload imports _sha256_bytes, document_exists,
                   register_document, delete_old_collections from rag
Phase 4 changes:
  [P4-1] UPDATED: /chat — unpacks 5-tuple, returns ChatResponseV2
                   (adds model_used, lead_intent fields)
  [P4-2] ADDED: /chat/stream — StreamingResponse wrapping stream_chat_response;
                   event_generator accumulates tokens + updates history on 'done';
                   falls back to non-streaming ChatResponseV2 when groq_streaming=False
"""

import json
import logging
import os
import smtplib
import tempfile
import time
from contextlib import asynccontextmanager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Generator, Optional

import gspread
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from google.oauth2 import service_account
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from cache import semantic_cache
from chat import get_chat_response, stream_chat_response
from config import get_settings
from models import (
    ChatRequest,
    ChatResponseV2,
    LeadCapture,
    SessionValidateResponse,
    UploadResponse,
)
from rag import (
    _sha256_bytes,
    delete_old_collections,
    document_exists,
    ping_qdrant,
    process_pdf,
    register_document,
    session_exists,
)

settings = get_settings()
logger = logging.getLogger("ragbot.main")
logging.basicConfig(level=logging.INFO)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Session store (Upstash Redis → in-memory fallback) ────────────────────────
_session_store: dict[str, dict] = {}
_redis_client = None


def _init_redis():
    global _redis_client
    if settings.upstash_redis_url and settings.upstash_redis_token:
        try:
            from upstash_redis import Redis
            _redis_client = Redis(
                url=settings.upstash_redis_url,
                token=settings.upstash_redis_token,
            )
            _redis_client.ping()
            logger.info("Upstash Redis session store connected")
        except Exception as exc:
            logger.warning(
                "Upstash Redis unavailable (%s) — falling back to in-memory store", exc,
            )
            _redis_client = None
    else:
        logger.info("Upstash not configured — using in-memory session store")


def _session_get(session_id: str) -> dict:
    if _redis_client:
        try:
            raw = _redis_client.get(f"session:{session_id}")
            return json.loads(raw) if raw else {}
        except Exception:
            pass
    return _session_store.get(session_id, {})


def _session_set(session_id: str, data: dict) -> None:
    if _redis_client:
        try:
            _redis_client.setex(f"session:{session_id}", 86400, json.dumps(data))
            return
        except Exception:
            pass
    _session_store[session_id] = data


def _session_delete(session_id: str) -> None:
    if _redis_client:
        try:
            _redis_client.delete(f"session:{session_id}")
        except Exception:
            pass
    _session_store.pop(session_id, None)


# ── Google credentials from env var ──────────────────────────────────────────
def _write_google_credentials() -> None:
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip()
    if not creds_json:
        logger.warning("GOOGLE_CREDENTIALS_JSON not set — Google Sheets integration disabled")
        return
    try:
        creds_data = json.loads(creds_json)
        path = settings.google_credentials_path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(creds_data, f)
        logger.info("Google credentials written to %s", path)
    except Exception as exc:
        logger.error("Failed to write Google credentials: %s", exc)


# ── Langfuse init ─────────────────────────────────────────────────────────────
_langfuse = None


def _init_langfuse():
    global _langfuse
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse
            _langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )
            trace = _langfuse.trace(name="startup", metadata={"env": settings.env})
            trace.update(output={"status": "healthy"})
            logger.info("Langfuse initialized — startup trace sent")
        except Exception as exc:
            logger.warning("Langfuse init failed: %s", exc)
    else:
        logger.info("Langfuse keys not set — observability disabled")


# ── Scheduler ─────────────────────────────────────────────────────────────────
_scheduler = BackgroundScheduler()


def _ping_self() -> None:
    try:
        import httpx
        httpx.get("https://ragbot-v2.onrender.com/health", timeout=10)
    except Exception:
        pass


def _keepalive_qdrant() -> None:
    alive = ping_qdrant()
    if alive:
        logger.debug("Qdrant keepalive ping OK")
    else:
        logger.warning("Qdrant keepalive ping failed")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _write_google_credentials()
    _init_redis()
    _init_langfuse()
    _scheduler.add_job(_ping_self, "interval", minutes=10, id="render_keepalive")
    _scheduler.add_job(_keepalive_qdrant, "interval", minutes=8, id="qdrant_keepalive")
    _scheduler.start()
    logger.info("Scheduler started — Render + Qdrant keepalive active")
    yield
    _scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAGbot v2",
    description="Hybrid RAG chatbot — Qdrant + Cohere + Groq",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ───────────────────────────────────────────────────────────
def _require_api_key(request: Request) -> None:
    key = request.headers.get("X-API-Key", "")
    if key != settings.ragbot_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Email helper (SMTP — replaced with Gmail API in Phase 5) ─────────────────
def _send_lead_email(lead: LeadCapture, bot_session_summary: str = "") -> None:
    if not settings.email_sender or not settings.email_password:
        logger.info("Email not configured — skipping lead email")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"New Lead: {lead.name}"
        msg["From"] = settings.email_sender
        msg["To"] = settings.email_recipient
        body = (
            f"Name: {lead.name}\n"
            f"Phone: {lead.phone}\n"
            f"Session: {lead.session_id}\n"
            f"Query: {lead.query or 'N/A'}\n"
            f"Session summary: {bot_session_summary or 'N/A'}\n"
        )
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(settings.email_sender, settings.email_password)
            smtp.sendmail(settings.email_sender, settings.email_recipient, msg.as_string())
        logger.info("Lead email sent for %s", lead.name)
    except Exception as exc:
        logger.error("Lead email failed: %s", exc)


# ── Google Sheets helper ──────────────────────────────────────────────────────
def _write_to_sheet(lead: LeadCapture) -> None:
    creds_path = settings.google_credentials_path
    if not os.path.exists(creds_path):
        logger.warning("Google credentials not found at %s — skipping sheet write", creds_path)
        return
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open(settings.google_sheet_name)
        ws = sh.sheet1
        ws.append_row([
            lead.name,
            lead.phone,
            lead.session_id,
            lead.query or "",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])
        logger.info("Lead written to Google Sheets: %s", lead.name)
    except Exception as exc:
        logger.error("Google Sheets write failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    qdrant_ok = ping_qdrant()
    upstash_ok = False
    if _redis_client:
        try:
            _redis_client.ping()
            upstash_ok = True
        except Exception:
            pass
    return {
        "status": "ok",
        "qdrant": "ok" if qdrant_ok else "unreachable",
        "upstash": "ok" if upstash_ok else ("not_configured" if not _redis_client else "error"),
        "langfuse": "enabled" if _langfuse else "disabled",
        "streaming": settings.groq_streaming,
        "env": settings.env,
    }


# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    _: None = Depends(_require_api_key),
):
    if not session_id or len(session_id) < 8 or len(session_id) > 64:
        raise HTTPException(status_code=422, detail="Invalid session_id")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    pdf_bytes = await file.read()

    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — max {settings.max_file_size_mb}MB",
        )

    doc_hash = _sha256_bytes(pdf_bytes)
    doc_name = file.filename

    if document_exists(session_id, doc_hash):
        logger.info(
            "Dedup HIT — session=%s | hash=%s… | skipping re-ingest",
            session_id, doc_hash[:12],
        )
        return UploadResponse(
            success=True,
            message="Document already loaded for this session.",
            session_id=session_id,
            pages_processed=0,
        )

    if session_exists(session_id):
        logger.info(
            "New document for existing session=%s — deleting old Qdrant collection",
            session_id,
        )
        delete_old_collections(session_id)
        semantic_cache.clear(session_id)
        _session_delete(session_id)

    def _progress(msg: str) -> None:
        logger.info("[upload:%s] %s", session_id[:8], msg)

    try:
        result = process_pdf(
            pdf_bytes=pdf_bytes,
            session_id=session_id,
            doc_name=doc_name,
            progress_callback=_progress,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("process_pdf failed for session=%s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="PDF processing failed")

    register_document(session_id, doc_hash, doc_name, result["chunks"])

    _session_set(session_id, {
        "doc_name": doc_name,
        "pages": result["pages"],
        "chunks": result["chunks"],
        "created_at": time.time(),
    })

    logger.info(
        "Upload complete — session=%s | %d pages | %d parents | %d children",
        session_id, result["pages"], result["parents"], result["children"],
    )

    return UploadResponse(
        success=True,
        message=(
            f"Ingested '{doc_name}' — "
            f"{result['pages']} pages, "
            f"{result['parents']} parent chunks, "
            f"{result['children']} child chunks."
        ),
        session_id=session_id,
        pages_processed=result["pages"],
    )


# ── Chat [P4-1] ───────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponseV2)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    _: None = Depends(_require_api_key),
):
    session_id = body.session_id

    if not session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail="No document loaded for this session. Please upload a PDF first.",
        )

    session_data = _session_get(session_id)
    history: list[dict] = session_data.get("history", [])

    reply, lead_triggered, source_chunks, model_used, intent_signals = get_chat_response(
        session_id=session_id,
        user_message=body.message,
        conversation_history=history,
    )

    history.append({"role": "user", "content": body.message})
    history.append({"role": "assistant", "content": reply})
    session_data["history"] = history
    _session_set(session_id, session_data)

    return ChatResponseV2(
        reply=reply,
        lead_triggered=lead_triggered,
        source_chunks=source_chunks,
        model_used=model_used,
        lead_intent=intent_signals,
    )


# ── Chat stream [P4-2] ────────────────────────────────────────────────────────
@app.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    _: None = Depends(_require_api_key),
):
    """
    Streaming chat endpoint — Server-Sent Events (text/event-stream).

    Falls back to non-streaming ChatResponseV2 when settings.groq_streaming=False.

    SSE protocol:
      data: {"token": "..."}                                — one per LLM token
      data: {"token": "<full>"}                             — cache hit (single burst)
      data: {"done": true, "model": "...",
             "guardrail": false, "lead": false}
      data: {"replace": "..."}                              — guardrail correction
      data: [DONE]                                          — stream terminator

    History update: performed inside event_generator on receipt of 'done' event,
    ensuring it happens before [DONE] is flushed to the client.
    """
    session_id = body.session_id

    if not session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail="No document loaded for this session. Please upload a PDF first.",
        )

    # ── Streaming disabled in config — serve same response on this route ──────
    if not settings.groq_streaming:
        session_data = _session_get(session_id)
        history: list[dict] = session_data.get("history", [])
        reply, lead_triggered, source_chunks, model_used, intent_signals = get_chat_response(
            session_id=session_id,
            user_message=body.message,
            conversation_history=history,
        )
        history.append({"role": "user", "content": body.message})
        history.append({"role": "assistant", "content": reply})
        session_data["history"] = history
        _session_set(session_id, session_data)
        return ChatResponseV2(
            reply=reply,
            lead_triggered=lead_triggered,
            source_chunks=source_chunks,
            model_used=model_used,
            lead_intent=intent_signals,
        )

    session_data = _session_get(session_id)
    history = session_data.get("history", [])

    def event_generator() -> Generator[str, None, None]:
        """
        Thin wrapper around stream_chat_response:
        - Forwards all SSE events verbatim to the client.
        - On 'done' event: writes accumulated reply to session history.
        Parsing our own SSE output is intentional — we control both sides
        and need to extract the final reply without a second network round-trip.
        """
        accumulated_tokens: list[str] = []
        guardrail_reply: Optional[str] = None

        for event in stream_chat_response(session_id, body.message, history):
            yield event

            if not event.startswith("data: ") or event.startswith("data: [DONE]"):
                continue

            try:
                payload = json.loads(event[6:].strip())
            except (json.JSONDecodeError, ValueError):
                continue

            if "token" in payload:
                accumulated_tokens.append(payload["token"])
            elif "replace" in payload:
                # Guardrail fired — override accumulated tokens with corrected reply
                guardrail_reply = payload["replace"]
            elif payload.get("done"):
                final_reply = guardrail_reply or "".join(accumulated_tokens)
                if final_reply:
                    history.append({"role": "user", "content": body.message})
                    history.append({"role": "assistant", "content": final_reply})
                    session_data["history"] = history
                    _session_set(session_id, session_data)
                    logger.info(
                        "Stream complete — session=%s | model=%s | history_turns=%d",
                        session_id, payload.get("model", "?"), len(history) // 2,
                    )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Session clear ─────────────────────────────────────────────────────────────
@app.post("/session/{session_id}/clear")
async def clear_session(
    session_id: str,
    _: None = Depends(_require_api_key),
):
    _session_delete(session_id)
    semantic_cache.clear(session_id)
    return {"success": True, "session_id": session_id}


# ── Session validate ──────────────────────────────────────────────────────────
@app.get("/session/{session_id}/validate", response_model=SessionValidateResponse)
async def validate_session(session_id: str):
    return SessionValidateResponse(
        valid=session_exists(session_id),
        session_id=session_id,
    )


# ── Lead capture ──────────────────────────────────────────────────────────────
@app.post("/leads")
@limiter.limit("5/minute")
async def capture_lead(
    request: Request,
    lead: LeadCapture,
    _: None = Depends(_require_api_key),
):
    session_data = _session_get(lead.session_id)
    summary = f"Doc: {session_data.get('doc_name', 'unknown')}"
    _write_to_sheet(lead)
    _send_lead_email(lead, summary)
    logger.info("Lead captured — name=%s | session=%s", lead.name, lead.session_id)
    return {"success": True, "message": "Lead captured successfully"}


# ── Pipeline logs (Phase 5 will populate Supabase) ────────────────────────────
@app.get("/pipeline/logs")
async def pipeline_logs(_: None = Depends(_require_api_key)):
    return {"logs": [], "note": "Populated in Phase 5"}


# ── Low-confidence leads (Phase 5) ────────────────────────────────────────────
@app.get("/leads/low_confidence")
async def low_confidence_leads(_: None = Depends(_require_api_key)):
    return {"leads": [], "note": "Populated in Phase 5"}