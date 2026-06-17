"""
main.py — RAGbot v2, Phase 1

Changes from v1:
  [P1-1] REMOVED: chromadb telemetry suppression (ChromaDB gone)
  [P1-2] ADDED:   Google credentials written from GOOGLE_CREDENTIALS_JSON env var
                  to a temp file at startup — survives Render ephemeral filesystem
  [P1-3] ADDED:   Langfuse client initialized at startup, one health trace sent
                  so the dashboard is visible before Phase 5 wires per-stage spans
  [P1-4] ADDED:   Qdrant keepalive job in scheduler (every 9 minutes)
                  Qdrant free clusters auto-suspend after 7 days of no traffic
  [P1-5] UPDATED: Session store prefers Upstash Redis (UPSTASH_REDIS_URL env var)
                  before falling back to legacy REDIS_URL, then in-memory
  [P1-6] ADDED:   /health now pings Qdrant and Upstash, returns their status
  [P1-7] ADDED:   /cache/stats and /cache/global-stats routes for observability
  [P1-8] KEPT:    All v1 routes unchanged (upload, chat, lead, session, history)
  [P1-9] KEPT:    SMTP email helper (replaced with Gmail API in Phase 5)
  [P1-10] KEPT:   SSE progress stream (/upload/progress, /upload/streamed)

Phases that will touch this file next:
  Phase 2: dedup SHA256 check in upload route, Supabase document write
  Phase 4: /chat/stream route added
  Phase 5: SMTP replaced by Gmail API, Supabase pipeline_logs write,
            lead dedup via Upstash, confidence gate, intent routing,
            Langfuse per-stage spans wired, /pipeline/logs route
  Phase 6: widget.js static serve, client CORS from CLIENT_DOMAIN env var
"""

import json
import logging
import os
import smtplib
import tempfile
import uuid
import asyncio
from contextlib import asynccontextmanager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Callable, Optional

import gspread
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from google.oauth2.service_account import Credentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from cache import semantic_cache
from chat import get_chat_response
from config import get_settings
from models import (
    ChatRequest,
    ChatResponse,
    LeadCapture,
    SessionValidateResponse,
    UploadResponse,
)
from rag import delete_old_collections, ping_qdrant, process_pdf, session_exists, validate_pdf_bytes

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("ragbot")

settings = get_settings()

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── API key auth ──────────────────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> None:
    expected = settings.ragbot_api_key
    if not expected:
        return
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── [P1-2] Google credentials from env var ────────────────────────────────────
# Render's filesystem resets on every deploy. Storing creds as a file on disk
# breaks on every restart. Instead: paste the full service account JSON as the
# env var GOOGLE_CREDENTIALS_JSON on Render, and we write it to a temp file
# at startup. The temp file survives the process lifetime (not across restarts,
# but it's recreated at every startup so it's always fresh).
_GOOGLE_CREDS_TEMP_PATH: Optional[str] = None


def _write_google_credentials() -> Optional[str]:
    """
    Write GOOGLE_CREDENTIALS_JSON env var content to a temp file.
    Returns the temp file path, or None if the env var is not set.

    Called once during lifespan startup. The returned path is stored in
    _GOOGLE_CREDS_TEMP_PATH and used by _get_gsheet() instead of the
    static google_credentials_path setting.

    Local dev: if GOOGLE_CREDENTIALS_JSON is empty, falls back to
    settings.google_credentials_path (file on disk).
    """
    json_str = settings.google_credentials_json
    if not json_str:
        logger.info(
            "GOOGLE_CREDENTIALS_JSON not set — "
            "falling back to file: %s",
            settings.google_credentials_path,
        )
        return None

    try:
        # Validate it's real JSON before writing
        json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error(
            "GOOGLE_CREDENTIALS_JSON is not valid JSON: %s — "
            "Google Sheets will be unavailable",
            exc,
        )
        return None

    try:
        # NamedTemporaryFile with delete=False — persists until process exits
        tf = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            prefix="ragbot_gcreds_",
        )
        tf.write(json_str)
        tf.flush()
        tf.close()
        logger.info("Google credentials written to temp file: %s", tf.name)
        return tf.name
    except Exception as exc:
        logger.error("Failed to write Google credentials temp file: %s", exc)
        return None


# ── [P1-3] Langfuse init ──────────────────────────────────────────────────────
# Phase 1: client initialized here, one startup trace sent.
# Phase 5: per-stage spans wired inside get_chat_response via trace_id injection.
_langfuse = None


def _init_langfuse():
    global _langfuse
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info(
            "Langfuse keys not set — observability disabled. "
            "Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable."
        )
        return

    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        # Send one startup trace so the dashboard shows the service is alive
        trace = _langfuse.trace(
            name="ragbot-startup",
            metadata={"env": settings.env, "phase": "1"},
        )
        trace.update(output={"status": "started"})
        _langfuse.flush()
        logger.info("Langfuse initialized — startup trace sent")
    except Exception as exc:
        logger.warning("Langfuse init failed: %s — observability disabled", exc)
        _langfuse = None


def get_langfuse():
    """Return the Langfuse client or None if not initialized."""
    return _langfuse


# ── Session store: Upstash Redis → legacy Redis → in-memory fallback ──────────
# [P1-5]: Upstash REST client tried first (UPSTASH_REDIS_URL + UPSTASH_REDIS_TOKEN).
# Falls back to legacy redis-py client (REDIS_URL) for local dev.
# Final fallback: in-memory dict (Render single-instance, ephemeral).
_get_history: Callable[[str], list[dict]]
_set_history: Callable[[str, list[dict]], None]
_init_history: Callable[[str], None]

SESSION_BACKEND = "memory"

# Try Upstash REST client first
_upstash_loaded = False
if settings.upstash_redis_url and settings.upstash_redis_token:
    try:
        from upstash_redis import Redis as UpstashRedis

        _upstash = UpstashRedis(
            url=settings.upstash_redis_url,
            token=settings.upstash_redis_token,
        )
        # Smoke test
        _upstash.ping()
        logger.info("Session store: Upstash Redis")

        def _get_history(sid: str) -> list[dict]:
            raw = _upstash.get(f"session:{sid}:history")
            return json.loads(raw) if raw else []

        def _set_history(sid: str, h: list[dict]) -> None:
            _upstash.setex(f"session:{sid}:history", 86400, json.dumps(h))

        def _init_history(sid: str) -> None:
            key = f"session:{sid}:history"
            if not _upstash.exists(key):
                _set_history(sid, [])

        SESSION_BACKEND = "upstash"
        _upstash_loaded = True
    except Exception as _e:
        logger.warning("Upstash Redis unavailable (%s) — trying legacy Redis", _e)

# Try legacy redis-py if Upstash failed
if not _upstash_loaded:
    try:
        import redis as _redis_lib

        _redis_url = os.environ.get("REDIS_URL")
        if not _redis_url:
            raise ImportError("No REDIS_URL")
        _redis_client = _redis_lib.from_url(_redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info("Session store: legacy Redis (%s)", _redis_url)

        def _get_history(sid: str) -> list[dict]:  # type: ignore[misc]
            raw = _redis_client.get(f"h:{sid}")
            return json.loads(raw) if raw else []

        def _set_history(sid: str, h: list[dict]) -> None:  # type: ignore[misc]
            _redis_client.setex(f"h:{sid}", 86400, json.dumps(h))

        def _init_history(sid: str) -> None:  # type: ignore[misc]
            if not _redis_client.exists(f"h:{sid}"):
                _set_history(sid, [])

        SESSION_BACKEND = "redis"
    except Exception as _e:
        logger.warning("Legacy Redis unavailable (%s) — using in-memory store", _e)
        _mem: dict[str, list[dict]] = {}

        def _get_history(sid: str) -> list[dict]:  # type: ignore[misc]
            return _mem.get(sid, [])

        def _set_history(sid: str, h: list[dict]) -> None:  # type: ignore[misc]
            _mem[sid] = h

        def _init_history(sid: str) -> None:  # type: ignore[misc]
            _mem.setdefault(sid, [])

        SESSION_BACKEND = "memory"


# ── Google Sheets helper ──────────────────────────────────────────────────────
_GSHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_gsheet() -> Optional[gspread.Worksheet]:
    """
    Return the first worksheet of the configured Google Sheet.
    Prefers the temp file written from GOOGLE_CREDENTIALS_JSON env var.
    Falls back to google_credentials_path setting for local dev.
    Returns None and logs a warning if unavailable.
    """
    creds_path = _GOOGLE_CREDS_TEMP_PATH or settings.google_credentials_path
    try:
        creds = Credentials.from_service_account_file(creds_path, scopes=_GSHEET_SCOPES)
        gc = gspread.authorize(creds)
        return gc.open(settings.google_sheet_name).sheet1
    except Exception as exc:
        logger.warning("Google Sheets unavailable: %s", exc)
        return None


def _append_lead_to_sheet(lead: LeadCapture) -> bool:
    sheet = _get_gsheet()
    if sheet is None:
        return False
    try:
        sheet.append_row(
            [lead.name, lead.phone, lead.session_id, lead.query or ""],
            value_input_option="USER_ENTERED",
        )
        return True
    except Exception as exc:
        logger.error("Failed to write lead to Google Sheet: %s", exc)
        return False


# ── Email helper (SMTP — replaced with Gmail API in Phase 5) ─────────────────
# [P1-9] KEPT unchanged from v1. Will fail silently on Render (SMTP ports blocked).
# Phase 5 replaces this entire function with Gmail API over HTTPS.
# Do NOT add new code that relies on SMTP after this phase.
def _send_lead_email(lead: LeadCapture) -> bool:
    placeholder_senders = {"yourbot@gmail.com", "", None}
    placeholder_recipients = {"you@yourdomain.com", "", None}
    if (
        settings.email_sender in placeholder_senders
        or settings.email_recipient in placeholder_recipients
        or not settings.email_password
        or settings.email_password.strip() == "xxxx xxxx xxxx xxxx"
    ):
        logger.warning("Email skipped — SMTP credentials not configured")
        return False

    try:
        import socket

        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(8)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"New Lead — {lead.name}"
        msg["From"] = settings.email_sender
        msg["To"] = settings.email_recipient
        html_body = f"""<html><body style="font-family:sans-serif;color:#222">
        <h2 style="color:#b8860b">New Lead Captured</h2>
        <table>
          <tr><td><b>Name</b></td><td>{lead.name}</td></tr>
          <tr><td><b>Phone</b></td><td>{lead.phone}</td></tr>
          <tr><td><b>Session</b></td><td><code>{lead.session_id}</code></td></tr>
          <tr><td><b>Query</b></td><td>{lead.query or "—"}</td></tr>
        </table></body></html>"""
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(settings.email_sender, settings.email_password)
            server.sendmail(
                settings.email_sender, settings.email_recipient, msg.as_string()
            )

        socket.setdefaulttimeout(old_timeout)
        return True

    except Exception as exc:
        logger.error("Failed to send lead email (SMTP): %s", exc)
        return False


# ── [P1-4] Scheduler ──────────────────────────────────────────────────────────
scheduler = BackgroundScheduler()

# Cleanup old ChromaDB collections (was 24h job in v1). Now a no-op in rag.py
# until Phase 2 replaces it with Qdrant collection cleanup.
scheduler.add_job(
    lambda: delete_old_collections(max_age_seconds=86400),
    "interval",
    hours=24,
    id="cleanup_old_collections",
)

# Qdrant keepalive — prevents free cluster auto-suspend (suspends after 7 days idle).
# Fires every 9 minutes so cron-job.org's 10-minute /health ping doesn't race with it.
# ping_qdrant() returns bool, logs outcome, never raises.
scheduler.add_job(
    ping_qdrant,
    "interval",
    minutes=9,
    id="qdrant_keepalive",
)


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _GOOGLE_CREDS_TEMP_PATH

    os.makedirs("uploads", exist_ok=True)
    # vectorstore dir no longer needed (was ChromaDB) — kept for compat during transition
    os.makedirs("vectorstore", exist_ok=True)

    # [P1-2] Write Google credentials from env var
    _GOOGLE_CREDS_TEMP_PATH = _write_google_credentials()

    # [P1-3] Initialize Langfuse
    _init_langfuse()

    scheduler.start()

    logger.info(
        "RAGbot v2 backend started — env=%s session_backend=%s langfuse=%s",
        settings.env,
        SESSION_BACKEND,
        "enabled" if _langfuse else "disabled",
    )

    yield

    # Cleanup temp credentials file on shutdown
    if _GOOGLE_CREDS_TEMP_PATH and os.path.exists(_GOOGLE_CREDS_TEMP_PATH):
        try:
            os.remove(_GOOGLE_CREDS_TEMP_PATH)
        except Exception:
            pass

    scheduler.shutdown(wait=False)

    if _langfuse:
        try:
            _langfuse.flush()
        except Exception:
            pass


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="RAGbot API", version="2.0.0-phase1", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.cors_origin,
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── [P1-6] Health ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health() -> dict:
    """
    Health check pinged by cron-job.org every 10 minutes (keeps Render awake).
    Also pings Qdrant and Upstash so we can verify new plumbing in Phase 1.
    Never raises — always returns 200 with status fields.
    """
    qdrant_ok = ping_qdrant()

    upstash_ok = False
    if SESSION_BACKEND == "upstash":
        try:
            from upstash_redis import Redis as UpstashRedis
            r = UpstashRedis(
                url=settings.upstash_redis_url,
                token=settings.upstash_redis_token,
            )
            r.ping()
            upstash_ok = True
        except Exception as exc:
            logger.warning("Upstash health ping failed: %s", exc)
    else:
        upstash_ok = None  # not configured — don't report false

    return {
        "status": "ok",
        "version": "2.0.0-phase1",
        "env": settings.env,
        "model": settings.groq_model,
        "session_backend": SESSION_BACKEND,
        "qdrant": "ok" if qdrant_ok else "unavailable",
        "upstash": "ok" if upstash_ok else ("not_configured" if upstash_ok is None else "unavailable"),
        "langfuse": "enabled" if _langfuse else "disabled",
    }


# ── Session validate ──────────────────────────────────────────────────────────
@app.get(
    "/session/{session_id}",
    response_model=SessionValidateResponse,
    tags=["session"],
    dependencies=[Depends(require_api_key)],
)
async def validate_session(session_id: str) -> SessionValidateResponse:
    return SessionValidateResponse(
        valid=session_exists(session_id), session_id=session_id
    )


# ── PDF upload ────────────────────────────────────────────────────────────────
@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["document"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("10/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit.",
        )

    if not validate_pdf_bytes(contents):
        raise HTTPException(
            status_code=415, detail="File is not a valid PDF (failed magic-byte check)."
        )

    session_id = str(uuid.uuid4()).replace("-", "")[:16]
    upload_path = os.path.join("uploads", f"{session_id}.pdf")
    with open(upload_path, "wb") as f:
        f.write(contents)

    try:
        page_count, chunk_count = process_pdf(upload_path, session_id)
    except ValueError as exc:
        os.remove(upload_path)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        logger.error("PDF processing failed for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Failed to process PDF.")
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

    _init_history(session_id)

    logger.info(
        "PDF uploaded — session=%s pages=%d chunks=%d",
        session_id,
        page_count,
        chunk_count,
    )
    return UploadResponse(
        success=True,
        message=f"Processed {page_count} page(s) into {chunk_count} searchable chunks.",
        session_id=session_id,
        pages_processed=page_count,
    )


# ── SSE progress stream ───────────────────────────────────────────────────────
_progress_queues: dict[str, asyncio.Queue] = {}


@app.get("/upload/progress/{token}", tags=["document"])
async def upload_progress(token: str) -> StreamingResponse:
    """Server-Sent Events stream for upload progress."""
    q: asyncio.Queue = asyncio.Queue()
    _progress_queues[token] = q

    async def event_stream():
        try:
            while True:
                msg = await asyncio.wait_for(q.get(), timeout=120)
                yield f"data: {msg}\n\n"
                if msg.startswith('{"stage":"done"}') or msg.startswith('{"stage":"error"}'):
                    break
        except asyncio.TimeoutError:
            yield 'data: {"stage":"error","detail":"timeout"}\n\n'
        finally:
            _progress_queues.pop(token, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _push_progress(token: Optional[str], payload: str) -> None:
    if not token:
        return
    q = _progress_queues.get(token)
    if q:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


@app.post(
    "/upload/streamed",
    tags=["document"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("10/minute")
async def upload_pdf_streamed(
    request: Request,
    file: UploadFile = File(...),
    progress_token: Optional[str] = None,
) -> UploadResponse:
    """Upload with real SSE progress events."""

    def emit(stage: str, pct: int, detail: str = "") -> None:
        _push_progress(
            progress_token, json.dumps({"stage": stage, "pct": pct, "detail": detail})
        )

    if file.content_type not in ("application/pdf",):
        emit("error", 0, "Only PDF files are accepted.")
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    emit("reading", 5)
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        emit("error", 0, f"File exceeds {settings.max_file_size_mb} MB.")
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit.",
        )

    emit("validating", 15)
    if not validate_pdf_bytes(contents):
        emit("error", 0, "Not a valid PDF.")
        raise HTTPException(status_code=415, detail="File is not a valid PDF.")

    session_id = str(uuid.uuid4()).replace("-", "")[:16]
    upload_path = os.path.join("uploads", f"{session_id}.pdf")
    with open(upload_path, "wb") as f:
        f.write(contents)

    try:
        emit("chunking", 40)
        page_count, chunk_count = process_pdf(
            upload_path,
            session_id,
            on_chunks_ready=lambda: emit("embedding", 70),
            on_done=lambda: emit("done", 100),
        )
    except ValueError as exc:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        emit("error", 0, str(exc))
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        emit("error", 0, "Processing failed.")
        logger.error("PDF processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process PDF.")
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

    _init_history(session_id)
    logger.info(
        "PDF uploaded (streamed) — session=%s pages=%d chunks=%d",
        session_id,
        page_count,
        chunk_count,
    )
    return UploadResponse(
        success=True,
        message=f"Processed {page_count} page(s) into {chunk_count} searchable chunks.",
        session_id=session_id,
        pages_processed=page_count,
    )


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    Phase 1: session_exists() always returns False (Phase 2 stub).
    All uploads will 404 here until Phase 2 wires real Qdrant persistence.
    /chat/stream added in Phase 4.
    """
    session_id = body.session_id
    if not session_exists(session_id):
        raise HTTPException(
            status_code=404, detail="Session not found. Please upload a PDF first."
        )

    history = _get_history(session_id)

    try:
        reply, lead_triggered, source_chunks = get_chat_response(
            session_id=session_id,
            user_message=body.message,
            conversation_history=history,
        )
    except Exception as exc:
        logger.error("Chat error — session=%s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Failed to generate a response.")

    history.append({"role": "user", "content": body.message})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 20:
        history = history[-20:]
    _set_history(session_id, history)

    return ChatResponse(
        reply=reply, lead_triggered=lead_triggered, source_chunks=source_chunks
    )


# ── Lead capture ──────────────────────────────────────────────────────────────
@app.post("/lead", tags=["lead"], dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def capture_lead(request: Request, lead: LeadCapture) -> dict:
    """
    Capture a lead to Google Sheets and send an email notification.
    Phase 5 adds: lead dedup via Upstash, confidence gate, intent routing
    across three Sheets tabs, Gmail API replacing SMTP.
    """
    sheet_ok = _append_lead_to_sheet(lead)
    email_ok = _send_lead_email(lead)
    logger.info("Lead captured — name=%s session=%s", lead.name, lead.session_id)
    return {
        "success": True,
        "message": "Thank you! A representative will be in touch shortly.",
        "sheet_saved": sheet_ok,
        "email_sent": email_ok,
    }


# ── Clear history ─────────────────────────────────────────────────────────────
@app.delete(
    "/session/{session_id}/history",
    tags=["session"],
    dependencies=[Depends(require_api_key)],
)
async def clear_history(session_id: str) -> dict:
    _set_history(session_id, [])
    semantic_cache.clear(session_id)
    return {"cleared": True, "session_id": session_id}


# ── [P1-7] Cache observability routes ─────────────────────────────────────────
@app.get(
    "/cache/stats/{session_id}",
    tags=["observability"],
    dependencies=[Depends(require_api_key)],
)
async def cache_stats(session_id: str) -> dict:
    """Per-session cache stats. Useful during Phase 3 threshold calibration."""
    return semantic_cache.stats(session_id)


@app.get(
    "/cache/global-stats",
    tags=["observability"],
    dependencies=[Depends(require_api_key)],
)
async def cache_global_stats() -> dict:
    """Global cache stats across all sessions."""
    return semantic_cache.global_stats()