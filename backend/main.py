import os
import uuid
import json
import smtplib
import logging
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import asynccontextmanager
from typing import Callable, Optional
from cache import semantic_cache

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from apscheduler.schedulers.background import BackgroundScheduler

import gspread
from google.oauth2.service_account import Credentials

from config import get_settings
from models import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
    LeadCapture,
    SessionValidateResponse,
)
from rag import process_pdf, session_exists, delete_old_collections, validate_pdf_bytes
from chat import get_chat_response

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("ragbot")

# FIX 14: Suppress noisy ChromaDB telemetry errors
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

settings = get_settings()

# ─── Rate limiter ────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ─── API key auth ────────────────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> None:
    expected = settings.ragbot_api_key
    if not expected:
        return
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ─── Session store (Redis → in-memory fallback) ───────────────────────────────
# Forward-declare so Pylance resolves the name regardless of which branch runs
_get_history: Callable[[str], list[dict]]
_set_history: Callable[[str, list[dict]], None]
_init_history: Callable[[str], None]

try:
    import redis as _redis_lib
    import json as _json

    _redis_url = os.environ.get("REDIS_URL")
    if not _redis_url:
        raise ImportError("No REDIS_URL")
    _redis_client = _redis_lib.from_url(_redis_url, decode_responses=True)
    _redis_client.ping()
    logger.info("Session store: Redis (%s)", _redis_url)

    def _get_history(sid: str) -> list[dict]:
        raw = _redis_client.get(f"h:{sid}")
        return _json.loads(raw) if raw else []

    def _set_history(sid: str, h: list[dict]) -> None:
        _redis_client.setex(f"h:{sid}", 86400, _json.dumps(h))

    def _init_history(sid: str) -> None:
        if not _redis_client.exists(f"h:{sid}"):
            _set_history(sid, [])

    SESSION_BACKEND = "redis"

except Exception as _e:
    logger.warning("Redis unavailable (%s) — in-memory session store", _e)
    _mem: dict[str, list[dict]] = {}

    def _get_history(sid: str) -> list[dict]:  # pyright: ignore[reportRedeclaration]
        return _mem.get(sid, [])

    def _set_history(
        sid: str, h: list[dict]
    ) -> None:  # pyright: ignore[reportRedeclaration]
        _mem[sid] = h

    def _init_history(sid: str) -> None:  # pyright: ignore[reportRedeclaration]
        _mem.setdefault(sid, [])

    SESSION_BACKEND = "memory"


# ─── Google Sheets helper ─────────────────────────────────────────────────────
def _get_gsheet() -> Optional[gspread.Worksheet]:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        creds = Credentials.from_service_account_file(
            settings.google_credentials_path, scopes=scopes
        )
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


# ─── Email helper ─────────────────────────────────────────────────────────────
def _send_lead_email(lead: LeadCapture) -> bool:
    # Skip silently if credentials are missing or still placeholders
    placeholder_senders = {"yourbot@gmail.com", "", None}
    placeholder_recipients = {"you@yourdomain.com", "", None}
    if (
        settings.email_sender in placeholder_senders
        or settings.email_recipient in placeholder_recipients
        or not settings.email_password
        or settings.email_password.strip() == "xxxx xxxx xxxx xxxx"
    ):
        logger.warning("Email skipped — credentials not configured")
        return False

    try:
        import socket

        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(8)  # never hang more than 8 seconds

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
        logger.error("Failed to send lead email: %s", exc)
        return False


# ─── Scheduler ───────────────────────────────────────────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(
    lambda: delete_old_collections(max_age_seconds=86400), "interval", hours=24
)


# ─── App lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)
    scheduler.start()
    logger.info(
        "RAGbot backend started — env=%s session_backend=%s",
        settings.env,
        SESSION_BACKEND,
    )
    yield
    scheduler.shutdown(wait=False)


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="RAGbot API", version="1.1.0", lifespan=lifespan)
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


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.env, "model": settings.groq_model}


# ─── Session validate ─────────────────────────────────────────────────────────
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


# ─── PDF upload (with SSE progress stream) ────────────────────────────────────
# FIX 6: Warn if session already exists (don't silently replace it)
# FIX 10: Validate actual PDF magic bytes before handing to pypdf
# FIX 12: Real SSE progress events so the frontend can show true status


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["document"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("10/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    # FIX 10: MIME check (defence layer 1 — spoofable, but cheap)
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit.",
        )

    # FIX 10: Validate magic bytes (defence layer 2 — not spoofable by rename)
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


# FIX 12: SSE progress endpoint — call this from frontend during upload
# The frontend opens an EventSource to /upload/progress/{token}, then POSTs
# to /upload concurrently. The backend pushes real stage events over SSE.
_progress_queues: dict[str, asyncio.Queue[str]] = {}


@app.get("/upload/progress/{token}", tags=["document"])
async def upload_progress(token: str) -> StreamingResponse:
    """Server-Sent Events stream for upload progress."""
    q: asyncio.Queue[str] = asyncio.Queue()
    _progress_queues[token] = q

    async def event_stream():
        try:
            while True:
                msg = await asyncio.wait_for(q.get(), timeout=120)
                yield f"data: {msg}\n\n"
                if msg.startswith('{"stage":"done"}') or msg.startswith(
                    '{"stage":"error"}'
                ):
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


def push_progress(token: Optional[str], payload: str) -> None:
    """Push a JSON progress event to a waiting SSE client."""
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
    """
    FIX 12: Upload variant that emits real progress events over SSE.
    Frontend opens /upload/progress/<token> first, then POSTs here with
    ?progress_token=<token>.  Events: reading → validating → chunking → embedding → done
    """

    def emit(stage: str, pct: int, detail: str = "") -> None:
        push_progress(
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
        # process_pdf internally calls emit callbacks if token provided
        page_count, chunk_count = process_pdf(
            upload_path,
            session_id,
            on_chunks_ready=lambda: emit("embedding", 70),
            on_done=lambda: emit("done", 100),
        )
    except ValueError as exc:
        os.remove(upload_path)
        emit("error", 0, str(exc))
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
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


# ─── Chat ─────────────────────────────────────────────────────────────────────
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
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


# ─── Lead capture ─────────────────────────────────────────────────────────────
@app.post("/lead", tags=["lead"], dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def capture_lead(request: Request, lead: LeadCapture) -> dict[str, object]:
    sheet_ok = _append_lead_to_sheet(lead)
    email_ok = _send_lead_email(lead)
    logger.info("Lead captured — name=%s session=%s", lead.name, lead.session_id)
    return {
        "success": True,
        "message": "Thank you! A representative will be in touch shortly.",
        "sheet_saved": sheet_ok,
        "email_sent": email_ok,
    }


# ─── Clear history ────────────────────────────────────────────────────────────
@app.delete(
    "/session/{session_id}/history",
    tags=["session"],
    dependencies=[Depends(require_api_key)],
)
async def clear_history(session_id: str):
    _set_history(session_id, [])
    semantic_cache.clear(session_id)
    return {"cleared": True, "session_id": session_id}
