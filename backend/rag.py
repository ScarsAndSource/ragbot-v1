"""
rag.py — RAGbot v2, Phase 1

Changes from v1:
  [P1-R1] REMOVED: chromadb import, chroma_client initialization,
                    ChromaDB telemetry suppression, _bm25_store dict.
                    ChromaDB is gone entirely from this file.
  [P1-R2] REMOVED: rank_bm25 import from top-level. Will be re-added in
                    Phase 2 for sparse vector generation (BM25 → Qdrant SparseVector).
                    Removed now so requirements.txt drop of chromadb doesn't
                    leave dead imports that confuse the Phase 2 diff.
  [P1-R3] ADDED:   qdrant_client initialization (QdrantClient).
                    Phase 1 only calls ping_qdrant() — no collection ops yet.
  [P1-R4] ADDED:   ping_qdrant() — health-check used by:
                      - /health route in main.py
                      - Qdrant keepalive scheduler job (every 9 min) in main.py
                    Returns bool, never raises, always logs outcome.
  [P1-R5] KEPT:    _embed_documents, _embed_query — untouched, no ChromaDB dependency.
  [P1-R6] KEPT:    validate_pdf_bytes — untouched.
  [P1-R7] KEPT:    extract_text_from_pdf — untouched (pypdf still in requirements
                    until Phase 2 replaces it with pdfplumber + OCR fallback).
  [P1-R8] KEPT:    chunk_text_hierarchical — untouched. Phase 2 changes chunk sizes
                    (parent 1200/240, child 400/80) and adds payload metadata schema.
  [P1-R9] STUBBED: store_chunks_hierarchical — logs and returns without writing.
                    Phase 2 replaces with Qdrant dense+sparse upsert.
  [P1-R10] STUBBED: retrieve_chunks_hybrid — returns [] with a warning.
                    Phase 2 replaces with Qdrant hybrid search + RRF.
  [P1-R11] STUBBED: session_exists — always returns False.
                    Phase 2 returns True after Qdrant collection confirmed present.
                    CONSEQUENCE: /chat 404s for all sessions until Phase 2.
                    /upload still runs extract+chunk but does not persist.
                    This is intentional — Phase 1 proves infrastructure, not data.
  [P1-R12] KEPT:   process_pdf — runs extract+chunk pipeline end-to-end.
                    Store step is stubbed so data is not persisted.
  [P1-R13] KEPT:   delete_old_collections — no-op stub (scheduler job in main.py).
                    Phase 2 replaces with Qdrant collection TTL cleanup.

Phases that will touch this file next:
  Phase 2: FULL REWRITE of everything below _embed_query.
            Qdrant dense+sparse upsert, pdfplumber, OCR fallback,
            SHA256 dedup, Supabase document registration, section headers.
"""

import logging
from typing import Callable, Optional

import cohere
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from config import get_settings

settings = get_settings()
logger = logging.getLogger("ragbot.rag")

# ── Cohere client ─────────────────────────────────────────────────────────────
co = cohere.Client(settings.cohere_api_key)

# ── [P1-R3] Qdrant client ────────────────────────────────────────────────────
# Initialized at module load. If QDRANT_URL is empty (local dev without Qdrant),
# the client init still succeeds — it will just fail on actual operations.
# ping_qdrant() handles the empty-URL case gracefully.
_qdrant: Optional[QdrantClient] = None

if settings.qdrant_url:
    try:
        _qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=10,
        )
        logger.info("Qdrant client initialized — url=%s", settings.qdrant_url)
    except Exception as _e:
        logger.warning("Qdrant client init failed: %s — storage unavailable", _e)
        _qdrant = None
else:
    logger.info(
        "QDRANT_URL not set — Qdrant disabled. "
        "Set QDRANT_URL and QDRANT_API_KEY to enable."
    )

# ── Cohere embedding constants ────────────────────────────────────────────────
_COHERE_EMBED_MODEL = "embed-english-v3.0"
_COHERE_BATCH_SIZE = 96


# ── [P1-R5] Embedding helpers — unchanged from v1 ───────────────────────────
def _embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document chunks via Cohere API.
    Uses search_document input type for indexing.
    Batches automatically for lists larger than 96.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _COHERE_BATCH_SIZE):
        batch = texts[i : i + _COHERE_BATCH_SIZE]
        response = co.embed(
            texts=batch,
            model=_COHERE_EMBED_MODEL,
            input_type="search_document",
        )
        all_embeddings.extend([[float(v) for v in row] for row in response.embeddings])
    return all_embeddings


def _embed_query(query: str) -> list[float]:
    """
    Embed a single query string via Cohere API.
    Uses search_query input type — distinct from search_document,
    improves retrieval quality on embed-english-v3.0.
    """
    response = co.embed(
        texts=[query],
        model=_COHERE_EMBED_MODEL,
        input_type="search_query",
    )
    return [float(v) for v in list(response.embeddings)[0]]


# ── [P1-R6] PDF validation — unchanged from v1 ──────────────────────────────
_PDF_MAGIC = b"%PDF-"


def validate_pdf_bytes(data: bytes) -> bool:
    return data[:5] == _PDF_MAGIC


# ── [P1-R7] PDF text extraction — unchanged from v1 ─────────────────────────
# Phase 2 replaces pypdf with pdfplumber + pytesseract OCR fallback.
# Function signature stays identical so process_pdf needs no changes.
def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    reader = PdfReader(file_path)
    page_count = len(reader.pages)
    pages = []
    image_only_pages = 0

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
        else:
            image_only_pages += 1

    full_text = "\n\n".join(pages)

    if image_only_pages == page_count:
        raise ValueError(
            "This PDF appears to be a scanned image document with no extractable text. "
            "Please use a text-based PDF or run OCR on it first."
        )

    if image_only_pages > 0:
        logger.warning(
            "PDF has %d/%d image-only pages — those will be skipped.",
            image_only_pages,
            page_count,
        )

    return full_text, page_count


# ── [P1-R8] Hierarchical chunking — unchanged from v1 ───────────────────────
# Phase 2 changes chunk sizes: parent 1200/240, child 400/80.
# Phase 2 also adds page_number and section_header to payload metadata.
# Function signature and return type stay identical.
def chunk_text_hierarchical(text: str) -> tuple[list[str], list[str], list[int]]:
    """
    Split text into two levels of chunks.

    Parent chunks (1000 chars, 100 overlap) — sent to the LLM.
    Child chunks  (350 chars, 40 overlap)  — used for retrieval.

    Each child knows its parent index via child_to_parent.

    Returns:
        child_chunks    — small retrieval-target strings
        parent_chunks   — large LLM-context strings
        child_to_parent — child_to_parent[i] = parent index for child i
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )

    parent_chunks = [
        c.strip() for c in parent_splitter.split_text(text) if len(c.strip()) > 80
    ]

    if not parent_chunks:
        raise ValueError("Could not extract any parent chunks from this document.")

    child_chunks: list[str] = []
    child_to_parent: list[int] = []

    for parent_idx, parent in enumerate(parent_chunks):
        children = child_splitter.split_text(parent)
        children = [c.strip() for c in children if len(c.strip()) > 40]

        if not children:
            children = [parent]

        child_chunks.extend(children)
        child_to_parent.extend([parent_idx] * len(children))

    logger.info(
        "Chunking complete — parents=%d | children=%d | avg children/parent=%.1f",
        len(parent_chunks),
        len(child_chunks),
        len(child_chunks) / max(len(parent_chunks), 1),
    )

    return child_chunks, parent_chunks, child_to_parent


# ── [P1-R4] Qdrant health check ──────────────────────────────────────────────
def ping_qdrant() -> bool:
    """
    Ping the Qdrant cluster to verify it is reachable and to prevent
    free-tier auto-suspension (Qdrant suspends after 7 days of no traffic).

    Called by:
      - /health route — returns result in health payload
      - Scheduler keepalive job — fires every 9 minutes

    Returns True if cluster responds, False otherwise.
    Never raises — all exceptions are caught and logged.
    """
    if _qdrant is None:
        logger.debug("ping_qdrant: client not initialized — skipping")
        return False

    try:
        info = _qdrant.get_collections()
        logger.debug(
            "Qdrant ping ok — %d collection(s) visible",
            len(info.collections),
        )
        return True
    except UnexpectedResponse as exc:
        logger.warning("Qdrant ping failed (HTTP %s): %s", exc.status_code, exc)
        return False
    except Exception as exc:
        logger.warning("Qdrant ping failed: %s", exc)
        return False


# ── [P1-R9] Storage stub ─────────────────────────────────────────────────────
def store_chunks_hierarchical(
    session_id: str,
    child_chunks: list[str],
    parent_chunks: list[str],
    child_to_parent: list[int],
) -> None:
    """
    STUB — Phase 1 only.

    Phase 2 replaces this with Qdrant dense+sparse upsert:
      - One collection per session: {QDRANT_COLLECTION_PREFIX}_{session_id}
      - Dense vector: Cohere embed-english-v3.0
      - Sparse vector: BM25 term weights → Qdrant SparseVector
      - Payload schema v2: doc_id, chunk_type, parent_id, page_number,
                           section_header, text, char_start, char_end
    """
    logger.warning(
        "store_chunks_hierarchical is a Phase 1 stub — "
        "session=%s | parents=%d | children=%d — data NOT persisted. "
        "Phase 2 wires Qdrant upsert.",
        session_id,
        len(parent_chunks),
        len(child_chunks),
    )


# ── [P1-R10] Retrieval stub ───────────────────────────────────────────────────
def retrieve_chunks_hybrid(
    session_id: str,
    query: str,
    top_k: Optional[int] = None,
    hyde_query: Optional[str] = None,
) -> list[str]:
    """
    STUB — Phase 1 only. Returns empty list.

    Phase 2 replaces this with Qdrant native hybrid search:
      - Prefetch dense + sparse results
      - RRF fusion server-side
      - Parent resolution by payload filter
    """
    logger.warning(
        "retrieve_chunks_hybrid is a Phase 1 stub — "
        "session=%s — returning []. Phase 2 wires Qdrant hybrid search.",
        session_id,
    )
    return []


# ── [P1-R11] Session existence stub ──────────────────────────────────────────
def session_exists(session_id: str) -> bool:
    """
    STUB — Phase 1 only. Always returns False.

    CONSEQUENCE: /chat 404s for every session until Phase 2.
    This is intentional — Phase 1 proves infrastructure (Qdrant ping,
    Langfuse trace, Upstash session store), not data persistence.

    Phase 2 replaces this with a Qdrant collection_exists() check:
        _qdrant.collection_exists(f"{prefix}_{session_id}")
    """
    return False


# ── [P1-R12] Full PDF pipeline ───────────────────────────────────────────────
def process_pdf(
    file_path: str,
    session_id: str,
    on_chunks_ready: Optional[Callable] = None,
    on_done: Optional[Callable] = None,
) -> tuple[int, int]:
    """
    Full ingestion pipeline: extract → chunk → store (stub in Phase 1).

    extract_text_from_pdf and chunk_text_hierarchical run normally.
    store_chunks_hierarchical is stubbed — data is NOT persisted until Phase 2.

    Returns (page_count, child_chunk_count) — accurate values even in Phase 1
    so the /upload response shows real numbers.
    """
    text, page_count = extract_text_from_pdf(file_path)

    if not text.strip():
        raise ValueError("PDF appears to be empty or image-only (no extractable text).")

    child_chunks, parent_chunks, child_to_parent = chunk_text_hierarchical(text)

    if not child_chunks:
        raise ValueError("Could not extract meaningful chunks from this PDF.")

    if on_chunks_ready:
        try:
            on_chunks_ready()
        except Exception:
            pass

    # Phase 1: store is a stub — logs warning, does not persist
    store_chunks_hierarchical(session_id, child_chunks, parent_chunks, child_to_parent)

    if on_done:
        try:
            on_done()
        except Exception:
            pass

    return page_count, len(child_chunks)


# ── [P1-R13] Cleanup stub ────────────────────────────────────────────────────
def delete_old_collections(max_age_seconds: int = 86400) -> None:
    """
    No-op stub. Called by the 24-hour scheduler job in main.py.
    Phase 2 replaces with Qdrant collection TTL cleanup based on
    collection metadata timestamp.
    """
    pass