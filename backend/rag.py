"""
rag.py — RAGbot v2, Phase 2
Changes from Phase 1:
  [P2-1] REPLACED: pypdf → pdfplumber + pytesseract OCR fallback
  [P2-2] REPLACED: ChromaDB stub → full Qdrant hybrid search (dense + sparse RRF)
  [P2-3] REPLACED: in-memory BM25 (_bm25_store) → hash-TF sparse vectors in Qdrant
  [P2-4] ADDED: parent/child chunk architecture with full payload schema v2
  [P2-5] ADDED: document dedup via SHA256 + Supabase documents table
  [P2-6] ADDED: section header detection in payload
  [P2-7] KEPT: _embed_documents, _embed_query, validate_pdf_bytes, ping_qdrant
"""

import io
import re
import uuid
import time
import hashlib
import logging
from typing import Optional, Callable

import cohere
import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from supabase import create_client, Client as SupabaseClient

from config import get_settings

settings = get_settings()
logger = logging.getLogger("ragbot.rag")

# ── Embedding constants ───────────────────────────────────────────────────────
_EMBED_MODEL = "embed-english-v3.0"
_EMBED_DIMS = 1024

# ── Chunking constants ────────────────────────────────────────────────────────
_PARENT_SIZE = 1200
_PARENT_OVERLAP = 240
_CHILD_SIZE = 400
_CHILD_OVERLAP = 80

# ── Sparse vector constants ───────────────────────────────────────────────────
_VOCAB_SIZE = 30_000
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "this", "that", "these", "those", "it", "its", "as",
    "from", "not", "no", "if", "so", "we", "you", "he", "she", "they",
    "their", "our", "my", "your", "his", "her", "can", "all", "also",
    "which", "who", "what", "when", "where", "how", "than", "then",
    "about", "up", "out", "into", "some", "any", "each", "both", "such",
    "through", "while", "after", "before", "between", "during", "same",
    "other", "only", "over", "own", "under", "again", "further",
})

# ── Lazy clients ──────────────────────────────────────────────────────────────
_qdrant: Optional[QdrantClient] = None
_cohere_client: Optional[cohere.Client] = None
_supabase: Optional[SupabaseClient] = None


def _get_qdrant() -> Optional[QdrantClient]:
    global _qdrant
    if _qdrant is None and settings.qdrant_url:
        try:
            _qdrant = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=30,
            )
            logger.info("Qdrant client initialized — %s", settings.qdrant_url)
        except Exception as exc:
            logger.error("Qdrant init failed: %s", exc)
    return _qdrant


def _get_cohere() -> Optional[cohere.Client]:
    global _cohere_client
    if _cohere_client is None and settings.cohere_api_key:
        _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
        logger.info("Cohere embed client initialized")
    return _cohere_client


def _get_supabase() -> Optional[SupabaseClient]:
    global _supabase
    if _supabase is None and settings.supabase_url and settings.supabase_key:
        try:
            _supabase = create_client(settings.supabase_url, settings.supabase_key)
            logger.info("Supabase client initialized")
        except Exception as exc:
            logger.error("Supabase init failed: %s", exc)
    return _supabase


# ── Health ────────────────────────────────────────────────────────────────────
def ping_qdrant() -> bool:
    """Ping Qdrant cluster — used by scheduler to prevent free-tier sleep."""
    client = _get_qdrant()
    if client is None:
        return False
    try:
        client.get_collections()
        return True
    except Exception as exc:
        logger.warning("Qdrant ping failed: %s", exc)
        return False


# ── Embeddings ────────────────────────────────────────────────────────────────
_COHERE_BATCH = 96  # Cohere embed batch limit


def _embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document texts with Cohere embed-english-v3.0."""
    client = _get_cohere()
    if client is None:
        logger.warning("Cohere not configured — returning zero embeddings")
        return [[0.0] * _EMBED_DIMS for _ in texts]
    try:
        resp = client.embed(
            texts=texts,
            model=_EMBED_MODEL,
            input_type="search_document",
        )
        return [list(e) for e in resp.embeddings]
    except Exception as exc:
        logger.error("Cohere embed_documents failed: %s", exc)
        return [[0.0] * _EMBED_DIMS for _ in texts]


def _embed_query(query: str) -> list[float]:
    """Embed a single query string with Cohere."""
    client = _get_cohere()
    if client is None:
        return [0.0] * _EMBED_DIMS
    try:
        resp = client.embed(
            texts=[query],
            model=_EMBED_MODEL,
            input_type="search_query",
        )
        return list(resp.embeddings[0])
    except Exception as exc:
        logger.error("Cohere embed_query failed: %s", exc)
        return [0.0] * _EMBED_DIMS


# ── Sparse vectors ─────────────────────────────────────────────────────────────
def _text_to_sparse(text: str) -> SparseVector:
    """
    Convert text to a sparse vector via term-frequency + hash trick.
    Indices are deterministic (hash(term) % VOCAB_SIZE), sorted ascending.
    Used for both indexing and querying — same mapping = consistent dot product.
    """
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS]

    tf: dict[int, float] = {}
    for tok in tokens:
        idx = abs(hash(tok)) % _VOCAB_SIZE
        tf[idx] = tf.get(idx, 0.0) + 1.0

    if not tf:
        # Qdrant requires at least one entry
        return SparseVector(indices=[0], values=[0.0])

    max_freq = max(tf.values())
    indices = sorted(tf.keys())
    values = [tf[i] / max_freq for i in indices]
    return SparseVector(indices=indices, values=values)


# ── PDF validation ─────────────────────────────────────────────────────────────
def validate_pdf_bytes(pdf_bytes: bytes) -> bool:
    """Check PDF magic bytes — fast, no parsing."""
    return len(pdf_bytes) >= 4 and pdf_bytes[:4] == b"%PDF"


# ── PDF text extraction ────────────────────────────────────────────────────────
def extract_text_from_pdf(
    pdf_bytes: bytes,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[str, list[tuple[int, int]]]:
    """
    Extract text from PDF using pdfplumber.
    Falls back to pdf2image + pytesseract if avg chars/page < 50 (scanned PDF).

    Returns:
        full_text          — entire document as one string
        page_boundaries    — list of (char_start, page_number) tuples
                             used by chunker to tag each chunk with its page
    """
    if progress_callback:
        progress_callback("Extracting text from PDF…")

    pages_text: list[str] = []
    total_pages = 0

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
    except Exception as exc:
        logger.error("pdfplumber extraction failed: %s", exc)
        raise

    total_chars = sum(len(t) for t in pages_text)
    avg_chars = total_chars / max(total_pages, 1)

    # OCR fallback for scanned PDFs
    if avg_chars < 50:
        logger.info(
            "Sparse text (%.0f chars/page avg) — triggering OCR fallback", avg_chars
        )
        if progress_callback:
            progress_callback("Sparse text detected — running OCR…")
        pages_text = _ocr_fallback(pdf_bytes, total_pages, progress_callback)

    # Build contiguous text + page boundary map
    parts: list[str] = []
    page_boundaries: list[tuple[int, int]] = []
    char_offset = 0

    for i, text in enumerate(pages_text):
        stripped = text.strip()
        if not stripped:
            continue
        page_boundaries.append((char_offset, i + 1))
        parts.append(stripped)
        char_offset += len(stripped) + 1  # +1 for the \n separator

    full_text = "\n".join(parts)
    logger.info(
        "PDF extracted — %d pages | %d chars | %.0f avg chars/page",
        total_pages, len(full_text), avg_chars,
    )
    return full_text, page_boundaries


def _ocr_fallback(
    pdf_bytes: bytes,
    total_pages: int,
    progress_callback: Optional[Callable] = None,
) -> list[str]:
    """OCR via pdf2image + pytesseract. Gracefully degrades if deps missing."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(pdf_bytes, dpi=200)
        pages: list[str] = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang="eng")
            pages.append(text)
            logger.debug("OCR page %d/%d — %d chars", i + 1, total_pages, len(text))

        logger.info("OCR complete — %d pages", len(pages))
        return pages

    except ImportError as exc:
        logger.warning("OCR deps missing (%s) — returning empty pages", exc)
        return [""] * total_pages
    except Exception as exc:
        logger.error("OCR failed: %s — returning empty pages", exc)
        return [""] * total_pages


# ── Section header detection ──────────────────────────────────────────────────
def _detect_header(line: str) -> Optional[str]:
    """
    Heuristic: short (<= 80 chars), no trailing period, either ALL CAPS or
    Title Case. Returns the cleaned header string or None.
    """
    s = line.strip()
    if not s or len(s) > 80 or s.endswith("."):
        return None
    if s.isupper() and len(s) > 3:
        return s
    if s.istitle() and len(s.split()) <= 10:
        return s
    return None


def _nearest_header(char_start: int, full_text: str) -> str:
    """Scan backwards up to 600 chars before chunk start for a section header."""
    snippet = full_text[max(0, char_start - 600) : char_start]
    for line in reversed(snippet.splitlines()):
        h = _detect_header(line)
        if h:
            return h
    return ""


def _char_to_page(char_start: int, page_boundaries: list[tuple[int, int]]) -> int:
    """Binary-search page_boundaries for the page containing char_start."""
    page = 1
    for boundary_start, page_num in page_boundaries:
        if char_start >= boundary_start:
            page = page_num
        else:
            break
    return page


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text_hierarchical(
    text: str,
    metadata: dict,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """
    Build parent (1200/240) and child (400/80) chunks from full document text.

    Each chunk dict carries the complete payload schema v2:
        point_id, doc_id, chunk_type, parent_id, page_number,
        section_header, text, char_start, char_end, schema_version

    point_id is a UUID string used as the Qdrant point ID.
    parent_id on children references the parent's point_id.
    """
    doc_id = metadata.get("doc_id", str(uuid.uuid4()))
    page_boundaries: list[tuple[int, int]] = metadata.get("page_boundaries", [])

    if progress_callback:
        progress_callback("Building hierarchical chunks…")

    chunks: list[dict] = []
    text_len = len(text)

    p_start = 0
    while p_start < text_len:
        p_end = min(p_start + _PARENT_SIZE, text_len)
        parent_text = text[p_start:p_end].strip()

        if len(parent_text) >= 50:
            parent_id = str(uuid.uuid4())
            page_num = _char_to_page(p_start, page_boundaries)
            section = _nearest_header(p_start, text)

            chunks.append({
                "point_id": parent_id,
                "doc_id": doc_id,
                "chunk_type": "parent",
                "parent_id": None,
                "page_number": page_num,
                "section_header": section,
                "text": parent_text,
                "char_start": p_start,
                "char_end": p_end,
                "schema_version": settings.qdrant_payload_schema_version,
            })

            # Children carved from parent text
            c_start = 0
            p_len = len(parent_text)
            while c_start < p_len:
                c_end = min(c_start + _CHILD_SIZE, p_len)
                child_text = parent_text[c_start:c_end].strip()

                if len(child_text) >= 30:
                    chunks.append({
                        "point_id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "page_number": page_num,
                        "section_header": section,
                        "text": child_text,
                        "char_start": p_start + c_start,
                        "char_end": p_start + c_end,
                        "schema_version": settings.qdrant_payload_schema_version,
                    })

                c_start += _CHILD_SIZE - _CHILD_OVERLAP

        p_start += _PARENT_SIZE - _PARENT_OVERLAP

    parents = sum(1 for c in chunks if c["chunk_type"] == "parent")
    children = len(chunks) - parents
    logger.info("Chunked — %d parents | %d children", parents, children)
    return chunks


# ── Qdrant collection helpers ─────────────────────────────────────────────────
def _col(session_id: str) -> str:
    prefix = settings.qdrant_collection_prefix or "ragbot"
    return f"{prefix}_{session_id}"


def _ensure_collection(session_id: str) -> None:
    """Create Qdrant collection with dense + sparse named vectors if absent."""
    client = _get_qdrant()
    if client is None:
        raise RuntimeError("Qdrant not configured")

    col = _col(session_id)
    if client.collection_exists(col):
        return

    client.create_collection(
        collection_name=col,
        vectors_config={
            "dense": VectorParams(size=_EMBED_DIMS, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    logger.info("Created Qdrant collection: %s", col)


# ── Store ─────────────────────────────────────────────────────────────────────
_UPSERT_BATCH = 50


def store_chunks_hierarchical(
    session_id: str,
    chunks: list[dict],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Embed all chunks in batches, build Qdrant PointStructs with dense + sparse
    vectors, upsert in batches of 50.
    """
    client = _get_qdrant()
    if client is None:
        logger.error("Qdrant not configured — store skipped")
        return

    _ensure_collection(session_id)
    col = _col(session_id)
    texts = [c["text"] for c in chunks]

    if progress_callback:
        progress_callback(f"Embedding {len(chunks)} chunks…")

    # Batch-embed (Cohere allows up to 96 per call)
    all_dense: list[list[float]] = []
    for i in range(0, len(texts), _COHERE_BATCH):
        all_dense.extend(_embed_documents(texts[i : i + _COHERE_BATCH]))

    if progress_callback:
        progress_callback("Storing vectors in Qdrant…")

    # Build points
    points: list[PointStruct] = []
    for chunk, dense_vec in zip(chunks, all_dense):
        payload = {k: v for k, v in chunk.items() if k != "point_id"}
        points.append(
            PointStruct(
                id=chunk["point_id"],
                vector={
                    "dense": dense_vec,
                    "sparse": _text_to_sparse(chunk["text"]),
                },
                payload=payload,
            )
        )

    # Batch upsert
    total_batches = (len(points) + _UPSERT_BATCH - 1) // _UPSERT_BATCH
    for i in range(0, len(points), _UPSERT_BATCH):
        client.upsert(collection_name=col, points=points[i : i + _UPSERT_BATCH])
        logger.debug(
            "Upserted batch %d/%d", i // _UPSERT_BATCH + 1, total_batches
        )

    logger.info("Stored %d points in %s", len(points), col)


# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve_chunks_hybrid(
    session_id: str,
    query: str,
    hyde_query: Optional[str] = None,
    top_k: Optional[int] = None,
) -> list[str]:
    """
    Hybrid retrieval pipeline:
      1. Embed query (dense, using HyDE passage when provided)
      2. Convert query to sparse vector (always original query)
      3. Prefetch child chunks via dense + sparse separately
      4. Fuse with Qdrant server-side RRF
      5. Collect unique parent_ids from top results
      6. Fetch parent points by ID — return their texts

    Returns list[str] of parent chunk texts, ready for Cohere reranker.
    """
    client = _get_qdrant()
    if client is None:
        logger.error("Qdrant not configured")
        return []

    col = _col(session_id)
    if not client.collection_exists(col):
        logger.warning("Collection not found for session=%s", session_id)
        return []

    if top_k is None:
        top_k = settings.top_k_results

    # Embed — use HyDE passage for dense, raw query for sparse
    dense_vec = _embed_query(hyde_query if hyde_query else query)
    sparse_vec = _text_to_sparse(query)

    child_filter = Filter(
        must=[FieldCondition(key="chunk_type", match=MatchValue(value="child"))]
    )

    try:
        results = client.query_points(
            collection_name=col,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    filter=child_filter,
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    filter=child_filter,
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        hits = results.points
    except Exception as exc:
        logger.error("Qdrant hybrid search failed: %s", exc)
        return []

    if not hits:
        logger.info("No child hits for session=%s", session_id)
        return []

    # Collect ordered unique parent IDs
    parent_ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        pid = hit.payload.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            parent_ids.append(pid)

    if not parent_ids:
        # Degenerate: return child texts as fallback
        logger.warning("No parent IDs in hits — returning child texts")
        return [h.payload.get("text", "") for h in hits if h.payload.get("text")]

    # Fetch parent points by ID (direct lookup, no search)
    try:
        parent_points = client.retrieve(
            collection_name=col,
            ids=parent_ids,
            with_payload=True,
        )
    except Exception as exc:
        logger.error("Parent retrieve failed: %s", exc)
        return [h.payload.get("text", "") for h in hits]

    parent_map = {str(p.id): p.payload.get("text", "") for p in parent_points}
    texts = [parent_map[pid] for pid in parent_ids if parent_map.get(pid)]

    logger.info(
        "Hybrid retrieve: %d children → %d parents | session=%s",
        len(hits), len(texts), session_id,
    )
    return texts


# ── Session management ────────────────────────────────────────────────────────
def session_exists(session_id: str) -> bool:
    """True if a Qdrant collection exists for this session."""
    client = _get_qdrant()
    if client is None:
        return False
    try:
        return client.collection_exists(_col(session_id))
    except Exception as exc:
        logger.error("session_exists failed: %s", exc)
        return False


def delete_old_collections(session_id: str) -> None:
    """
    Delete Qdrant collection for a session.
    Called when a new PDF is uploaded for the same session (re-ingest).
    """
    client = _get_qdrant()
    if client is None:
        return
    col = _col(session_id)
    try:
        if client.collection_exists(col):
            client.delete_collection(col)
            logger.info("Deleted Qdrant collection: %s", col)
    except Exception as exc:
        logger.error("delete_old_collections failed for %s: %s", col, exc)


# ── Document dedup via Supabase ────────────────────────────────────────────────
def _sha256_bytes(data: bytes) -> str:
    """SHA256 hex digest — used as document fingerprint."""
    return hashlib.sha256(data).hexdigest()


def document_exists(session_id: str, doc_hash: str) -> bool:
    """
    True if this exact document (session_id + SHA256 hash) is already in
    Supabase documents table. Skips dedup gracefully if Supabase not configured.
    """
    sb = _get_supabase()
    if sb is None:
        logger.warning("Supabase not configured — dedup check skipped")
        return False
    try:
        result = (
            sb.table("documents")
            .select("session_id")
            .eq("session_id", session_id)
            .eq("doc_hash", doc_hash)
            .limit(1)
            .execute()
        )
        return len(result.data) > 0
    except Exception as exc:
        logger.error("document_exists Supabase query failed: %s", exc)
        return False


def register_document(
    session_id: str,
    doc_hash: str,
    doc_name: str,
    chunk_count: int,
) -> None:
    """Write ingested document metadata to Supabase documents table."""
    sb = _get_supabase()
    if sb is None:
        logger.warning("Supabase not configured — document registration skipped")
        return
    try:
        sb.table("documents").upsert({
            "session_id": session_id,
            "doc_hash": doc_hash,
            "doc_name": doc_name,
            "chunk_count": chunk_count,
        }).execute()
        logger.info(
            "Registered document — session=%s | hash=%s… | chunks=%d",
            session_id, doc_hash[:12], chunk_count,
        )
    except Exception as exc:
        logger.error("register_document Supabase write failed: %s", exc)


# ── Main ingestion pipeline ────────────────────────────────────────────────────
def process_pdf(
    pdf_bytes: bytes,
    session_id: str,
    doc_name: str = "document.pdf",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Full ingestion pipeline:
    validate → extract → chunk → embed → store

    Dedup and Supabase registration are handled in main.py /upload route
    so this function stays pure (no side effects beyond Qdrant).

    Returns:
        {"pages": int, "chunks": int, "parents": int, "children": int}
    """
    if not validate_pdf_bytes(pdf_bytes):
        raise ValueError("File is not a valid PDF")

    full_text, page_boundaries = extract_text_from_pdf(pdf_bytes, progress_callback)

    if not full_text.strip():
        raise ValueError("No text could be extracted from this PDF")

    doc_id = str(uuid.uuid4())
    metadata = {"doc_id": doc_id, "page_boundaries": page_boundaries}

    chunks = chunk_text_hierarchical(full_text, metadata, progress_callback)
    if not chunks:
        raise ValueError("PDF produced no usable chunks")

    store_chunks_hierarchical(session_id, chunks, progress_callback)

    parents = [c for c in chunks if c["chunk_type"] == "parent"]
    children = [c for c in chunks if c["chunk_type"] == "child"]

    return {
        "pages": len(page_boundaries) or 1,
        "chunks": len(chunks),
        "parents": len(parents),
        "children": len(children),
    }