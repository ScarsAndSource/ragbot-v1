import os
import logging
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import get_settings
from typing import Callable, Optional

settings = get_settings()

# FIX 14: Suppress ChromaDB telemetry noise before client is created
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

logger = logging.getLogger("ragbot.rag")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(
    path="vectorstore/",
    settings=ChromaSettings(anonymized_telemetry=False)
)

# FIX 10: Valid PDF magic bytes — %PDF-
_PDF_MAGIC = b"%PDF-"

def validate_pdf_bytes(data: bytes) -> bool:
    """
    FIX 10: Check actual file magic bytes, not just MIME type.
    A file renamed to .pdf with wrong content will fail here before
    ever reaching pypdf (which raises confusing internal errors).
    """
    return data[:5] == _PDF_MAGIC


def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """
    Extract all text from PDF. Returns (text, page_count).
    FIX 7: Pre-validate for image-only content before full processing.
    """
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

    # FIX 7: If every single page yielded no text, it's image-only.
    # Raise a clear, user-facing error immediately (not after minutes of work).
    if image_only_pages == page_count:
        raise ValueError(
            "This PDF appears to be a scanned image document with no extractable text. "
            "Please use a text-based PDF or run OCR on it first."
        )

    # Partial warning: some pages are image-only
    if image_only_pages > 0:
        logger.warning(
            "PDF has %d/%d image-only pages — those will be skipped.",
            image_only_pages, page_count,
        )

    return full_text, page_count


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def store_chunks(session_id: str, chunks: list[str]) -> str:
    collection_name = f"session_{session_id}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection_name


def retrieve_chunks(session_id: str, query: str, top_k: int = None) -> list[str]:
    if top_k is None:
        top_k = settings.top_k_results
    collection_name = f"session_{session_id}"
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        return []
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )
    return results["documents"][0] if results["documents"] else []


def session_exists(session_id: str) -> bool:
    try:
        chroma_client.get_collection(f"session_{session_id}")
        return True
    except Exception:
        return False


def process_pdf(
    file_path: str,
    session_id: str,
    on_chunks_ready: Optional[Callable] = None,
    on_done: Optional[Callable] = None,
) -> tuple[int, int]:
    """
    Full pipeline: extract → chunk → store.
    FIX 7: image-only detection happens inside extract_text_from_pdf.
    FIX 12: optional callbacks for real progress reporting over SSE.
    Returns (page_count, chunk_count).
    """
    text, page_count = extract_text_from_pdf(file_path)

    if not text.strip():
        raise ValueError("PDF appears to be empty or image-only (no extractable text).")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Could not extract meaningful text chunks from this PDF.")

    if on_chunks_ready:
        try:
            on_chunks_ready()
        except Exception:
            pass

    store_chunks(session_id, chunks)

    if on_done:
        try:
            on_done()
        except Exception:
            pass

    return page_count, len(chunks)


def delete_old_collections(max_age_seconds: int = 86400):
    """
    Background cleanup stub.
    Full implementation requires tracking creation timestamps;
    a simple approach is to keep a Redis ZSET of (session_id, created_at)
    and delete collections where now - created_at > max_age_seconds.
    Left as a pass for now — Render free tier wipes the filesystem anyway.
    """
    pass
