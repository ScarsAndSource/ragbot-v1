import logging

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import get_settings
from typing import Callable, Optional

settings = get_settings()

# ── Suppress ChromaDB telemetry noise ─────────────────────────────────────────
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

logger = logging.getLogger("ragbot.rag")

# ── Models and clients ────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(
    path="vectorstore/", settings=ChromaSettings(anonymized_telemetry=False)
)

# ── BM25 in-memory store ───────────────────────────────────────────────────────
# Maps session_id → (BM25Okapi index, list of raw chunk strings)
# Lives in memory — same lifecycle as ChromaDB on Render free tier.
# If server restarts, both ChromaDB (free tier) and BM25 are wiped together.
# retrieve_chunks_hybrid() has a fallback to pure vector if BM25 is missing.
_bm25_store: dict[str, tuple[BM25Okapi, list[str]]] = {}

# ── PDF validation ────────────────────────────────────────────────────────────
_PDF_MAGIC = b"%PDF-"


def validate_pdf_bytes(data: bytes) -> bool:
    """
    Check actual file magic bytes, not just MIME type.
    A file renamed to .pdf with wrong content will fail here before
    ever reaching pypdf.
    """
    return data[:5] == _PDF_MAGIC


# ── PDF text extraction ───────────────────────────────────────────────────────
def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """
    Extract all text from PDF. Returns (full_text, page_count).
    Raises ValueError if PDF is fully image-only with no extractable text.
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


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks using recursive character splitting.
    chunk_size and chunk_overlap are controlled via config.py.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]


# ── Storage ───────────────────────────────────────────────────────────────────
def store_chunks(session_id: str, chunks: list[str]) -> str:
    """
    Store chunks in two indexes simultaneously:

    1. ChromaDB collection (vector search) — cosine similarity via embeddings
    2. BM25 in-memory index (keyword search) — exact token matching

    Both indexes are built from the same chunk list.
    retrieve_chunks_hybrid() queries both and merges results via RRF.
    """
    collection_name = f"session_{session_id}"

    # ── ChromaDB: delete stale collection, create fresh ───────────────────────
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    # ── BM25: tokenize chunks and build keyword index ─────────────────────────
    # Lowercase + whitespace tokenization — simple but effective for BM25
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    _bm25_store[session_id] = (bm25_index, chunks)

    logger.info(
        "Indexed %d chunks for session=%s — ChromaDB (vector) + BM25 (keyword) both ready",
        len(chunks),
        session_id,
    )

    return collection_name


# ── RRF fusion ────────────────────────────────────────────────────────────────
def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge multiple ranked lists of documents using Reciprocal Rank Fusion.

    Formula: RRF_score(doc) = Σ  1 / (k + rank)
    - rank is 1-indexed (first position = rank 1)
    - k=60 is the standard constant from the original 2009 RRF paper
    - A document that ranks highly in multiple lists gets a much higher fused score

    Why this works: if a chunk is rank 1 in vector search AND rank 2 in BM25,
    it scores 1/61 + 1/62 = 0.032. A chunk that only appears in one list at rank 5
    scores 1/65 = 0.015. The dual-list champion wins decisively.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank_idx, doc in enumerate(ranked_list):
            rank = rank_idx + 1  # 1-indexed
            contribution = 1.0 / (k + rank)
            scores[doc] = scores.get(doc, 0.0) + contribution

    # Sort descending — highest fused score first
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


# ── Hybrid retrieval ──────────────────────────────────────────────────────────
def retrieve_chunks_hybrid(
    session_id: str, query: str, top_k: Optional[int] = None
) -> list[str]:
    """
    PRIMARY RETRIEVAL FUNCTION — use this everywhere, not retrieve_chunks().

    Runs vector search and BM25 keyword search in parallel, then merges
    results using Reciprocal Rank Fusion. Returns top_k fused chunks.

    Fallback behavior:
    - If BM25 index is missing (server restarted) → pure vector search
    - If vector search fails → pure BM25
    - If both fail → empty list

    Each source retrieves (top_k * 3) candidates before fusion, so the
    fusion has a wide pool to work with and can surface the best chunks
    even if they ranked lower in one individual source.
    """
    if top_k is None:
        top_k = settings.top_k_results

    # How many candidates to pull from each source before fusion
    # More candidates = better fusion quality, but slightly slower
    n_candidates = min(top_k * 3, 50)

    collection_name = f"session_{session_id}"

    # ── Step 1: Vector search ─────────────────────────────────────────────────
    vector_ranked: list[str] = []
    try:
        collection = chroma_client.get_collection(collection_name)
        query_embedding = embedding_model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_candidates, collection.count()),
        )
        vector_ranked = results["documents"][0] if results["documents"] else []

    except Exception as exc:
        logger.warning("Vector search failed for session=%s: %s", session_id, exc)

    # ── Step 2: BM25 keyword search ───────────────────────────────────────────
    bm25_ranked: list[str] = []

    if session_id in _bm25_store:
        bm25_index, all_chunks = _bm25_store[session_id]
        tokenized_query = query.lower().split()

        # get_scores returns a numpy array — one score per chunk
        bm25_scores = bm25_index.get_scores(tokenized_query)

        # Only keep chunks with a positive BM25 score (actual keyword matches)
        scored = [
            (all_chunks[i], float(bm25_scores[i]))
            for i in range(len(all_chunks))
            if bm25_scores[i] > 0.0
        ]

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            bm25_ranked = [chunk for chunk, _ in scored[:n_candidates]]
            logger.info(
                "BM25 keyword matches: %d chunks for session=%s",
                len(bm25_ranked),
                session_id,
            )
        else:
            logger.info(
                "BM25: zero keyword matches for this query in session=%s "
                "— query words not present verbatim in any chunk",
                session_id,
            )
    else:
        logger.warning(
            "BM25 index not found for session=%s — "
            "server likely restarted after this session was created. "
            "Falling back to pure vector search.",
            session_id,
        )

    # ── Step 3: Decide what to return ─────────────────────────────────────────
    if not vector_ranked and not bm25_ranked:
        logger.error("Both vector and BM25 returned empty for session=%s", session_id)
        return []

    # Only one source available — return it directly, no fusion needed
    if not bm25_ranked:
        logger.info("Returning pure vector results (no BM25 candidates)")
        return vector_ranked[:top_k]

    if not vector_ranked:
        logger.info("Returning pure BM25 results (vector search failed)")
        return bm25_ranked[:top_k]

    # ── Step 4: RRF fusion ────────────────────────────────────────────────────
    fused = _reciprocal_rank_fusion([vector_ranked, bm25_ranked])

    logger.info(
        "Hybrid RRF complete — session=%s | vector=%d | bm25=%d | fused=%d | returning=%d",
        session_id,
        len(vector_ranked),
        len(bm25_ranked),
        len(fused),
        min(top_k, len(fused)),
    )

    return fused[:top_k]


# ── Legacy pure-vector retrieval (kept for backward compat) ───────────────────
def retrieve_chunks(
    session_id: str, query: str, top_k: Optional[int] = None
) -> list[str]:
    """
    Original pure-vector retrieval. Not used in the main pipeline anymore.
    retrieve_chunks_hybrid() is the correct function to call.
    Kept here so nothing breaks if anything else references this by name.
    """
    if top_k is None:
        top_k = settings.top_k_results
    collection_name = f"session_{session_id}"
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        return []
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding, n_results=min(top_k, collection.count())
    )
    return results["documents"][0] if results["documents"] else []


# ── Session existence check ───────────────────────────────────────────────────
def session_exists(session_id: str) -> bool:
    try:
        chroma_client.get_collection(f"session_{session_id}")
        return True
    except Exception:
        return False


# ── Full PDF pipeline ─────────────────────────────────────────────────────────
def process_pdf(
    file_path: str,
    session_id: str,
    on_chunks_ready: Optional[Callable] = None,
    on_done: Optional[Callable] = None,
) -> tuple[int, int]:
    """
    Full ingestion pipeline: extract → chunk → store (ChromaDB + BM25).
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


# ── Background cleanup stub ───────────────────────────────────────────────────
def delete_old_collections(max_age_seconds: int = 86400):
    """
    Cleanup stub. Full implementation would:
    1. Track session creation timestamps in Redis
    2. Delete ChromaDB collections older than max_age_seconds
    3. Also pop the corresponding entry from _bm25_store
    Render free tier wipes everything on redeploy anyway.
    """
    pass
