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
# Maps session_id → (BM25Okapi index, child_chunks list, child_to_parent_ids list)
# Third element is new in Phase 4: lets us map any child text → its parent index
# without hitting ChromaDB for a metadata lookup.
_bm25_store: dict[str, tuple[BM25Okapi, list[str], list[int]]] = {}

# ── PDF validation ────────────────────────────────────────────────────────────
_PDF_MAGIC = b"%PDF-"


def validate_pdf_bytes(data: bytes) -> bool:
    return data[:5] == _PDF_MAGIC


# ── PDF text extraction ───────────────────────────────────────────────────────
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


# ── Hierarchical chunking ─────────────────────────────────────────────────────
def chunk_text_hierarchical(text: str) -> tuple[list[str], list[str], list[int]]:
    """
    Split text into two levels of chunks.

    Parent chunks (1000 chars, 100 overlap):
        Sent to the LLM. Large enough to contain a complete thought,
        a full policy clause, a complete instruction sequence.
        These are what Groq reads. Context-rich.

    Child chunks (350 chars, 40 overlap):
        Used for retrieval — searched by vector and BM25.
        Small enough for embeddings to be precise and focused.
        Each child knows which parent it came from via child_to_parent.

    Why this works:
        Search precision comes from small chunks.
        Answer quality comes from large chunks.
        Hierarchical chunking gives you both simultaneously.

    Returns:
        child_chunks     — list of small retrieval-target strings
        parent_chunks    — list of large LLM-context strings
        child_to_parent  — child_to_parent[i] = index of parent for child i
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350, chunk_overlap=40, separators=["\n\n", "\n", ".", "!", "?", " "]
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

        # If parent is too short to split further, it becomes its own child
        if not children:
            children = [parent]

        child_chunks.extend(children)
        child_to_parent.extend([parent_idx] * len(children))

    logger.info(
        "Chunking complete — parents: %d | children: %d | avg children/parent: %.1f",
        len(parent_chunks),
        len(child_chunks),
        len(child_chunks) / max(len(parent_chunks), 1),
    )

    return child_chunks, parent_chunks, child_to_parent


# ── Storage ───────────────────────────────────────────────────────────────────
def store_chunks_hierarchical(
    session_id: str,
    child_chunks: list[str],
    parent_chunks: list[str],
    child_to_parent: list[int],
) -> None:
    """
    Build three indexes for a session:

    1. ChromaDB child collection  — vector search on small, precise chunks
    2. ChromaDB parent collection — fetch full-context chunks by ID
    3. BM25 in-memory index       — keyword search on child chunks

    Child chunks carry metadata {"parent_id": int} so we can resolve
    any retrieved child back to its parent during retrieval.
    """
    child_col_name = f"session_{session_id}_child"
    parent_col_name = f"session_{session_id}_parent"

    # Delete any existing collections for this session (including old flat ones)
    for name in [child_col_name, parent_col_name, f"session_{session_id}"]:
        try:
            chroma_client.delete_collection(name)
        except Exception:
            pass

    # ── Child collection: embed + store with parent_id metadata ───────────────
    child_col = chroma_client.create_collection(
        name=child_col_name, metadata={"hnsw:space": "cosine"}
    )
    child_embeddings = embedding_model.encode(child_chunks).tolist()
    child_col.add(
        documents=child_chunks,
        embeddings=child_embeddings,
        ids=[f"child_{i}" for i in range(len(child_chunks))],
        metadatas=[{"parent_id": child_to_parent[i]} for i in range(len(child_chunks))],
    )

    # ── Parent collection: embed + store (fetched by ID during retrieval) ─────
    # We embed parents too so the collection is valid and future-queryable.
    parent_col = chroma_client.create_collection(
        name=parent_col_name, metadata={"hnsw:space": "cosine"}
    )
    parent_embeddings = embedding_model.encode(parent_chunks).tolist()
    parent_col.add(
        documents=parent_chunks,
        embeddings=parent_embeddings,
        ids=[f"parent_{i}" for i in range(len(parent_chunks))],
    )

    # ── BM25 in-memory index on child chunks ──────────────────────────────────
    tokenized_children = [chunk.lower().split() for chunk in child_chunks]
    bm25_index = BM25Okapi(tokenized_children)

    # Store (index, child texts, child→parent mapping) as a triple
    _bm25_store[session_id] = (bm25_index, child_chunks, child_to_parent)

    logger.info(
        "Indexes built — session=%s | parents=%d | children=%d | BM25 ready",
        session_id,
        len(parent_chunks),
        len(child_chunks),
    )


# ── RRF fusion ────────────────────────────────────────────────────────────────
def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.
    RRF_score(doc) = Σ 1 / (k + rank), rank is 1-indexed.
    k=60 is the standard constant from the 2009 paper.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_idx, doc in enumerate(ranked_list):
            rank = rank_idx + 1
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


# ── Hybrid retrieval ──────────────────────────────────────────────────────────
def retrieve_chunks_hybrid(
    session_id: str,
    query: str,
    top_k: Optional[int] = None,
    hyde_query: Optional[str] = None,
) -> list[str]:
    """
    Full hybrid retrieval pipeline with hierarchical parent resolution.

    Step 1 — Vector search on child collection
        Uses hyde_query for embedding if provided (Phase 5 HyDE),
        otherwise uses original query. Returns child texts + parent_id metadata.

    Step 2 — BM25 keyword search on child chunks
        Always uses original query — keyword search benefits from
        the user's exact words, not a hypothetical document passage.

    Step 3 — RRF fusion
        Merges vector and BM25 child ranked lists into one fused list.

    Step 4 — Parent resolution
        For each top child, look up its parent_id.
        Fetch the full parent chunk from ChromaDB.
        Deduplicate (multiple children can share a parent).

    Returns parent chunks — large, context-rich strings ready for
    the reranker and then the LLM.

    Args:
        query       — user's original question (used for BM25 + fallback vector)
        hyde_query  — hypothetical document passage (used for vector if set)
        top_k       — max number of unique parent chunks to return
    """
    if top_k is None:
        top_k = settings.top_k_results

    # Pull more candidates than top_k — RRF needs a wide pool to work well
    n_candidates = min(top_k * 3, 60)

    child_col_name = f"session_{session_id}_child"

    # ── Step 1: Vector search on child collection ─────────────────────────────
    vector_children: list[str] = []
    # Maps child text → its parent_id — built from ChromaDB metadata
    text_to_parent: dict[str, int] = {}

    # For vector embedding: prefer HyDE query if available (Phase 5)
    # HyDE bridges the phrasing gap between user language and document language
    vector_query = hyde_query if hyde_query else query

    try:
        child_col = chroma_client.get_collection(child_col_name)
        query_embedding = embedding_model.encode([vector_query]).tolist()

        results = child_col.query(
            query_embeddings=query_embedding,
            n_results=min(n_candidates, child_col.count()),
            include=["documents", "metadatas"],
        )

        if results["documents"] and results["documents"][0]:
            vector_children = results["documents"][0]
            # Build text→parent_id mapping from returned metadata
            for i, doc in enumerate(vector_children):
                meta = results["metadatas"][0][i]
                parent_id = meta.get("parent_id")
                if parent_id is not None:
                    text_to_parent[doc] = int(parent_id)

    except Exception as exc:
        logger.warning("Vector search failed for session=%s: %s", session_id, exc)

    # ── Step 2: BM25 keyword search on child chunks ───────────────────────────
    bm25_children: list[str] = []

    if session_id in _bm25_store:
        bm25_index, all_children, child_to_parent_map = _bm25_store[session_id]

        # BM25 always uses original query — exact user words matter here
        tokenized_query = query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)

        scored = [
            (all_children[i], float(bm25_scores[i]), child_to_parent_map[i])
            for i in range(len(all_children))
            if bm25_scores[i] > 0.0
        ]

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            for child_text, _, parent_id in scored[:n_candidates]:
                bm25_children.append(child_text)
                text_to_parent[child_text] = parent_id

            logger.info(
                "BM25 keyword matches: %d children for session=%s",
                len(bm25_children),
                session_id,
            )
        else:
            logger.info(
                "BM25: no keyword matches for session=%s — "
                "query words not present verbatim in any child chunk",
                session_id,
            )
    else:
        logger.warning(
            "BM25 index missing for session=%s "
            "(server restarted after upload?) — vector only",
            session_id,
        )

    # ── Step 3: Handle empty results ─────────────────────────────────────────
    if not vector_children and not bm25_children:
        logger.error("Both vector and BM25 returned empty for session=%s", session_id)
        return []

    # ── Step 4: RRF fusion on child texts ─────────────────────────────────────
    if not bm25_children:
        top_children = vector_children
    elif not vector_children:
        top_children = bm25_children
    else:
        fused = _reciprocal_rank_fusion([vector_children, bm25_children])
        top_children = fused

    logger.info(
        "Hybrid RRF — session=%s | vector=%d | bm25=%d | fused pool=%d",
        session_id,
        len(vector_children),
        len(bm25_children),
        len(top_children),
    )

    # ── Step 5: Resolve children → parent chunks ──────────────────────────────
    try:
        parent_col = chroma_client.get_collection(f"session_{session_id}_parent")
    except Exception as exc:
        # Parent collection missing — this shouldn't happen but degrade gracefully
        logger.error(
            "Parent collection not found for session=%s: %s — "
            "returning raw children as fallback",
            session_id,
            exc,
        )
        return top_children[:top_k]

    seen_parent_ids: set[int] = set()
    parent_docs: list[str] = []

    for child_text in top_children:
        if len(parent_docs) >= top_k:
            break

        parent_id = text_to_parent.get(child_text)
        if parent_id is None:
            continue

        if parent_id in seen_parent_ids:
            # Multiple children mapped to same parent — skip duplicate
            continue

        seen_parent_ids.add(parent_id)

        try:
            result = parent_col.get(ids=[f"parent_{parent_id}"])
            if result["documents"]:
                parent_docs.append(result["documents"][0])
        except Exception as exc:
            logger.warning(
                "Failed to fetch parent_%d for session=%s: %s",
                parent_id,
                session_id,
                exc,
            )

    logger.info(
        "Parent resolution — session=%s | unique parents fetched: %d | returning: %d",
        session_id,
        len(seen_parent_ids),
        len(parent_docs),
    )

    return parent_docs


# ── Session existence check ───────────────────────────────────────────────────
def session_exists(session_id: str) -> bool:
    """
    Check for Phase 4 hierarchical collections first.
    Falls back to legacy flat collection check for pre-Phase 4 sessions.
    Pre-Phase 4 sessions cannot be retrieved with the new pipeline —
    they'll need to be re-uploaded to get hierarchical indexes built.
    """
    try:
        chroma_client.get_collection(f"session_{session_id}_child")
        return True
    except Exception:
        try:
            # Backward compat — old flat sessions still "exist" for session checks
            # but retrieval will fail gracefully for them
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
    Full ingestion pipeline: extract → hierarchical chunk → store.
    Returns (page_count, child_chunk_count).
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

    store_chunks_hierarchical(session_id, child_chunks, parent_chunks, child_to_parent)

    if on_done:
        try:
            on_done()
        except Exception:
            pass

    # Return child count so upload response shows meaningful chunk count
    return page_count, len(child_chunks)


# ── Background cleanup stub ───────────────────────────────────────────────────
def delete_old_collections(max_age_seconds: int = 86400):
    """
    Cleanup stub. Full implementation would delete both _child and _parent
    collections for expired sessions, and pop from _bm25_store.
    Render free tier wipes everything on redeploy anyway.
    """
    pass
