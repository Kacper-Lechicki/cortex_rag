import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from rich.console import Console

from .chunker import Chunk
from .config import config
from .logging_utils import get_logger

# Heavy deps (ChromaDB + SentenceTransformers) are intentionally imported lazily
# inside functions. Importing them at module import time can take seconds and
# looks like the program is "stuck" (no loader can render before imports finish).
if TYPE_CHECKING:
    import chromadb  # noqa: F401
    from chromadb.api.client import Client  # noqa: F401

# Rich console for lightweight CLI feedback during long operations.
# We keep this inside the store layer because initialization and scans happen
# here and can otherwise look like the program is "stuck".
_console = Console(stderr=True)

# Module-level variables holding the singleton instances.
# Using None as sentinel value (not yet initialized).
# The | None syntax for Optional[...]
_client: Any | None = None
_collection: Any | None = None


def reset_store() -> None:
    """
    Reset the module-level Chroma singletons.

    This is necessary when the underlying on-disk store is deleted (e.g. `cortex clear`)
    during the same process lifetime. Otherwise, we can keep a stale in-memory client
    that points at removed files and later writes may fail.
    """

    global _client, _collection

    # Chroma's Rust backend holds file handles; just dropping references is not enough.
    # Best-effort close to release locks before the store directory is deleted.
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass

    _collection = None
    _client = None


@contextmanager
def _status(message: str):
    """
    Show progress feedback for long operations.

    - In an interactive terminal: show a Rich spinner.
    - Otherwise: print a single line so the user sees forward progress.
    """

    if _console.is_terminal and (sys.stderr.isatty() or sys.stdout.isatty()):
        with _console.status(f"[dim]{message}[/dim]", spinner="dots"):
            yield
    else:
        try:
            _console.print(f"[dim]{message}[/dim]")
        except Exception:
            # If output is heavily redirected or Rich can't write, stay silent.
            pass

        yield


def _get_collection():
    """
    Return the ChromaDB collection, initializing it on first call (lazy init).

    Lazy initialization pattern: we defer expensive setup until it's actually needed.
    This prevents loading the embedding model if the user just runs 'cortex --help'.

    Returns:
        The ChromaDB collection with cosine distance and sentence-transformer embeddings.
    """

    import contextlib
    import io
    import os
    import time

    global _client, _collection  # We need to modify module-level variables

    # If the on-disk store was deleted while this process is still alive (e.g. via
    # `cortex clear`), the cached client/collection becomes stale. Reset and re-init.
    if _collection is not None and not config.chroma_path.exists():
        reset_store()

    if _collection is not None:
        return _collection  # Already initialized - return immediately

    log = get_logger(__name__)
    t0 = time.perf_counter()

    # Import heavy dependencies only when needed.
    with _status("Loading vector store dependencies..."):
        import chromadb

        from chromadb.config import Settings

        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

    # Ensure Hugging Face Hub sees the token for model downloads.
    # (SentenceTransformers/huggingface_hub read it from env.)
    os.environ.setdefault("HF_TOKEN", config.hf_token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", config.hf_token)

    # Initialize embedding function.
    # IMPORTANT: normalize_embeddings=True ensures vectors have unit length,
    # which makes cosine similarity mathematically equivalent to dot product.
    # Without normalization, cosine scores can be misleading.
    # Sentence-Transformers / Transformers can print noisy load reports and progress bars
    # directly to stdout/stderr. Keep CLI output clean and send that noise to debug logs.
    buf = io.StringIO()

    with _status("Loading embedding model (first run may take a while)..."):
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=config.embedding_model,
                device="cpu",
                normalize_embeddings=True,  # CRITICAL
            )

    noisy = buf.getvalue().strip()

    if noisy:
        log.debug("Embedding model load output suppressed:\n%s", noisy)

    # PersistentClient: data is saved to disk automatically after every write.
    # anonymized_telemetry=False: opt out of usage statistics sent to ChromaDB.
    log.info("Initializing Chroma PersistentClient at %s", str(config.chroma_path))

    with _status("Opening knowledge base..."):
        _client = chromadb.PersistentClient(
            path=str(config.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )

    # get_or_create_collection: idempotent — safe to call multiple times.
    # Creates the collection if it doesn't exist, returns it if it does.
    # configuration={"hnsw": {"space": "cosine"}} sets cosine distance for HNSW index.
    # HNSW (Hierarchical Navigable Small World) is the approximate nearest-neighbor
    # algorithm ChromaDB uses internally for fast similarity search.
    with _status("Preparing vector store..."):
        _collection = _client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=embedding_fn,
            configuration={"hnsw": {"space": "cosine"}},
        )

    log.info(
        "Chroma collection ready (name=%s) in %.2fs",
        config.collection_name,
        time.perf_counter() - t0,
    )

    return _collection


def upsert_chunks(chunks: list[Chunk]) -> None:
    """
    Add or update chunks in the vector store.

    Uses upsert (not add): if a chunk with the same ID already exists, it will be overwritten.
    Additionally, we delete any existing chunks for the same source URL(s) before upserting.
    This prevents stale/low-quality chunks from older scraper versions from coexisting with
    newer chunks for the same URL.

    Args:
        chunks: List of Chunk objects to store.
    """

    if not chunks:
        return

    log = get_logger(__name__)
    collection = _get_collection()

    # Ensure stable chunk_id is queryable later (e.g., for RRF fusion).
    for c in chunks:
        if c.metadata is None:
            c.metadata = {}

        c.metadata.setdefault("chunk_id", c.chunk_id)

    sources = sorted({(c.metadata or {}).get("source", "") for c in chunks})
    sources = [s for s in sources if s]

    # Best-effort cleanup: remove previous chunks for these sources.
    for src in sources:
        try:
            with _status("Refreshing source chunks..."):
                collection.delete(where={"source": src})
        except Exception as e:
            # Chroma's underlying DB can become read-only due to filesystem permissions
            # or a locked database file. Deleting is an optimization; don't crash ingest.
            log.warning("Failed to delete existing chunks for source=%s: %s", src, e)

    # upsert requires parallel lists of ids, documents and metadatas.
    # They must be the same length and in the same order.
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.content for c in chunks],
        metadatas=[c.metadata for c in chunks],
        # We don't pass embeddings - ChromaDB computes them using embedding_fn
    )

    log.info(
        "Upserted %d chunks into collection=%s", len(chunks), config.collection_name
    )


def query_store(query_text: str, n_results: int | None = None) -> dict:
    """
    Search the vector store for chunks semantically similar to query_text.

    ChromaDB embeds query_text using the same embedding_fn, then finds
    the nearest vectors using cosine distance.

    Args:
        query_text: The search query (plain text, not pre-embedded).
        n_results: Number of results to return. Defaults to config.top_k.

    Returns:
        Raw ChromaDB result dict with 'documents', 'metadatas', 'distances' keys.
    """

    collection = _get_collection()
    k = n_results or config.top_k

    return collection.query(
        query_texts=[query_text],
        n_results=min(k, collection.count()),  # can't request more than we have
        include=["documents", "metadatas", "distances"],
    )


def get_all_for_visualization() -> tuple[list, list]:
    """
    Return all embeddings and metadata for visualization with t-SNE.

    Returns:
        Tuple of (embeddings, metadatas). embeddings is a list of float vectors.
    """

    collection = _get_collection()
    data = collection.get(include=["embeddings", "metadatas"])

    return data["embeddings"], data["metadatas"]


def count() -> int:
    """
    Return the total number of chunks stored.
    """

    collection = _get_collection()

    with _status("Counting chunks..."):
        return collection.count()


def collection_exists() -> bool:
    """
    Check if the vector store has been initialized (chroma dir exists).
    """

    if not config.chroma_path.exists():
        return False

    return count() > 0


def list_sources() -> list[dict]:
    """
    List unique ingested sources aggregated from chunk metadatas.

    Returns:
        List of dicts: {source, title, domain, count}, sorted by count desc.
    """

    if not collection_exists():
        return []

    collection = _get_collection()

    with _status("Scanning sources..."):
        data = collection.get(include=["metadatas"])

    counts: dict[str, int] = {}
    titles: dict[str, str] = {}
    domains: dict[str, str] = {}

    for md in data.get("metadatas") or []:
        if not md:
            continue

        source = md.get("source")

        if not source:
            continue

        counts[source] = counts.get(source, 0) + 1

        if source not in titles and md.get("title"):
            titles[source] = md["title"]

        if source not in domains and md.get("domain"):
            domains[source] = md["domain"]

    out = [
        {
            "source": src,
            "title": titles.get(src, ""),
            "domain": domains.get(src, ""),
            "count": n,
        }
        for src, n in counts.items()
    ]

    out.sort(key=lambda x: x["count"], reverse=True)

    return out


def delete_source(source_url: str) -> int:
    """
    Delete all chunks that match a given source URL.

    Returns:
        Number of deleted chunks (best-effort count).
    """

    if not collection_exists():
        return 0

    collection = _get_collection()
    log = get_logger(__name__)

    # Best-effort count before deletion (Chroma delete result doesn't always include a count).
    with _status("Looking up chunks to delete..."):
        before = collection.get(where={"source": source_url}, include=["metadatas"])

    metadatas = before.get("metadatas") or []
    n_before = len([m for m in metadatas if m])

    with _status("Deleting chunks..."):
        collection.delete(where={"source": source_url})
        
    log.info("Deleted source=%s (chunks=%d)", source_url, n_before)

    return n_before
