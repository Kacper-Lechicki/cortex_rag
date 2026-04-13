import chromadb

from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .chunker import Chunk
from .config import config

# Module-level variables holding the singleton instances.
# Using None as sentinel value (not yet initialized).
# The | None syntax for Optional[...]
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """
    Return the ChromaDB collection, initializing it on first call (lazy init).

    Lazy initialization pattern: we defer expensive setup until it's actually needed.
    This prevents loading the embedding model if the user just runs 'cortex --help'.

    Returns:
        The ChromaDB collection with cosine distance and sentence-transformer embeddings.
    """

    global _client, _collection # We need to modify module-level variables

    if _collection is not None:
        return _collection # Already initialized - return immediately

    # Initialize embedding function.
    # IMPORTANT: normalize_embeddings=True ensures vectors have unit length,
    # which makes cosine similarity mathematically equivalent to dot product.
    # Without normalization, cosine scores can be misleading.
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=config.embedding_model,
        device="cpu",
        normalize_embeddings=True # CRITICAL
    )

    # PersistentClient: data is saved to disk automatically after every write.
    # anonymized_telemetry=False: opt out of usage statistics sent to ChromaDB.
    _client = chromadb.PersistentClient(
        path=str(config.chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )

    # get_or_create_collection: idempotent — safe to call multiple times.
    # Creates the collection if it doesn't exist, returns it if it does.
    # configuration={"hnsw": {"space": "cosine"}} sets cosine distance for HNSW index.
    # HNSW (Hierarchical Navigable Small World) is the approximate nearest-neighbor
    # algorithm ChromaDB uses internally for fast similarity search.
    _collection = _client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=embedding_fn,
        configuration={"hnsw": {"space": "cosine"}}
    )

    return _collection


def upsert_chunks(chunks: list[Chunk]) -> None:
    """
    Add or update chunks in the vector store.

    Uses upsert (not add): if a chunk with the same ID already exists, it will be overwritten. This makes ingestion idempotent - you can safely re-run 'cortex add' on the same URL's.

    Args:
        chunks: List of Chunk objects to store.
    """

    if not chunks:
        return

    collection = _get_collection()

    # upsert requires parallel lists of ids, documents and metadatas.
    # They must be the same length and in the same order.
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.content for c in chunks],
        metadatas=[c.metadata for c in chunks],
        # We don't pass embeddings - ChromaDB computes them using embedding_fn
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

    return _get_collection().count()


def collection_exists() -> bool:
    """
    Check if the vector store has been initialized (chroma dir exists).
    """

    return config.chroma_path.exists() and count() > 0
