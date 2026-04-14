from dataclasses import dataclass

from .store import query_store
from .config import config


@dataclass
class SearchResult:
    """
    A single retrieved chunk with its metadata and similarity score.

    Attributes:
        content: The text content of the chunk.
        metadata: Source URL, title, domain, etc.
        distance: Cosine distance to query (0=identical, 1=orthogonal, 2=opposite).
    """

    content: str
    metadata: dict[str, str]
    distance: float

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "")

    @property
    def source(self) -> str:
        """
        Conveniece accessor for the source URL from metadata.
        """

        return self.metadata.get("source", "unknown")

    @property
    def similarity(self) -> float:
        """
        Convert cosine distance to similarity score (0-1, higher=better).
        """

        # ChromaDB returns distance, not similarity.
        # For normalized vectors with cosine space: distance = 1 - cosine_similarity
        return 1.0 - self.distance

    def __repr__(self) -> str:
        preview = self.content[:40].replace("\n", " ")
        return f"SearchResult(sim={self.similarity:.3f}, source={self.source!r}, text={preview!r})"


def retrieve(question: str, n: int | None = None) -> list[SearchResult]:
    """
    Retrieve relevant chunks for a question.

    Dispatches to either simple retrieval or query expansion based on config.

    Args:
        question: Natural language question.
        n: Number of results. Defaults to config.top_k.

    Returns:
        List of SearchResult objects, ranked by relevance.
    """

    k = n or config.top_k

    if config.use_query_expansion:
        return _retrieve_with_expansion(question, k)

    return _retrieve_simple(question, k)


def _retrieve_simple(question: str, k: int) -> list[SearchResult]:
    """
    Single-query retrieval - baseline approach.
    """

    raw = query_store(question, n_results=k)
    return _parse_raw_results(raw)


def _retrieve_with_expansion(question: str, k: int) -> list[SearchResult]:
    """
    Multi-query retrieval with Reciprocal Rank Fusion.

    1. Generate N query variants using the LLM
    2. Retrieve top-K chunks for each variant
    3. Merge rankings using RRF to surface consistently relevant chunks
    4. Return final top-K from merged ranking
    """

    # Import here to avoid circular imports (generator imports config)
    from .generator import generate_query_variants

    variants = generate_query_variants(question, n=3)

    # Dict mapping a stable chunk id to its SearchResult object
    id_to_result: dict[str, SearchResult] = {}

    # List of ranked lists (one per query variant)
    all_rankings: list[list[str]] = []

    for variant in variants:
        raw = query_store(variant, n_results=k)
        results = _parse_raw_results(raw)

        ranking: list[str] = []

        for result in results:
            chunk_id = result.chunk_id or (result.source + "|" + result.content[:50])
            id_to_result[chunk_id] = result
            ranking.append(chunk_id)

        all_rankings.append(ranking)

    # Fuse rankings and return top-K
    fused_keys = _reciprocal_rank_fusion(all_rankings)[:k]

    return [id_to_result[key] for key in fused_keys if key in id_to_result]


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF formula: score(doc) = Σ_rankings 1 / (k + rank_in_ranking) where k=60 is the smoothing constant from the original 2009 paper.

    Higher k → less penalization for lower ranks.
    Higher score → document appeared consistently across query variants.

    Args:
        rankings: List of ranked result ID lists (one list per query variant).
        k: RRF smoothing constant. 60 is the recommended default.

    Returns:
        Merged list of IDs sorted by descending RRF score.
    """

    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            # += 0.0 if key doesn't exist yet (dict.get default)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by score descending, return just the IDs
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def _parse_raw_results(raw: dict) -> list[SearchResult]:
    """
    Convert ChromaDB raw query result into SearchResult objects.

    ChromaDB returns results as parallel lists:
        raw["documents"][0] = ["chunk text 1", "chunk text 2", ...]
        raw["metadatas"][0] = [{"source": "..."}, ...]
        raw["distances"][0] = [0.12, 0.34, ...]

    The [0] indexing is because ChromaDB supports batched queries —
    we always send one query at a time, so we take the first (only) result set.
    """

    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    # zip() pairs up three parallel lists element by element
    return [
        SearchResult(content=doc, metadata=meta, distance=dist)
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
