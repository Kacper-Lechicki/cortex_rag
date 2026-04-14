import hashlib

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    Represents a single text chunk ready for embedding.

    A chunk is a fragment of an article, small enough for the embedding model and annotated with metadata about its origin.

    Attributes:
        content: The text content of this chunk.
        metadata: Dict with source URL, title, domain, etc.
        chunk_id: Deterministic SHA-256 hash of content (first 16 hex chars).
    """

    content: str
    metadata: dict[str, str]

    # field(default="") means this field has a default value - required for
    # fields after fields without defaults in dataclasses
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        """
        Called automatically by @dataclass after __init__.

        Use __post_init__ to compute derived fields or validate data.
        Here we generate a deterministic ID from content hash if not provided.
        """

        if not self.chunk_id:
            # SHA-256 hash of content -> deterministic, unique-ish 16-char ID
            # encode() converts str to bytes (required by hashlib)
            # hexdigest() returns hex string representation
            self.chunk_id = hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ")
        return f"Chunk(id={self.chunk_id!r}, chars={len(self.content)}, preview={preview!r})"


class TextChunker:
    """
    Splits plain text into overlapping chunks using recursive character splitting.

    Strategy: try to split on paragraph boundaries first (\n\n), then single newlines (\n), then sentence boundaries ('. '), then spaces (' ').
    Falls back to hard character split as a last resort.

    This mirrors LangChain's RecursiveCharacterSplitter behavior but implemented from scratch for educational purposes.

    Attributes:
        chunk_size: Maximum chunk size in characters.
        overlap: Number of characters to overlap between adjacent chunks.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, str]) -> list[Chunk]:
        """
        Split text into overlapping Chunk objects.

        Args:
            text: The full text to split.
            metadata: Metadata dict attached to every chunk (source URL, title, etc.)

        Returns:
            List of Chunk objects. Empty list if text is empty.
        """

        if not text.strip():
            return []

        raw_parts = self._split_recursive(text.strip())

        # List comprehension: build list by filtering and transforming in one expression
        # Equivalent to a for loop with an if condition
        chunks: list[Chunk] = []

        for i, part in enumerate(raw_parts):
            part = part.strip()

            if len(part) <= 30:
                continue

            # IDs must be unique within a Chroma collection.
            # Content-only hashes can collide when overlap produces identical text.
            source = metadata.get("source", "")
            chunk_id = hashlib.sha256(f"{source}|{i}|{part}".encode()).hexdigest()[:16]
            chunks.append(Chunk(content=part, metadata=metadata, chunk_id=chunk_id))

        return chunks

    def _split_recursive(self, text: str) -> list[str]:
        """
        Split text into chunks, preferring natural boundaries.

        This is intentionally iterative and near-linear time. The previous
        recursive splitter could degrade badly on long documents with many
        separators.
        """

        separators = ["\n\n", "\n", ". ", " "]
        text = text.strip()

        if not text:
            return []

        out: list[str] = []
        i = 0
        n = len(text)

        while i < n:
            end = min(i + self.chunk_size, n)
            window = text[i:end]

            split_end = end

            # Prefer a boundary near the end of the window (avoid tiny chunks).
            min_idx = max(0, int(len(window) * 0.6))

            for sep in separators:
                idx = window.rfind(sep)

                if idx >= min_idx:
                    split_end = i + idx + len(sep)
                    break

            chunk = text[i:split_end].strip()

            if chunk:
                out.append(chunk)

            # Move forward, keeping overlap, but ensure progress.
            next_i = split_end - self.overlap
            i = max(next_i, i + 1)

        return out
