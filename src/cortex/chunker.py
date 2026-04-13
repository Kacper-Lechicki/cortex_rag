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
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

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
        return [
            Chunk(content=part, metadata=metadata)
            for part in raw_parts
            if len(part.strip()) > 30 # skip fragments that are too short to be useful
        ]


    def _split_recursive(self, text: str) -> list[str]:
        """
        Recursively split text using progressively smaller separators.

        Tries to find a natural split point that keeps chunks within size limit.
        """

        if len(text) <= self.chunk_size:
            return [text.strip()]

        # Try each separator in order of preference
        for separator in ["\n\n", "\n", ". ", " "]:
            parts = text.split(separator)

            if len(parts) <= 1:
                continue # this separator not found, try next

            merged: list[str] = []
            current = parts[0]

            for part in parts[1:]:
                candidate = current + separator + part

                if len(candidate) <= self.chunk_size:
                    # Still fits - keep building the current chunk
                    current = candidate
                else:
                    # Doesn't fit - save current chunk, start a new one
                    if current.strip():
                        merged.append(current.strip())

                    # Add overlap: take last N chars of current as start of new chunk
                    overlap_text = current[-self.overlap:] if self.overlap > 0 else ""
                    current = (overlap_text + separator + part).strip()

            if current.strip():
                merged.append(current.strip())

            # Only accept this split if it actually divided the text
            if len(merged) > 1:
                # Recursive: some merged chunks might still be too large
                result: list[str] = []

                for m in merged:
                    result.extend(self._split_recursive(m))

                return result

        # Last resort: hard cut at chunk_size characters
        return [
            text[i : i + self.chunk_size]
            for  i in range(0, len(text), self.chunk_size - self.overlap)
        ]
