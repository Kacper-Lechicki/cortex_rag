import json
import math

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalQuery:
    """
    A single evaluation question with expected answer metadata.

    Attributes:
        id: Unique identifier
        question: The natural language question.
        relevant_source: Partial URL/domain that should appear in retrieved results.
    """

    id: str
    question: str
    relevant_source: str


@dataclass
class EvalResult:
    """
    Evaluation result for a single query.

    Attributes:
        query_id: ID of the evaluated question.
        question: The question text.
        retrieved_sources: List of source URLs from retrieved chunks (in rank order).
        relevant_source: The expected relevant source (partial URL/domain).
    """

    query_id: str
    question: str
    retrieved_sources: list[str]
    relevant_source: str

    @property
    def reciprocal_rank(self) -> float:
        """
        Compute Reciprocal Rank for this query.

        RR = 1 / rank_of_first_relevant_result = 0 if relevant result not found

        Example:
            retrieved = ["a.com", "b.com", "relevant.com"]
            relevant_source = "relevant.com" -> rank = 3, RR = 1/3 = 0.333
        """

        for rank, source in enumerate(self.retrieved_sources, start=1):
            if self.relevant_source in source:
                return 1.0 / rank

        return 0.0

    @property
    def hit(self) -> bool:
        """
        True if the relevant source appears anywhere in the retrieved results.

        This is Hit Rate @ K where K = len(retrieved_sources).
        Simpler than MMR - just cheks presence, not position.
        """

        return any(self.relevant_source in s for s in self.retrieved_sources)


@dataclass
class EvalReport:
    """
    Aggregated evaluation report for a full evaluation set run.

    Attributes:
        experiment_name: Label for this run (e.g. 'baseline', 'smaller_chunks').
        results: List of per-query EvalResult objects.
        config_snapshot: Copy of relevant config at evaluation time.
    """

    experiment_name: str
    results: list[EvalResult]
    
    # field(default_factory=dict) means: default value is a new empty dict for each instance.
    config_snapshot: dict = field(default_factory=dict)

    @property
    def mrr(self) -> float:
        """
        Mean Reciprocal Rank across all queries.

        MRR = (1/N) * Σ RR_i

        Interpretation:
            0.0 - 0.3 : poor retrieval
            0.3 - 0.6 : acceptable
            0.6 - 0.8 : good
            0.8 - 1.0 : excellent
        """

        if not self.results:
            return 0.0

        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def hit_rate(self) -> float:
        """
        Fraction of queries where the relevant source appeared in Top-K results.

        Hit Rate = (# queries with at least one relevant result) / (# total queries)
        """

        if not self.results:
            return 0.0

        return sum(1 for r in self.results if r.hit) / len(self.results)

    def to_dict(self) -> dict:
        """
        Serialize report to a JSON-compatible dict.
        """

        return {
            "experiment": self.experiment_name,
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "n_queries": len(self.results),
            "config": self.config_snapshot,
        }

    def save(self, path: Path) -> None:
        """
        Save evaluation report to a JSON file.

        Creates parent directories if they don't exist.

        Args:
            path: Full path to the output JSON file.
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        # Context manager for file I/O: file is automatically closed after the block
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_all(directory: Path) -> list[dict]:
        """
        Load all saved evaluation reports from a directory.

        @staticmethod: a method that belongs to the class logically but doesn't
        need access to instance (self) or class (cls). It's just a namespace tool.

        Args:
            directory: Path to directory containing eval_*.json files.

        Returns:
            List of report dicts, sorted by filename (chronological).
        """

        reports = []

        # Path.glob() returns a generator of matching paths
        for file in sorted(directory.glob("eval_*.json")):
            with file.open(encoding="utf-8") as f:
                reports.append(json.load(f))

        return reports


def ndcg_at_k(retrieved_sources: list[str], relevant_source: str, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    A more sophisticated metric than MRR that accounts for the quality
    of the full ranking, not just the first relevant result.

    DCG penalizes relevant results that appear lower in the ranking
    using a logarithmic scale (log2).

    Args:
        retrieved_sources: Ranked list of source URLs.
        relevant_source: The expected relevant source (partial URL/domain).
        k: Cutoff rank.

    Returns:
        NDCG score between 0.0 and 1.0.
    """

    # Compute DCG: relevance / log2(rank + 1) for each position
    relevances = [
        1.0 if relevant_source in src else 0.0 for src in retrieved_sources[:k]
    ]

    dcg = sum(
        rel / math.log2(rank + 2)  # rank+2 because rank is 0-indexed here
        for rank, rel in enumerate(relevances)
    )

    # Ideal DCG: all relevant results at the top
    ideal_relevances = sorted(relevances, reverse=True)

    ideal_dcg = sum(
        rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_relevances)
    )

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
