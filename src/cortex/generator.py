import functools
import time

from typing import TypeVar, Callable
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

from .config import config

# TypeVar is used for generic type annotations.
# F bound to Callable means 'any callable type'.
F = TypeVar("F", bound=Callable)


def retry(max_attempts: int = 3, base_delay: float = 2.0) -> Callable[[F], F]:
    """
    Decorator factory that adds exponential backoff retry logic to a function.

    When the decorated function raises a retryable exception (network errors, rate limits, timeouts), it automatically waits and tries again.

    Backoff schedule with base_delay=2.0:
        Attempt 1 fails -> wait 2s
        Attempt 2 fails -> wait 4s
        Attempt 3 fails -> raise last exception

    Args:
        max_attempts: Maximum number of tries before giving up.
        base_delay: Base delay in seconds (doubles after each failure).

    Returns:
        A decorator function.

    Example:
    @retry(max_attempts=3, base_delay=1.0)
    def call_api():
        ...
    """


    # This is the actual decorator - takes the function to be wrapped
    def decorator(func: F) -> F:
        # functools.wraps copies __name__, __doc__, __annotations__ from func to wrapper.
        # Without this, help(generate_answer) would show "wrapper" instead of "generate_answer".
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            *args = positional arguments passed to the original function
            **kwargs = keyword arguments passed to the original function
            We forward them unchanged - the wrapper is transparent.
            """

            last_error: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    # Call the original function and return its result
                    return func(*args, **kwargs)
                except (HfHubHTTPError, InferenceTimeoutError, OSError) as e:
                    last_error = e
                    error_str = str(e)

                    # Non-retryable: out of credits -> fail immediately
                    if "402" in error_str:
                        raise RuntimeError(
                            "HuggingFace monthly credits exhausted. "
                            "See https://huggingface.co/settings/billing"
                        ) from e

                    if attempt < max_attempts:
                        # Exponential backoff: delay doubles with each attempt
                        # attempt=1 -> 2s, attempt=2 -> 4s, attempt=3 -> (won't reach)
                        delay = base_delay * (2 ** (attempt - 1))

                        print(f"    [retry] Attempt {attempt}/{max_attempts} failed: {e}")
                        print(f"    [retry] Waiting {delay:.0f}s before next attempt...")

                        time.sleep(delay)

                # All attempt exhausted
                raise RuntimeError(
                    f"All {max_attempts} attempts failed for {func.__name__}"
                ) from last_error

        # The cats here tells type checkers that wrapper has the same signature as func
        return wrapper

    return decorator


# --- HuggingFace InferenceClient ---

# Lazy-loaded singleton. None = not yet initialized.
_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    """
    Return a shared InferenceClient instance (lazy singleton).

    Using provider='hf-inference' routes to HuggingFace's own inference servers.
    This is free (rate-limited) and does NOT consume your $0.10/month credit.
    Third-party providers (Together, Cerebras, etc.) DO consume credits.
    """

    global _client

    if _client is None:
        _client = InferenceClient(
            provider="hf-inference",
            api_key=config.hf_token
        )

    return _client


@retry(max_attempts=3, base_delay=2.0)
def generate_answer(question: str, context: str) -> str:
    """
    Generate a grounded answer using the HuggingFace Inference API.

    The model is instructed to answer ONLY from the provided context.
    If the context is insufficent, it should say so explicitly.

    Args:
        question: The user's question.
        context: Retrieved document chunks joined with separators.

    Returns:
        Generated answer as a string.
    """

    client = _get_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based ONLY on "
                "the provided context. If the context does not contain enough "
                "information to answer the question, respond with: "
                "'I don't have enough information in my knowledge base to answer this.'"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        }
    ]

    response = client.chat_completion(
        model=config.generation_model,
        messages=messages,
        max_tokens=512,
        temperature=0.2 # Low temperature = more factual, less creative
    )

    return response.choices[0].message.content.strip()


@retry(max_attempts=2, base_delay=1.0)
def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """
    Ask the LLM to generate N diverse reformulations of the query.

    Used for query expansion in retriever.py.

    Args:
        query: Original user question.
        n: Number of variants to generate (not counting the original).

    Returns:
        List of query strings starting with the original, followed by variants.
    """

    client = _get_client()

    response = client.chat_completion(
        model=config.generation_model,
        messages=[
            {
                "role": "system",
                "content": "Generate diverse search queries. Return ONLY numbered queries, one per line. No explanations."
            },
            {
                "role": "user",
                "content": f"Generate {n} different ways to search for information about:\n'{query}'"
            }
        ],
        max_tokens=200,
        temperature=0.7 # Higher temperature = more diverse variants
    )

    variants: list[str] = [query] # Always include the original query first

    for line in response.choices[0].message.content.strip().split("\n"):
        # Strip leading "1. " or "- " or "• " prefixes
        cleaned = line.strip().lstrip("0123456789.").strip()

        if cleaned and len(cleaned) > 5 and cleaned not in variants:
            variants.append(cleaned)

    return variants[: n + 1] # Return at most n+1 items (original + n variants)