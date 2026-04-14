import functools
import re
import time

from typing import TypeVar, Callable
import httpx
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

from .config import config
from .logging_utils import get_logger

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

            log = get_logger(__name__)
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
                        delay = base_delay * (2 ** (attempt - 1))
                        log.warning(
                            "Retryable error in %s (attempt %d/%d): %s; sleeping %.0fs",
                            func.__name__,
                            attempt,
                            max_attempts,
                            e,
                            delay,
                        )

                        time.sleep(delay)
                except httpx.HTTPError as e:
                    last_error = e

                    # Do not retry on non-retryable HTTP client errors (except 429).
                    # Retrying 400/401/403/404 etc. just adds latency and noise.
                    if isinstance(e, httpx.HTTPStatusError):
                        code = e.response.status_code

                        if 400 <= code < 500 and code != 429:
                            raise

                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        
                        log.warning(
                            "HTTP error in %s (attempt %d/%d): %s; sleeping %.0fs",
                            func.__name__,
                            attempt,
                            max_attempts,
                            e,
                            delay,
                        )

                        time.sleep(delay)

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
        provider = (config.generation_provider or "").strip()

        if provider.lower() in {"openai"}:
            raise RuntimeError(
                "HuggingFace client requested but GENERATION_PROVIDER=openai"
            )

        if not config.hf_token.strip():
            raise RuntimeError("HF_TOKEN is required when using HuggingFace providers.")

        _client = InferenceClient(
            provider=provider if provider and provider.lower() != "auto" else None,
            api_key=config.hf_token,
        )

    return _client


def _openai_chat_completion(
    messages: list[dict], max_tokens: int, temperature: float
) -> str:
    if not config.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is required when GENERATION_PROVIDER=openai")

    url = config.openai_base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": config.openai_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            url,
            headers={"Authorization": f"Bearer {config.openai_api_key}"},
            json=payload,
        )

        r.raise_for_status()
        data = r.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        keys = (
            sorted(list(data.keys())) if isinstance(data, dict) else type(data).__name__
        )

        raise RuntimeError(f"Unexpected OpenAI response shape (keys={keys})") from e


def _chat_completion(messages: list[dict], max_tokens: int, temperature: float) -> str:
    provider = (config.generation_provider or "hf-inference").strip().lower()

    if provider == "openai":
        return _openai_chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature
        )

    client = _get_client()

    response = client.chat_completion(
        model=config.generation_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


def _extractive_answer(question: str, context: str) -> str:
    """
    Heuristic fallback when a strong LLM isn't available.

    Picks a few sentences from the retrieved context based on token overlap
    with the question. This is strictly extractive (no hallucinations).
    """

    def _tokens(text: str) -> list[str]:
        # Normalize common "run-together" brand terms (e.g. HuggingFace vs Hugging Face)
        # by splitting CamelCase into words before tokenization.
        spaced = re.sub(r"([a-ząćęłńóśźż])([A-Z])", r"\1 \2", text)
        return [t for t in re.findall(r"[\wąćęłńóśźż]+", spaced.lower()) if len(t) >= 3]

    def _keys(tokens: list[str]) -> set[str]:
        # Add:
        # - full tokens (e.g. "hugging", "face")
        # - 4-char prefixes (helps minor morphology variance)
        # - bigrams concatenated (e.g. "huggingface") to bridge space/no-space variants
        out: set[str] = set(tokens)
        out |= {t[:4] for t in tokens if len(t) >= 4}
        out |= {tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)}
        return out

    q_keys = _keys(_tokens(question))

    # Split context into sentences (best-effort).
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if s.strip()
    ]

    scored: list[tuple[int, str]] = []

    for s in sentences:
        s_keys = _keys(_tokens(s))
        score = len(q_keys & s_keys)
       
        if score > 0:
            scored.append((score, s))

    if not scored:
        # If retrieval returned something but overlap scoring fails (common with
        # very short or oddly tokenized questions), fall back to a short excerpt
        # from the top chunk. Still strictly extractive, but more useful UX.
        top_chunk = context.split("\n\n---\n\n", 1)[0].strip()
        if top_chunk:
            top_sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+|\n+", top_chunk)
                if s.strip()
            ]
            excerpt = " ".join(top_sentences[:2]).strip()
            if excerpt:
                return excerpt[:800] if len(excerpt) > 800 else excerpt

        return "I don't have enough information in my knowledge base to answer this."

    scored.sort(key=lambda x: x[0], reverse=True)

    picked: list[str] = []

    for _, s in scored[:3]:
        if s not in picked:
            picked.append(s)

    answer = " ".join(picked).strip()

    return answer[:800] if len(answer) > 800 else answer


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

    if not context.strip():
        return "I don't have enough information in my knowledge base to answer this."

    # The default hf-inference model is a router and produces low-quality,
    # often ungrounded answers. In that specific setup, an extractive answer is more reliable.
    provider = (config.generation_provider or "hf-inference").strip().lower()

    if (
        provider != "openai"
        and config.generation_model.strip().lower()
        == "katanemo/arch-router-1.5b".lower()
    ):
        return _extractive_answer(question, context)

    # Keep requests small and predictable; HF inference endpoints can degrade with huge prompts.
    max_context_chars = 8_000

    context_trimmed = (
        context if len(context) <= max_context_chars else context[:max_context_chars]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise assistant for retrieval-augmented QA.\n"
                "Rules:\n"
                "- Answer using ONLY the provided context.\n"
                "- If the context is insufficient or does not contain the answer, reply exactly:\n"
                "  I don't have enough information in my knowledge base to answer this.\n"
                "- Do not guess. Do not add facts not present in the context.\n"
                "- Keep the answer short and factual.\n"
                "- When you use information from the context, include a brief Sources section at the end.\n"
                "  Sources must be URLs that appear in the context (no new links).\n"
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_trimmed}\n\nQuestion: {question}\n\nAnswer:",
        },
    ]

    insufficient = "I don't have enough information in my knowledge base to answer this."

    try:
        out = _chat_completion(messages, max_tokens=512, temperature=0.1).strip()
        if out == insufficient:
            # If the model declines but we do have retrieved text, prefer a strictly
            # extractive snippet over a hard "no answer" for reliability/UX.
            extracted = _extractive_answer(question, context).strip()
            if extracted and extracted != insufficient:
                return extracted

        return out
    except Exception as e:
        msg = str(e).lower()

        if (
            "model not supported" in msg
            or "model_not_supported" in msg
            or "not supported by provider" in msg
        ):
            return _extractive_answer(question, context)
            
        raise


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

    try:
        text = _chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Generate diverse search queries. Return ONLY numbered queries, one per line. No explanations.",
                },
                {
                    "role": "user",
                    "content": f"Generate {n} different ways to search for information about:\n'{query}'",
                },
            ],
            max_tokens=200,
            temperature=0.7,
        )
    except Exception as e:
        # Query expansion is an optimization; if it fails, we still want retrieval to work.
        log = get_logger(__name__)
        log.warning("Query expansion failed; continuing without variants: %s", e)

        return [query]

    variants: list[str] = [query]  # Always include the original query first

    for line in text.strip().split("\n"):
        # Strip leading "1. " or "- " or "• " prefixes
        cleaned = line.strip().lstrip("0123456789.").strip()

        if cleaned and len(cleaned) > 5 and cleaned not in variants:
            variants.append(cleaned)

    return variants[: n + 1]  # Return at most n+1 items (original + n variants)
