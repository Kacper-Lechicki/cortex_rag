import httpx

from dataclasses import dataclass
from urllib.parse import urlparse
from bs4 import BeautifulSoup


# --- Custom Exception ---
# By subclassing Exception we create our own error type.
# This lets callers do: except ScraperError as e: ...
# instead of catching all possible exceptions blindly.
class ScraperError(Exception):
    """
    Raised when an article cannot be fetched or parsed.
    """
    pass


@dataclass
class Article:
    """
    Represents a scraped article.

    Attributes:
        url: Original URL of the article.
        title: Page title extracted from <title> tag.
        content: Clean text content (HTML stripped).
        domain: Domain name extracted from URL (e.g. 'docs.python.org')
    """

    url: str
    title: str
    content: str
    domain: str


    def __repr__(self) -> str:
        # __repr__ defines how the object looks when you print() it or inspect in REPL.
        preview = self.content[:60].replace("\n", " ")
        return f"Article(domain={self.domain!r}, title={self.title!r}, preview={preview!r})"


# Tags that contain navigation, ads, footers - not useful content
_NOISE_TAGS = [
    "nav", "header", "footer", "script", "style",
    "aside", "form"
]

# Candidate tags for main content, in order of preference
_CONTENT_SELECTORS = [
    "main", "article", "[role=main]", "#content", "#main",
    "body"
]


def scrape_article(url: str) -> Article:
    """
    Fetch a URL and extract clean article text.

    Uses httpx as an HTTP client and BeautifulSoup for HTML parsing.
    Removed navigation, scripts and other noise before extracting.

    Args:
        url: The URL to fetch.

    Returns:
        An Article dataclass with title, content and domain.

    Raises:
        ScraperError: If the request fails or returns a non-200 status.
    """

    # Context manager: httpx.Client is created, used, then closed automatically.
    # Even if an exception occurs inside the 'with' block, the client closes properly.
    try:
        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            header={"User-Agent": "CortexRag/1.0 (educational project)"}
        ) as client:
            response = client.get(url)
            # raise_for_status() raises httpx.HTTPStatusError for 4xx/5xx responses
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        # 'raise X from e' preserves the original exception as the __cause__
        # This gives a better traceback when debugging
        raise ScraperError(
            f"HTTP {e.response.status_code} when fetching: {url}"
        ) from e
    except httpx.RequestError as e:
        raise ScraperError(f"Network error when fetching: {url}") from e

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Remove noise tags in-place (modifies the soup tree)
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Find main content area using a priority list of selectors
    main_content = None

    for selector in _CONTENT_SELECTORS:
        main_content = soup.select_one(selector)

        if main_content:
            break

    # get_text() extracts all text nodes, separator="\n" preserves paragraphs
    content = main_content.get_text(separator="\n", strip=True) if main_content else ""
    content = _clean_text(content)

    domain = urlparse(url).netloc

    return Article(url=url, title=title, content=content, domain=domain)


def _clean_text(text: str) -> str:
    """
    Remove excessive whitespace from scraped text.

    Collapses multiple blank lines into single blank lines and strips leading/trailing whitespace.
    """

    import re

    # Replace 3+ consecutive newlines with exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just whitespace
    lines = [line.rstrip() for line in text.split("\n")]

    return "\n".join(lines).strip()
