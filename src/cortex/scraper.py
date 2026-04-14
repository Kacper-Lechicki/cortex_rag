import httpx

from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from .config import config


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
_NOISE_TAGS = ["nav", "header", "footer", "script", "style", "aside", "form"]

# Candidate tags for main content, in order of preference
_CONTENT_SELECTORS = [
    "main",
    "article",
    "[role=main]",
    "#content",
    "#main",
    # Common CMS/blog containers
    ".entry-content",
    ".post-content",
    ".article-content",
    ".content",
    ".page-content",
    ".blog-content",
    ".blog-post",
    ".post",
    ".article",
    ".main-content",
    ".wp-block-post-content",
    "body",
]


def _host_allowed(host: str) -> bool:
    host = (host or "").strip().lower().strip(".")

    if not host:
        return False

    allowed = [
        d.strip().lower().strip(".")
        for d in (config.allowed_domains or [])
        if d.strip()
    ]

    if not allowed:
        return False

    if host in allowed:
        return True

    if config.allow_subdomains:
        return any(host.endswith("." + d) for d in allowed)

    return False


def _resolve_host_ips(host: str) -> list[str]:
    import socket

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return []

    ips: list[str] = []

    for family, _, _, _, sockaddr in infos:
        if family == socket.AF_INET:
            ip = sockaddr[0]
        elif family == socket.AF_INET6:
            ip = sockaddr[0]
        else:
            continue

        ips.append(ip)

    return sorted(set(ips))


def _is_private_or_local_ip(ip_str: str) -> bool:
    import ipaddress

    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True

    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_target_url(url: str) -> httpx.URL:
    try:
        u = httpx.URL(url)
    except Exception as e:  # noqa: BLE001
        raise ScraperError(f"Invalid URL: {url}") from e

    if u.scheme not in {"http", "https"}:
        raise ScraperError(f"Blocked URL scheme: {u.scheme!r} ({url})")

    if not u.host:
        raise ScraperError(f"Invalid URL (missing host): {url}")

    host = u.host.strip().lower().strip(".")

    if host in {"localhost", "localhost.localdomain"}:
        raise ScraperError(f"Blocked host: {host} ({url})")

    if not _host_allowed(host):
        allowed = ", ".join(config.allowed_domains or []) or "(empty)"

        raise ScraperError(
            f"Blocked by allowlist. Host={host!r}, ALLOWED_DOMAINS={allowed}"
        )

    if u.port is not None and u.port not in {80, 443}:
        raise ScraperError(f"Blocked non-standard port: {u.port} ({url})")

    ips = _resolve_host_ips(host)

    if not ips:
        raise ScraperError(f"Could not resolve host: {host}")

    if config.deny_private_ips:
        blocked = [ip for ip in ips if _is_private_or_local_ip(ip)]

        if blocked:
            raise ScraperError(
                f"Blocked private/local IP resolution for host={host!r}: {blocked}"
            )

    return u


def _fetch_with_safe_redirects(client: httpx.Client, url: str) -> httpx.Response:
    current = _validate_target_url(url)

    for i in range(config.max_redirects + 1):
        try:
            r = client.get(str(current))
        except httpx.RequestError as e:
            raise ScraperError(f"Network error when fetching: {current}") from e

        if r.status_code in {301, 302, 303, 307, 308}:
            loc = r.headers.get("Location") or r.headers.get("location")

            if not loc:
                raise ScraperError(f"Redirect without Location header: {current}")

            if i >= config.max_redirects:
                raise ScraperError(
                    f"Too many redirects (>{config.max_redirects}) starting from: {url}"
                )

            next_url = urljoin(str(current), loc)
            current = _validate_target_url(next_url)

            continue

        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ScraperError(
                f"HTTP {e.response.status_code} when fetching: {current}"
            ) from e

        return r

    raise ScraperError(f"Too many redirects starting from: {url}")


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
    with httpx.Client(
        timeout=30.0,
        follow_redirects=False,  # manual redirects with per-hop validation
        headers={"User-Agent": "CortexRag/1.0 (educational project)"},
    ) as client:
        response = _fetch_with_safe_redirects(client, url)

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Remove noise tags in-place (modifies the soup tree)
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Find main content area by selecting the *largest* candidate block.
    # Some sites match multiple selectors (or have a small <main> wrapper);
    # picking the first match often yields nav snippets instead of the article.
    best_text = ""

    for selector in _CONTENT_SELECTORS:
        for node in soup.select(selector):
            text = node.get_text(separator="\n", strip=True)
            
            if len(text) > len(best_text):
                best_text = text

    content = best_text
    content = _clean_text(content)
    domain = urlparse(str(response.url)).netloc

    return Article(url=str(response.url), title=title, content=content, domain=domain)


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
