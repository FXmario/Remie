"""Web fetching and search tools built on curl.

Both tools shell out to ``curl`` as an argv list (never ``shell=True``), so
URLs, headers, and bodies cannot inject shell syntax. Responses are capped in
size and truncated before they reach the model's context window.
"""

import html
import os
import re
import subprocess
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from remie.tools.common import resolve_abs_path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _timeout() -> int:
    """Seconds before a web request is killed (REMIE_WEB_TIMEOUT override)."""
    return _env_int("REMIE_WEB_TIMEOUT", 20)


#: Hard cap on how many bytes curl may download (--max-filesize).
WEB_MAX_BYTES = 2_000_000

#: Maximum characters of a response body returned to the model.
WEB_BODY_MAX_CHARS = 30_000

#: Maximum number of search results returned per web_search call.
SEARCH_RESULT_LIMIT = 10

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Remie coding agent"
)

# Marker written by curl's -w flag after the response body on stdout. Parsed
# out of the combined output to recover status code / final URL / type.
_META_OPEN = "\n<<<REMIE_HTTP_META>>>"
_META_CLOSE = "<<<END_REMIE_HTTP_META>>>"

_ALLOWED_SCHEMES = ("http", "https")


class WebToolError(Exception):
    """Raised for malformed requests (bad scheme, bad header, ...) ."""


def _validate_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "(none)"
    if scheme not in _ALLOWED_SCHEMES:
        raise WebToolError(
            f"Unsupported URL scheme '{scheme}': only "
            "http:// and https:// are allowed."
        )
    if not parsed.netloc:
        raise WebToolError(f"URL has no host: {url!r}")
    return url.strip()


def _build_curl_command(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str = "",
    output_path: str | None = None,
) -> list[str]:
    """Assemble a curl invocation as an argv list."""
    command = [
        "curl",
        "-sS",
        "-L",
        "--max-time",
        str(_timeout()),
        "--max-filesize",
        str(WEB_MAX_BYTES),
        "--compressed",
        "-A",
        USER_AGENT,
        "-H",
        "Accept: text/html,application/json,text/plain;q=0.9,*/*;q=0.8",
        "-w",
        f"{_META_OPEN}%{{http_code}}\t%{{url_effective}}\t%{{content_type}}{_META_CLOSE}",
    ]
    method = method.upper()
    if method != "GET":
        command += ["-X", method]
    for name, value in (headers or {}).items():
        if ":" in name or "\n" in f"{name}{value}":
            raise WebToolError(f"Invalid header name or value: {name!r}")
        command += ["-H", f"{name}: {value}"]
    if data:
        if method in ("GET", "HEAD"):
            # Merge body parameters into the query string instead.
            separator = "&" if urlparse(url).query else "?"
            url = url + separator + urlencode(_parse_query_pairs(data))
        else:
            command += ["--data-binary", data]
    if output_path:
        command += ["-o", output_path]
    command.append(url)
    return command


def _parse_query_pairs(data: str) -> dict[str, str]:
    """Best-effort conversion of a JSON-ish/k=v body into query pairs."""
    import json

    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    pairs: dict[str, str] = {}
    for chunk in data.split("&"):
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            pairs[key] = value
        elif chunk:
            pairs[chunk] = ""
    return pairs or {"q": data}


def _truncate(text: str, limit: int = WEB_BODY_MAX_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n[output truncated]", True


def _run_curl(command: list[str]) -> tuple[int, str, str]:
    """Run curl and return (exit_code, stdout, stderr), handling timeouts."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_timeout() + 5,
            input=None,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        stderr = (error.stderr or b"").decode("utf-8", errors="replace") if isinstance(
            error.stderr, bytes
        ) else (error.stderr or "")
        return 124, "", stderr or "[request timed out]"


def _parse_meta(stdout: str) -> tuple[str, int, str, str]:
    """Split the -w meta marker off curl's stdout."""
    start = stdout.rfind(_META_OPEN)
    if start < 0:
        return stdout, 0, "", ""
    head = stdout[:start]
    meta = stdout[start + len(_META_OPEN) :]
    end = meta.find(_META_CLOSE)
    if end >= 0:
        meta = meta[:end]
    parts = meta.split("\t")
    status_code = int(parts[0]) if parts and parts[0].isdigit() else 0
    final_url = parts[1] if len(parts) > 1 else ""
    content_type = parts[2].split(";")[0].strip() if len(parts) > 2 else ""
    return head, status_code, final_url, content_type


# --- HTML to readable text ---------------------------------------------------

_SCRIPT_RE = re.compile(r"(?is)<(script|style|noscript|template|svg)[^>]*>.*?</\1>")
_COMMENT_RE = re.compile(r"(?is)<!--.*?-->")
_BLOCK_TAG_RE = re.compile(
    r"(?i)</?(?:p|div|section|article|header|footer|nav|aside|main|ul|ol|dl|li"
    r"|dt|dd|table|thead|tbody|tr|td|th|h[1-6]|br|hr|blockquote|pre|form|figure"
    r"|figcaption|option|select|textarea|button|label)\b[^>]*>"
)
_ANY_TAG_RE = re.compile(r"(?s)<[^>]+>")
_TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")


def html_to_text(page: str) -> str:
    """Reduce an HTML document to readable plain text using stdlib only."""
    page = _SCRIPT_RE.sub("", page)
    page = _COMMENT_RE.sub("", page)
    page = _BLOCK_TAG_RE.sub("\n", page)
    page = _ANY_TAG_RE.sub("", page)
    page = html.unescape(page)
    lines = []
    for raw_line in page.splitlines():
        line = re.sub(r"[ \t\xa0]+", " ", raw_line).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_title(page: str) -> str:
    match = _TITLE_RE.search(page)
    if not match:
        return ""
    return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()


def _looks_like_html(content_type: str, url: str) -> bool:
    if "html" in content_type.lower():
        return True
    return "html" not in content_type.lower() and urlparse(url).path.endswith(
        (".htm", ".html")
    )


# --- Tools -------------------------------------------------------------------


def web_fetch_tool(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str = "",
    save_to: str = "",
) -> dict[str, Any]:
    """
    Fetch a URL over HTTP(S) with curl and return the response.

    :param url: Absolute http:// or https:// URL to fetch.
    :param method: HTTP method (GET, POST, PUT, DELETE, HEAD). Defaults to GET.
    :param headers: Optional extra request headers, e.g. {"Authorization": "Bearer ..."}.
    :param data: Request body. For GET it is merged into the query string;
        otherwise sent as the request body.
    :param save_to: Optional project-relative path to save the raw response body
        to instead of returning it inline (useful for downloads).
    :return: A dictionary with status_code, final_url, content_type and either
        the (text-extracted for HTML) body or the path the body was saved to.
    """
    try:
        url = _validate_url(url)
    except WebToolError as error:
        return {"error": str(error)}

    output_path = ""
    if save_to:
        full_path = resolve_abs_path(save_to)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = str(full_path)

    try:
        command = _build_curl_command(url, method, headers, data, output_path)
    except WebToolError as error:
        return {"error": str(error)}

    exit_code, stdout, stderr = _run_curl(command)
    body, status_code, final_url, content_type = _parse_meta(stdout)

    # 124: killed by our subprocess timeout; 28: curl's own --max-time.
    if exit_code in (124, 28):
        return {
            "error": (
                f"Request timed out after {_timeout()}s. Try again with a "
                "smaller scope or raise REMIE_WEB_TIMEOUT."
            ),
            "timed_out": True,
        }
    if exit_code != 0:
        detail = stderr.strip().replace("\n", "; ")
        return {
            "error": f"curl exited with code {exit_code}: {detail}",
            "status_code": status_code,
        }

    if output_path:
        saved_size = len(body.encode("utf-8"))
        return {
            "url": url,
            "status_code": status_code,
            "final_url": final_url or url,
            "content_type": content_type,
            "saved_to": output_path,
            "bytes_saved": saved_size,
            "truncated": False,
        }

    if status_code >= 400:
        snippet, _ = _truncate(body, 2000)
        return {
            "error": f"HTTP {status_code} from {final_url or url}",
            "status_code": status_code,
            "content_type": content_type,
            "body": snippet,
        }

    mode = "raw"
    if not save_to and _looks_like_html(content_type, final_url or url):
        title = _extract_title(body)
        body = html_to_text(body)
        if title:
            body = f"{title}\n{'=' * min(len(title), 60)}\n{body}"
        mode = "readable_text"

    body, truncated = _truncate(body)
    return {
        "url": url,
        "status_code": status_code,
        "final_url": final_url or url,
        "content_type": content_type,
        "mode": mode,
        "body": body,
        "truncated": truncated,
    }


# --- Search backends ---------------------------------------------------------
#
# Two keyless HTML search endpoints via curl: DuckDuckGo's html endpoint
# (preferred) and Bing (fallback for networks where DuckDuckGo is blocked).

_DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
_BING_ENDPOINT = "https://www.bing.com/search?q="

_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]+href="(?P<href>[^"]+)"'
    r"[^>]*>(?P<title>.*?)</a>",
    re.DOTALL,
)
_RESULT_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_BING_RESULT_RE = re.compile(
    r"<h2[^>]*>\s*<a[^>]+href=\"(?P<href>[^\"]+)\"[^>]*>(?P<title>.*?)</a>",
    re.DOTALL,
)
_BING_SNIPPET_RE = re.compile(
    r"<p[^>]+class=\"b_lineclamp[^\"]*\"[^>]*>(?P<snippet>.*?)</p>", re.DOTALL
)


def _strip_tags(markup: str) -> str:
    text = _ANY_TAG_RE.sub("", markup)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _clean_ddg_href(href: str) -> str:
    """Resolve DuckDuckGo redirect links ('//duckduckgo.com/l/?uddg=...')."""
    if "uddg=" in href:
        query = parse_qs(urlparse(href).query)
        targets = query.get("uddg")
        if targets:
            return targets[0]
    if href.startswith("//"):
        return "https:" + href
    return href


def _clean_bing_href(href: str) -> str:
    """Decode Bing click-tracking links (ck/a?...&u=a1<base64-url>)."""
    href = html.unescape(href)
    match = re.search(r"[?&]u=a1([A-Za-z0-9+/=_-]+)", href)
    if match:
        import base64

        encoded = match.group(1)
        encoded += "=" * (-len(encoded) % 4)
        try:
            decoded = base64.urlsafe_b64decode(encoded).decode("utf-8", errors="replace")
            if decoded.startswith(("http://", "https://")):
                return decoded
        except Exception:
            pass
    return href


def _collect_results(
    page: str,
    link_re: re.Pattern[str],
    snippet_re: re.Pattern[str],
    clean_href,
    max_results: int,
) -> list[dict[str, Any]]:
    """Pair up result titles/links with snippets, de-duplicated."""
    titles = [
        (clean_href(match.group("href")), _strip_tags(match.group("title")))
        for match in link_re.finditer(page)
    ]
    snippets = [_strip_tags(m.group("snippet")) for m in snippet_re.finditer(page)]

    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for index, (href, title) in enumerate(titles):
        if not href.startswith(("http://", "https://")):
            continue
        if href in seen_urls:
            continue
        seen_urls.add(href)
        results.append(
            {
                "title": title,
                "url": href,
                "snippet": snippets[index] if index < len(snippets) else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def _curl_page(
    url: str, post_fields: list[str] | None = None
) -> tuple[int, int, str]:
    """Fetch a page with curl; returns (exit_code, status_code, body)."""
    command = [
        "curl",
        "-sS",
        "-L",
        "--max-time",
        str(_timeout()),
        "--compressed",
        "-A",
        USER_AGENT,
        "-w",
        f"{_META_OPEN}%{{http_code}}{_META_CLOSE}",
    ]
    if post_fields:
        command += ["--data-urlencode", *post_fields]
    command.append(url)

    exit_code, stdout, stderr = _run_curl(command)
    body, status_code = _parse_meta(stdout)[:2]
    return exit_code, status_code, body


def _search_ddg(query: str, max_results: int) -> list[dict[str, Any]] | None:
    """DuckDuckGo backend. Returns results, [] when blocked, or None on failure."""
    exit_code, status_code, page = _curl_page(_DDG_ENDPOINT, [f"q={query}"])
    if exit_code != 0:
        return None  # unreachable — caller should try the next backend
    results = _collect_results(
        page, _RESULT_LINK_RE, _RESULT_SNIPPET_RE, _clean_ddg_href, max_results
    )
    if results:
        return results
    # A 200 page without results means genuinely no hits; anything else is
    # usually a block/captcha page, worth retrying on another backend.
    return [] if status_code == 200 else None


def _search_bing(query: str, max_results: int) -> list[dict[str, Any]] | None:
    """Bing backend. Same contract as _search_ddg."""
    from urllib.parse import quote_plus

    exit_code, status_code, page = _curl_page(_BING_ENDPOINT + quote_plus(query))
    if exit_code != 0:
        return None
    results = _collect_results(
        page, _BING_RESULT_RE, _BING_SNIPPET_RE, _clean_bing_href, max_results
    )
    if results:
        return results
    return [] if status_code == 200 else None


def web_search_tool(query: str, max_results: int = SEARCH_RESULT_LIMIT) -> dict[str, Any]:
    """
    Search the web via curl using DuckDuckGo's HTML endpoint, falling back to
    Bing when DuckDuckGo is unreachable; no API key needed.

    :param query: The search query text.
    :param max_results: Maximum number of results to return (default 10).
    :return: A dictionary with the query, backend used, and a list of {title, url, snippet}.
    """
    query = query.strip()
    if not query:
        return {"error": "Empty search query."}
    max_results = max(1, min(int(max_results), SEARCH_RESULT_LIMIT))

    last_error = ""
    for backend_name, backend in (("duckduckgo", _search_ddg), ("bing", _search_bing)):
        try:
            results = backend(query, max_results)
        except Exception as error:  # defensive: never crash the agent loop
            results = None
            last_error = f"{type(error).__name__}: {error}"
        if results is None:
            continue
        payload: dict[str, Any] = {
            "query": query,
            "backend": backend_name,
            "results": results,
            "count": len(results),
        }
        if not results and last_error:
            payload["note"] = f"previous backend error: {last_error}"
        return payload

    return {
        "error": (
            "No search backend reachable "
            f"(last error: {last_error or 'blocked or empty results'})."
        )
    }
