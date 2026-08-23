"""Tests for the curl-based web_fetch / web_search tools."""

import base64
import json
import subprocess
from types import SimpleNamespace

import pytest

from remie.tools import (
    TOOL_REGISTRY,
    TOOL_SUMMARIES,
    get_tool_schemas,
    get_tool_summary,
)
from remie.tools.web import (
    WEB_BODY_MAX_CHARS,
    WebToolError,
    _build_curl_command,
    _clean_bing_href,
    _clean_ddg_href,
    _parse_meta,
    html_to_text,
    web_fetch_tool,
    web_search_tool,
)


def _fake_curl(stdout="", stderr="", returncode=0):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return fake_run


def _patch_curl(monkeypatch, stdout="", stderr="", returncode=0):
    monkeypatch.setattr(
        "remie.tools.web.subprocess.run",
        _fake_curl(stdout, stderr, returncode),
        raising=False,
    )


META = "\n<<<REMIE_HTTP_META>>>200\thttps://example.com/final\thtml<<<END_REMIE_HTTP_META>>>"


class TestBuildCurlCommand:
    def test_basic_get(self):
        command = _build_curl_command("https://example.com")
        assert command[0] == "curl"
        assert "-L" in command
        assert "--max-time" in command
        assert command[-1] == "https://example.com"
        assert "-X" not in command  # GET is the default, no explicit flag

    def test_method_flag(self):
        command = _build_curl_command("https://example.com", method="post")
        assert command[command.index("-X") + 1] == "POST"

    def test_headers_are_passed(self):
        command = _build_curl_command(
            "https://example.com", headers={"Authorization": "Bearer tok"}
        )
        header_values = [
            command[i + 1] for i, part in enumerate(command) if part == "-H"
        ]
        assert "Authorization: Bearer tok" in header_values

    def test_header_with_colon_in_name_rejected(self):
        with pytest.raises(WebToolError):
            _build_curl_command("https://example.com", headers={"a:b": "x"})

    def test_data_sent_as_body_for_post(self):
        command = _build_curl_command("https://example.com", method="POST", data="a=1")
        assert command[command.index("--data-binary") + 1] == "a=1"

    def test_data_merged_into_query_for_get(self):
        command = _build_curl_command("https://example.com/x", data='{"q": "hi"}')
        assert command[-1].startswith("https://example.com/x?")
        assert "q=hi" in command[-1]

    def test_save_to_uses_output_flag(self):
        command = _build_curl_command("https://example.com", output_path="/tmp/f.bin")
        assert command[command.index("-o") + 1] == "/tmp/f.bin"


class TestParseMeta:
    def test_parses_status_url_and_type(self):
        body, status, url, ctype = _parse_meta(f"hello{META}")
        assert body == "hello"
        assert status == 200
        assert url == "https://example.com/final"
        assert ctype == "html"

    def test_missing_marker_returns_raw_stdout(self):
        assert _parse_meta("plain") == ("plain", 0, "", "")

    def test_content_type_charset_stripped(self):
        _, _, _, ctype = _parse_meta(
            "\n<<<REMIE_HTTP_META>>>200\tu\ttext/html; charset=utf-8<<<END_REMIE_HTTP_META>>>"
        )
        assert ctype == "text/html"


class TestHtmlToText:
    def test_strips_tags_and_scripts(self):
        page = (
            "<html><head><style>b{}</style><script>evil()</script></head>"
            "<body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"
        )
        text = html_to_text(page)
        assert "Title" in text
        assert "Hello world" in text
        assert "evil" not in text

    def test_unescapes_entities_and_collapses_blank_lines(self):
        text = html_to_text("<p>A &amp; B</p>\n\n\n\n<p>C</p>")
        assert "A & B" in text
        assert "\n\n\n" not in text


class TestWebFetchTool:
    def test_html_page_converted_to_readable_text(self, monkeypatch):
        page = "<html><head><title>Example</title></head><body><p>Fine print</p></body></html>"
        _patch_curl(monkeypatch, stdout=page + META)
        result = web_fetch_tool("http://example.com/page.html")
        assert result["status_code"] == 200
        assert result["mode"] == "readable_text"
        assert result["final_url"] == "https://example.com/final"
        assert result["body"].startswith("Example")
        assert "Fine print" in result["body"]

    def test_non_html_returned_raw(self, monkeypatch):
        meta = "\n<<<REMIE_HTTP_META>>>200\tu\tapplication/json<<<END_REMIE_HTTP_META>>>"
        payload = json.dumps({"ok": True})
        _patch_curl(monkeypatch, stdout=payload + meta)
        result = web_fetch_tool("https://api.example.com/v1/data")
        assert result["mode"] == "raw"
        assert json.loads(result["body"]) == {"ok": True}

    def test_rejects_non_http_scheme(self):
        for url in ("file:///etc/passwd", "ftp://host/x", "example.com"):
            result = web_fetch_tool(url)
            assert "error" in result

    def test_http_error_includes_snippet(self, monkeypatch):
        meta = "\n<<<REMIE_HTTP_META>>>404\tu\ttext/html<<<END_REMIE_HTTP_META>>>"
        _patch_curl(monkeypatch, stdout="<p>gone</p>" + meta)
        result = web_fetch_tool("https://example.com/missing")
        assert result["status_code"] == 404
        assert result["error"].startswith("HTTP 404")

    def test_timeout_reported(self, monkeypatch):
        _patch_curl(monkeypatch, returncode=28)  # curl operation timed out
        result = web_fetch_tool("https://slow.example.com")
        assert result.get("timed_out") is True or "timed out" in result["error"].lower()

    def test_curl_failure_surfaces_stderr(self, monkeypatch):
        _patch_curl(monkeypatch, stderr="curl: (6) no dns", returncode=6)
        result = web_fetch_tool("https://nope.example.com")
        assert "curl exited with code 6" in result["error"]
        assert "no dns" in result["error"]

    def test_save_to_writes_file(self, tmp_path, monkeypatch):
        target = tmp_path / "out.bin"
        meta = "\n<<<REMIE_HTTP_META>>>200\tu\tapplication/octet-stream<<<END_REMIE_HTTP_META>>>"
        _patch_curl(
            monkeypatch, stdout=b"\x00\x01".decode("latin-1") + meta
        )
        result = web_fetch_tool(
            "https://example.com/file.bin", save_to=str(target)
        )
        assert result["saved_to"] == str(target.resolve())
        assert result["bytes_saved"] >= 2

    def test_body_truncated_at_limit(self, monkeypatch):
        long_body = "x" * (WEB_BODY_MAX_CHARS + 5000)
        meta = "\n<<<REMIE_HTTP_META>>>200\tu\ttext/plain<<<END_REMIE_HTTP_META>>>"
        _patch_curl(monkeypatch, stdout=long_body + meta)
        result = web_fetch_tool("https://example.com/big.txt")
        assert result["truncated"] is True
        assert len(result["body"]) < WEB_BODY_MAX_CHARS + 100

    def test_dispatch_via_run_tool(self, monkeypatch):
        from remie.agent import run_tool

        meta = "\n<<<REMIE_HTTP_META>>>200\tu\ttext/plain<<<END_REMIE_HTTP_META>>>"
        _patch_curl(monkeypatch, stdout="hi there" + meta)
        result = run_tool("web_fetch", {"url": "https://example.com/hello.txt"})
        assert "hi there" in result["body"]


DDG_PAGE = """
<html><body>
<div class="result">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.com%2Fintro">Intro <b>Guide</b></a>
<a class="result__snippet" href="">The quick start for everyone.</a>
</div>
<div class="result">
<a class="result__a" href="https://dup.example.com">Dup</a>
<a class="result__snippet" href="">Snippet two</a>
</div>
<div class="result">
<a class="result__a" href="https://dup.example.com">Dup again</a>
</div>
</body></html>
"""


def _bing_page():
    target = "https://real.example.com/page"
    token = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    return f"""
<html><body>
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;u=a1{token}">Bing Result</a></h2>
<p class="b_lineclamp4">A useful snippet.</p></li>
<li class="b_algo"><h2><a href="/relative">Local link</a></h2></li>
</body></html>
"""


class TestCleanHrefs:
    def test_ddg_uddg_redirect_resolved(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa"
        assert _clean_ddg_href(href) == "https://example.com/a"

    def test_bing_base64_link_decoded(self):
        target = "https://real.example.com/page"
        token = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        assert (
            _clean_bing_href(f"https://www.bing.com/ck/a?!&u=a1{token}") == target
        )

    def test_bing_plain_link_unchanged(self):
        assert _clean_bing_href("https://plain.example.com") == "https://plain.example.com"


class TestWebSearchTool:
    def test_empty_query_rejected(self):
        assert "error" in web_search_tool("   ")

    def test_ddg_results_parsed_and_deduped(self, monkeypatch):
        meta = "\n<<<REMIE_HTTP_META>>>200\tu\thtml<<<END_REMIE_HTTP_META>>>"
        _patch_curl(monkeypatch, stdout=DDG_PAGE + meta)
        result = web_search_tool("anything", max_results=5)
        assert result["backend"] == "duckduckgo"
        urls = [item["url"] for item in result["results"]]
        assert len(urls) == len(set(urls))  # deduped
        first = result["results"][0]
        assert first["url"] == "https://docs.example.com/intro"
        assert "Guide" in first["title"]
        assert "quick start" in first["snippet"]

    def test_falls_back_to_bing_when_ddg_unreachable(self, monkeypatch):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if any("duckduckgo" in part for part in command):
                raise subprocess.TimeoutExpired(command, 20)
            page = _bing_page()
            meta = "\n<<<REMIE_HTTP_META>>>200\tu\thtml<<<END_REMIE_HTTP_META>>>"
            return SimpleNamespace(returncode=0, stdout=page + meta, stderr="")

        monkeypatch.setattr(
            "remie.tools.web.subprocess.run", fake_run, raising=False
        )
        result = web_search_tool("test query", max_results=3)
        assert result["backend"] == "bing"
        top = result["results"][0]
        assert top["url"] == "https://real.example.com/page"
        assert top["title"] == "Bing Result"
        assert "useful snippet" in top["snippet"]

    def test_all_backends_down_reports_error(self, monkeypatch):
        _patch_curl(monkeypatch, returncode=7)  # connection refused
        result = web_search_tool("query")
        assert "error" in result

    def test_max_results_clamped(self, monkeypatch):
        meta = "\n<<<REMIE_HTTP_META>>>200\tu\thtml<<<END_REMIE_HTTP_META>>>"
        _patch_curl(monkeypatch, stdout=DDG_PAGE + meta)
        result = web_search_tool("q", max_results=999)
        assert result["count"] <= 10

    def test_dispatch_via_run_tool(self, monkeypatch):
        from remie.agent import run_tool

        meta = "\n<<<REMIE_HTTP_META>>>200\tu\thtml<<<END_REMIE_HTTP_META>>>"
        _patch_curl(monkeypatch, stdout=DDG_PAGE + meta)
        result = run_tool("web_search", {"query": "hello"})
        assert result["count"] >= 1


class TestRegistration:
    def test_registered_with_summaries(self):
        assert TOOL_REGISTRY["web_fetch"].__name__ == "web_fetch_tool"
        assert TOOL_REGISTRY["web_search"].__name__ == "web_search_tool"
        assert get_tool_summary("web_fetch") == "fetch a URL over HTTP(S) with curl"
        assert get_tool_summary("web_search") == "search the web with DuckDuckGo"

    def test_native_schemas_present(self):
        names = {schema["name"] for schema in get_tool_schemas()}
        assert {"web_fetch", "web_search"} <= names
        fetch_schema = next(s for s in get_tool_schemas() if s["name"] == "web_fetch")
        assert set(fetch_schema["parameters"]["properties"]) == {
            "url",
            "method",
            "headers",
            "data",
            "save_to",
        }
        assert fetch_schema["parameters"]["required"] == ["url"]
