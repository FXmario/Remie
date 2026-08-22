"""ChatGPT OAuth (PKCE) sign-in for the Codex subscription backend.

Implements the same browser-based authorization flow as the Codex CLI, so
Remie can use a ChatGPT Plus/Pro subscription directly — no Codex CLI or npm
install required. Tokens are stored in ``~/.codex/auth.json`` using the same
layout the Codex CLI writes, so an existing ``codex login`` is picked up
automatically and both tools stay signed in with one set of credentials.
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

CODEX_AUTH_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
# Public client ID of the Codex CLI (PKCE flow, no client secret).
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_REDIRECT_PORT = 1455
CODEX_REDIRECT_URI = f"http://localhost:{CODEX_REDIRECT_PORT}/auth/callback"
CODEX_SCOPE = "openid profile email offline_access"
OPENAI_AUTH_CLAIM = "https://api.openai.com/auth"

LOGIN_TIMEOUT_SECONDS = 600.0
# Refresh when the access token is about to expire within this window.
TOKEN_REFRESH_SKEW_SECONDS = 300.0

_SUCCESS_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Remie — sign-in complete</title></head>"
    "<body style='font-family:sans-serif;background:#14141a;color:#eee;"
    "display:flex;align-items:center;justify-content:center;height:100vh'>"
    "<h1>Sign-in complete. You can close this tab and return to Remie.</h1>"
    "</body></html>"
)


class CodexAuthError(RuntimeError):
    """Raised when ChatGPT sign-in or token handling fails."""


@dataclass
class CodexAuth:
    access_token: str
    refresh_token: str = ""
    id_token: str = ""
    account_id: str = ""
    plan_type: str = ""
    email: str = ""
    last_refresh: float = 0.0


def auth_json_path() -> Path:
    """Token file location, matching the Codex CLI's CODEX_HOME layout."""
    home = os.environ.get("CODEX_HOME", "~/.codex")
    return Path(home).expanduser() / "auth.json"


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verifying the signature.

    Signature verification is not needed here: the token is only read back by
    the tool that received it over TLS from OpenAI, and every API request is
    itself authenticated server-side.
    """
    try:
        segment = token.split(".")[1]
        segment += "=" * (-len(segment) % 4)
        payload = json.loads(base64.urlsafe_b64decode(segment))
    except (IndexError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _claims(token: str) -> dict[str, Any]:
    payload = _decode_jwt_payload(token)
    claim = payload.get(OPENAI_AUTH_CLAIM)
    return claim if isinstance(claim, dict) else {}


def _parse_iso_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return parsed.timestamp()


def _iso_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _auth_from_tokens(
    access_token: str,
    refresh_token: str = "",
    id_token: str = "",
) -> CodexAuth:
    source = id_token or access_token
    auth_claim = _claims(source)
    payload = _decode_jwt_payload(source)
    return CodexAuth(
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
        account_id=str(auth_claim.get("chatgpt_account_id") or ""),
        plan_type=str(auth_claim.get("chatgpt_plan_type") or ""),
        email=str(payload.get("email") or ""),
        last_refresh=time.time(),
    )


def load_auth() -> CodexAuth | None:
    """Read stored ChatGPT tokens, returning None when signed out.

    Accepts both the nested Codex CLI layout and flat variants written by
    other tools that reuse the same file.
    """
    try:
        data = json.loads(auth_json_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    access = tokens.get("access_token") or data.get("access_token") or data.get("access")
    if not isinstance(access, str) or not access:
        return None
    refresh = (
        tokens.get("refresh_token") or data.get("refresh_token") or data.get("refresh")
    )
    id_token = tokens.get("id_token") or data.get("id_token") or ""
    if not isinstance(refresh, str):
        refresh = ""
    if not isinstance(id_token, str):
        id_token = ""
    auth = _auth_from_tokens(access, refresh, id_token)
    stored = data.get("last_refresh")
    if isinstance(stored, (int, float)):
        auth.last_refresh = float(stored)
    else:
        auth.last_refresh = _parse_iso_timestamp(stored)
    return auth


def save_auth(auth: CodexAuth) -> None:
    """Persist tokens in the Codex CLI's auth.json layout (atomically)."""
    path = auth_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": auth.id_token,
            "access_token": auth.access_token,
            "refresh_token": auth.refresh_token,
            "account_id": auth.account_id,
        },
        "last_refresh": _iso_timestamp(auth.last_refresh),
    }
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def clear_auth() -> bool:
    """Delete stored ChatGPT tokens. Returns True when a file was removed."""
    path = auth_json_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return True


def is_signed_in() -> bool:
    return load_auth() is not None


def generate_pkce() -> tuple[str, str]:
    """Return a (code_verifier, S256 code_challenge) pair."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def build_authorize_url(state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CODEX_CLIENT_ID,
        "redirect_uri": CODEX_REDIRECT_URI,
        "scope": CODEX_SCOPE,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "login",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{CODEX_AUTH_URL}?{urllib.parse.urlencode(params)}"


async def _token_request(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(CODEX_TOKEN_URL, json=payload)
    except httpx.HTTPError as error:
        raise CodexAuthError(f"Could not reach {CODEX_TOKEN_URL}: {error}") from error
    if response.status_code != 200:
        detail = response.text[:300] or f"HTTP {response.status_code}"
        raise CodexAuthError(f"ChatGPT token request failed: {detail}")
    try:
        data = response.json()
    except ValueError as error:
        raise CodexAuthError("ChatGPT token endpoint returned invalid JSON") from error
    if not isinstance(data.get("access_token"), str) or not data["access_token"]:
        raise CodexAuthError("ChatGPT token response did not include an access token")
    return data


async def exchange_code(code: str, code_verifier: str) -> CodexAuth:
    """Swap an OAuth authorization code for tokens and persist them."""
    data = await _token_request(
        {
            "grant_type": "authorization_code",
            "client_id": CODEX_CLIENT_ID,
            "code": code,
            "redirect_uri": CODEX_REDIRECT_URI,
            "code_verifier": code_verifier,
        }
    )
    auth = _auth_from_tokens(
        data["access_token"],
        str(data.get("refresh_token") or ""),
        str(data.get("id_token") or ""),
    )
    save_auth(auth)
    return auth


async def refresh_auth(auth: CodexAuth) -> CodexAuth:
    """Exchange the refresh token for a fresh set and persist it."""
    if not auth.refresh_token:
        raise CodexAuthError(
            "No refresh token stored. Run ChatGPT sign-in again from the "
            "connection picker (Ctrl+P)."
        )
    data = await _token_request(
        {
            "grant_type": "refresh_token",
            "client_id": CODEX_CLIENT_ID,
            "refresh_token": auth.refresh_token,
            "scope": CODEX_SCOPE,
        }
    )
    refreshed = _auth_from_tokens(
        data["access_token"],
        str(data.get("refresh_token") or auth.refresh_token),
        str(data.get("id_token") or auth.id_token),
    )
    save_auth(refreshed)
    return refreshed


def access_token_expiry(auth: CodexAuth) -> float:
    exp = _decode_jwt_payload(auth.access_token).get("exp")
    return float(exp) if isinstance(exp, (int, float)) else 0.0


async def ensure_valid_auth() -> CodexAuth:
    """Return stored auth, refreshing the access token when it nears expiry.

    Falls back to a still-valid current token if the refresh request fails.
    """
    auth = load_auth()
    if auth is None:
        raise CodexAuthError(
            "Not signed in to ChatGPT. Open the connection picker (Ctrl+P), "
            "choose Codex (ChatGPT) and press 'Sign in with ChatGPT'."
        )
    if access_token_expiry(auth) <= time.time() + TOKEN_REFRESH_SKEW_SECONDS:
        try:
            auth = await refresh_auth(auth)
        except CodexAuthError:
            if access_token_expiry(auth) > time.time():
                # Keep serving the still-valid token; retry refresh next call.
                pass
            else:
                raise
    return auth


def account_summary(auth: CodexAuth) -> str:
    plan = auth.plan_type.strip().replace("_", " ").title() or "ChatGPT"
    if auth.email:
        return f"{auth.email} · {plan}"
    return f"{plan} account"


async def login(
    timeout: float = LOGIN_TIMEOUT_SECONDS,
    on_login_url: Callable[[str], None] | None = None,
) -> CodexAuth:
    """Run the browser PKCE flow end-to-end and store the resulting tokens.

    Starts the localhost callback server, opens the default browser at the
    authorization URL (falling back to surfacing the URL via ``on_login_url``
    when no browser can be opened), waits for the redirect, then exchanges the
    authorization code.
    """
    state = secrets.token_urlsafe(24)
    verifier, challenge = generate_pkce()
    loop = asyncio.get_running_loop()
    code_future: asyncio.Future[str] = loop.create_future()

    async def handle_callback(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            try:
                target = request_line.decode("latin-1").split(" ")[1]
            except (IndexError, UnicodeDecodeError):
                target = "/"
            while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                pass  # drain request headers
            params = urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)
            if not code_future.done():
                error = (params.get("error") or [""])[0]
                if error:
                    code_future.set_exception(
                        CodexAuthError(f"Sign-in failed: {error}")
                    )
                elif (params.get("state") or [""])[0] != state:
                    code_future.set_exception(
                        CodexAuthError("OAuth state mismatch; please retry sign-in.")
                    )
                else:
                    code = (params.get("code") or [""])[0]
                    if code:
                        code_future.set_result(code)
                    else:
                        code_future.set_exception(
                            CodexAuthError("Sign-in callback had no authorization code.")
                        )
            body = _SUCCESS_HTML.encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            await writer.drain()
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            try:
                writer.close()
            except OSError:
                pass

    try:
        server = await asyncio.start_server(handle_callback, "127.0.0.1", CODEX_REDIRECT_PORT)
    except OSError as error:
        raise CodexAuthError(
            f"Port {CODEX_REDIRECT_PORT} is already in use; is another sign-in "
            "or the Codex CLI login running?"
        ) from error

    url = build_authorize_url(state, challenge)
    browser_opened = False
    try:
        try:
            browser_opened = await asyncio.to_thread(webbrowser.open, url)
        except Exception:
            browser_opened = False
        if on_login_url is not None:
            on_login_url(url)
        try:
            code = await asyncio.wait_for(asyncio.shield(code_future), timeout)
        except asyncio.TimeoutError as error:
            raise CodexAuthError(
                "ChatGPT sign-in timed out before the browser callback returned."
            ) from error
    finally:
        server.close()
        await server.wait_closed()

    auth = await exchange_code(code, verifier)
    return auth
