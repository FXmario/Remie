"""Small JSON-RPC client for the Codex CLI app-server."""

import asyncio
import base64
import json
import os
import shutil
from collections.abc import AsyncIterator
from typing import Any


class CodexConnectorError(RuntimeError):
    """Raised when the Codex CLI cannot serve a request."""


def codex_binary_available(binary: str = "codex") -> bool:
    """Return whether the configured Codex executable can be resolved."""
    return bool(shutil.which(binary) or os.path.isfile(binary))


class CodexAppServer:
    """Persistent Codex app-server session bound to one Remie project."""

    def __init__(self, binary: str = "codex", home: str = "") -> None:
        self.binary = binary or "codex"
        self.home = home
        self.process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._thread_id: str | None = None
        self._cwd: str | None = None

    async def start(
        self, cwd: str, model: str = "", developer_instructions: str = ""
    ) -> None:
        async with self._start_lock:
            if self.process is not None and self.process.returncode is None:
                if self._cwd == cwd:
                    return
                await self.close()
            if not codex_binary_available(self.binary):
                raise CodexConnectorError(
                    f"Codex CLI '{self.binary}' is not installed or not on PATH. "
                    "Install it with `npm install -g @openai/codex`, then run `codex login`."
                )
            env = os.environ.copy()
            if self.home.strip():
                env["CODEX_HOME"] = os.path.expanduser(self.home.strip())
            try:
                self.process = await asyncio.create_subprocess_exec(
                    self.binary,
                    "app-server",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )
            except OSError as error:
                raise CodexConnectorError(f"Could not start Codex CLI: {error}") from error
            self._cwd = cwd
            self._thread_id = None
            self._reader_task = asyncio.create_task(self._read_stdout())
            asyncio.create_task(self._drain_stderr())
            await self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "remie",
                        "title": "Remie",
                        "version": "0.1.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self._notify("initialized", {})
            account = await self._request("account/read", {})
            if not account.get("account") and account.get("requiresOpenaiAuth"):
                raise CodexConnectorError(
                    "Codex CLI is not authenticated. Run `codex login`, then reconnect."
                )
            thread = await self._request(
                "thread/start",
                {
                    "cwd": cwd,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    **({"model": model} if model else {}),
                    **(
                        {"developerInstructions": developer_instructions}
                        if developer_instructions
                        else {}
                    ),
                },
            )
            self._thread_id = ((thread.get("thread") or {}).get("id"))
            if not self._thread_id:
                raise CodexConnectorError("Codex did not return a thread id")

    async def stream(
        self,
        conversation: list[dict[str, Any]],
        model: str = "",
        reasoning_effort: str = "medium",
        usage_box: dict[str, int] | None = None,
        reasoning_box: list[str] | None = None,
        finish_box: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        await self.start(os.getcwd(), model, _system_prompt(conversation))
        if self._thread_id is None:
            raise CodexConnectorError("Codex thread was not initialized")
        text, images = _latest_input(conversation)
        input_items: list[dict[str, Any]] = []
        if text:
            input_items.append({"type": "text", "text": text})
        input_items.extend(images)
        if not input_items:
            input_items.append({"type": "text", "text": "Continue."})
        await self._request(
            "turn/start",
            {
                "threadId": self._thread_id,
                "input": input_items,
                **({"effort": reasoning_effort} if reasoning_effort != "off" else {}),
            },
        )
        while True:
            event = await self._notifications.get()
            method = event.get("method", "")
            params = event.get("params") or {}
            if method in {"item/agentMessage/delta", "item/reasoning/textDelta", "item/reasoning/summaryTextDelta"}:
                delta = params.get("delta", "")
                if method == "item/agentMessage/delta":
                    if delta:
                        yield delta
                elif delta and reasoning_box is not None:
                    reasoning_box.append(delta)
            elif method == "thread/tokenUsage/updated":
                _update_usage(params, usage_box)
            elif method == "error":
                message = _event_message(params) or "Codex returned an error"
                raise CodexConnectorError(message)
            elif method == "turn/completed":
                turn = params.get("turn") or {}
                status = turn.get("status", "completed")
                if status not in {"completed", "interrupted"}:
                    raise CodexConnectorError(
                        (turn.get("error") or {}).get("message", f"Codex turn {status}")
                    )
                if finish_box is not None:
                    finish_box["finish_reason"] = "stop"
                    finish_box["stream_complete"] = True
                    finish_box["truncated"] = False
                return
            elif method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
                request_id = event.get("id")
                if request_id is not None:
                    await self._respond(request_id, {"decision": "decline"})

    async def close(self) -> None:
        process = self.process
        self.process = None
        self._thread_id = None
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(CodexConnectorError("Codex process closed"))
        self._pending.clear()

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                raw = await self.process.stdout.readline()
                if not raw:
                    if self.process.returncode not in (None, 0):
                        error = await self._stderr_text()
                        self._fail_pending(error or "Codex process exited unexpectedly")
                    return
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if "id" in message and ("result" in message or "error" in message):
                    future = self._pending.pop(message["id"], None)
                    if future is not None and not future.done():
                        if "error" in message:
                            future.set_exception(
                                CodexConnectorError(_event_message(message["error"]) or "Codex RPC error")
                            )
                        else:
                            future.set_result(message.get("result") or {})
                elif "method" in message:
                    await self._notifications.put(message)
        except asyncio.CancelledError:
            return

    async def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        while await self.process.stderr.readline():
            pass

    async def _stderr_text(self) -> str:
        return ""

    def _fail_pending(self, message: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(CodexConnectorError(message))
        self._pending.clear()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError as error:
            self._pending.pop(request_id, None)
            raise CodexConnectorError(f"Codex request '{method}' timed out") from error

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _respond(self, request_id: Any, result: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _send(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise CodexConnectorError("Codex process is not running")
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        async with self._write_lock:
            self.process.stdin.write(data)
            await self.process.stdin.drain()


def _event_message(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("message", "detail", "error"):
            if isinstance(value.get(key), str):
                return value[key]
    return ""


def _update_usage(params: dict[str, Any], usage_box: dict[str, int] | None) -> None:
    if usage_box is None:
        return
    usage = params.get("tokenUsage") or params.get("usage") or params
    last = usage.get("last") if isinstance(usage, dict) else None
    values = last if isinstance(last, dict) else usage
    if not isinstance(values, dict):
        return
    for target, keys in {
        "prompt_tokens": ("inputTokens", "prompt_tokens"),
        "completion_tokens": ("outputTokens", "completion_tokens"),
    }.items():
        for key in keys:
            if isinstance(values.get(key), int):
                usage_box[target] = values[key]
                break


def _latest_input(conversation: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    message = conversation[-1] if conversation else {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
            image = part.get("image_url")
            if isinstance(image, dict) and isinstance(image.get("url"), str):
                url = image["url"]
                if url.startswith("data:"):
                    url = url
                images.append({"type": "image", "url": url})
    return "\n".join(texts), images


def _system_prompt(conversation: list[dict[str, Any]]) -> str:
    if conversation and conversation[0].get("role") == "system":
        content = conversation[0].get("content", "")
        if isinstance(content, str):
            return content
    return "You are the assistant. Respond helpfully and concisely."
