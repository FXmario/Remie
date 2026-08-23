"""Provider model discovery, display metadata, and context limits."""

import os

import httpx

from remie.config import (
    CODEX_MODELS,
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_MODELS,
    OPENROUTER_MODELS,
)
from remie.model_names import ModelInfo, prettify_model_id, prettify_model_name

CODEX_DEFAULT_CONTEXT_LIMIT = 272_000
OPENROUTER_DEFAULT_CONTEXT_LIMIT = 128_000
OPENCODE_GO_DEFAULT_CONTEXT_LIMIT = 128_000

NON_REASONING_EFFORT_MODELS = {
    "grok-4.5",
    "gpt-5.6-luna",
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "qwen3.8-max",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.5-plus",
}


def supports_reasoning_effort(model: str, provider: str = "local") -> bool:
    if provider != "opencode-go":
        return True
    return model not in NON_REASONING_EFFORT_MODELS


def get_max_output_tokens(provider: str = "local") -> int:
    env_value = os.environ.get("REMIE_MAX_OUTPUT_TOKENS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return 32_768 if provider in ("opencode-go", "openrouter") else 8_192


def get_model_info(model_id: str, info_cache: dict[str, ModelInfo]) -> ModelInfo:
    return info_cache.get(model_id) or prettify_model_id(model_id)


def _cache(info: ModelInfo, info_cache: dict[str, ModelInfo]) -> ModelInfo:
    if info.id:
        info_cache[info.id] = info
    return info


async def fetch_opencode_go_models(
    api_key: str,
    context_cache: dict[str, int],
    info_cache: dict[str, ModelInfo],
) -> list[ModelInfo]:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(
                f"{OPENCODE_GO_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
            infos: list[ModelInfo] = []
            for item in payload.get("data", []):
                model_id = item.get("id")
                if not model_id:
                    continue
                context = item.get("context_length")
                if isinstance(context, int) and context > 0:
                    context_cache[model_id] = context
                infos.append(_cache(prettify_model_id(str(model_id)), info_cache))
            if infos:
                return infos
        except httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError:
            pass
    return [_cache(prettify_model_id(m), info_cache) for m in OPENCODE_GO_MODELS]


async def fetch_codex_models(
    context_cache: dict[str, int],
    info_cache: dict[str, ModelInfo],
) -> list[ModelInfo]:
    from remie.codex_client import fetch_codex_models as fetch_live

    try:
        rows = await fetch_live()
    except Exception:
        rows = []
    if not rows:
        return [
            _cache(
                ModelInfo(id=m, display=prettify_model_name(m), vendor="OpenAI"),
                info_cache,
            )
            for m in CODEX_MODELS
        ]
    infos: list[ModelInfo] = []
    for row in rows:
        context = row.get("context_window") or 0
        if isinstance(context, int) and context > 0:
            context_cache[row["id"]] = context
        infos.append(
            _cache(
                ModelInfo(
                    id=row["id"],
                    display=row.get("display") or row["id"],
                    vendor="OpenAI",
                ),
                info_cache,
            )
        )
    return infos


async def fetch_openrouter_models(
    context_cache: dict[str, int],
    info_cache: dict[str, ModelInfo],
) -> list[ModelInfo]:
    from remie.openrouter_client import fetch_openrouter_models as fetch_live

    try:
        rows = await fetch_live()
    except Exception:
        rows = []
    if not rows:
        return [_cache(prettify_model_id(m), info_cache) for m in OPENROUTER_MODELS]
    infos: list[ModelInfo] = []
    for row in rows:
        context = row.get("context_length") or 0
        if isinstance(context, int) and context > 0:
            context_cache[row["id"]] = context
        infos.append(
            _cache(
                ModelInfo(
                    id=row["id"],
                    display=row.get("display") or row["id"],
                    vendor=row.get("vendor") or "",
                    free=bool(row.get("free")),
                ),
                info_cache,
            )
        )
    return infos


def get_model_context_limit(
    model: str,
    provider: str,
    opencode_context: dict[str, int],
    openrouter_context: dict[str, int],
    _codex_context: dict[str, int],
) -> int | None:
    if provider == "codex":
        return CODEX_DEFAULT_CONTEXT_LIMIT
    if provider == "openrouter":
        return openrouter_context.get(model, OPENROUTER_DEFAULT_CONTEXT_LIMIT)
    if provider != "opencode-go":
        return None
    return opencode_context.get(model, OPENCODE_GO_DEFAULT_CONTEXT_LIMIT)
