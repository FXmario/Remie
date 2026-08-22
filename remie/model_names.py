"""Pretty display names for model ids.

Raw model ids (``stealth/ox-alpha``, ``dots-studio/dots-3-note-preview:free``)
are what APIs accept and configs store, but dropdowns should show readable
names. Catalogs that ship display metadata (OpenRouter ``name``, Codex
``display_name``) are preferred; everything else goes through
:func:`prettify_model_id`, a heuristic splitter with brand/vendor maps.
"""

import re
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """A model option as shown in pickers; ``id`` stays the stored value."""

    id: str
    display: str = ""
    vendor: str = ""
    free: bool = False

    def resolved_display(self) -> str:
        return self.display or prettify_model_name(self.id)


# Tokens rendered as-is or upper-cased rather than title-cased.
ACRONYMS = {
    "ai": "AI",
    "gpt": "GPT",
    "glm": "GLM",
    "llm": "LLM",
    "mcp": "MCP",
    "ocr": "OCR",
    "vl": "VL",
}

VENDORS = {
    "~z-ai": "Z.AI",
    "z-ai": "Z.AI",
    "x-ai": "xAI",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "meta": "Meta",
    "mistralai": "Mistral AI",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "tencent": "Tencent",
    "moonshotai": "Moonshot AI",
    "minimax": "MiniMax",
    "nvidia": "NVIDIA",
    "microsoft": "Microsoft",
    "amazon": "Amazon",
    "cohere": "Cohere",
    "perplexity": "Perplexity",
    "openrouter": "OpenRouter",
    "stealth": "Stealth",
    "dots-studio": "Dots Studio",
    "opencode": "OpenCode Zen",
}

# Brand tokens that need inner capitalization when they appear in a name.
BRAND_TOKENS = {
    "deepseek": "DeepSeek",
    "minimax": "MiniMax",
    "glm": "GLM",
    "gpt": "GPT",
    "ai": "AI",
    "hy": "HY",
    "mi": "Mi",
    "mo": "Mo",
}

_VERSIONISH = re.compile(r"^(\d.*|v\d.*|[a-z]{1,2}\d[\w.]*)$", re.IGNORECASE)


def prettify_token(token: str) -> str:
    """Title-case one token, keeping acronyms and version-ish tokens intact."""
    lowered = token.lower()
    if lowered in ACRONYMS:
        return ACRONYMS[lowered]
    if lowered in BRAND_TOKENS:
        return BRAND_TOKENS[lowered]
    if _VERSIONISH.match(token):
        # 1.8b -> 1.8B, v4 -> V4, mt2 -> MT2? keep digits, upper the letters.
        return token.upper() if any(c.isalpha() for c in token) else token
    return token[:1].upper() + token[1:]


def prettify_model_name(name: str) -> str:
    """Humanize a bare model name/id without vendor prefix handling."""
    cleaned = name.strip().strip("~")
    tokens = re.split(r"[-_]+", cleaned)
    return " ".join(prettify_token(t) for t in tokens if t).strip()


def split_vendor(model_id: str) -> tuple[str, str]:
    """Split 'vendor/model' into (vendor_label, bare_name); tilde-tolerant."""
    cleaned = model_id.strip()
    vendor = ""
    if "/" in cleaned:
        raw_vendor, bare = cleaned.split("/", 1)
        vendor_key = raw_vendor.strip().lower()
        vendor = VENDORS.get(vendor_key) or prettify_model_name(raw_vendor)
        return vendor, bare.strip()
    return "", cleaned


def prettify_model_id(model_id: str) -> ModelInfo:
    """Best-effort ModelInfo for catalogs that only expose raw ids."""
    vendor, bare = split_vendor(model_id)
    free = False
    name_part = bare
    if name_part.lower().endswith(":free"):
        free = True
        name_part = name_part[: -len(":free")]
    display = prettify_model_name(name_part)
    return ModelInfo(id=model_id, display=display, vendor=vendor, free=free)
