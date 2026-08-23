"""Connection profiles and user-level Remie configuration.

The store is path-injected so configuration behavior can be tested without
module-global monkeypatching. Compatibility wrappers remain in ``remie.agent``
while callers migrate to this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("REMIE_CONFIG_DIR", "~/.config/remie")).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.json"

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOCAL_BASE_URL = "http://localhost:7070/v1"
CODEX_BACKEND_BASE = "https://chatgpt.com/backend-api/codex"
CONFIG_VERSION = 2
SUPPORTED_PROVIDERS = ("local", "opencode-go", "codex", "openrouter")
STATUS_ANIMATION_CONFIG_KEY = "status_animation"

CODEX_MODELS = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
]
OPENROUTER_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.6",
    "google/gemini-3-pro",
    "deepseek/deepseek-v4",
    "x-ai/grok-5",
    "qwen/qwen4-max",
]
OPENCODE_GO_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "grok-4.5",
    "glm-5.2",
    "glm-5.1",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "hy3",
]


@dataclass
class ConnectionConfig:
    base_url: str
    api_key: str
    model: str
    provider: str = "local"
    reasoning_effort: str = "medium"
    verify_ssl: bool = False


def provider_defaults(provider: str) -> ConnectionConfig:
    if provider == "codex":
        return ConnectionConfig(
            CODEX_BACKEND_BASE, "", CODEX_MODELS[0], "codex", "medium", True
        )
    if provider == "openrouter":
        return ConnectionConfig(
            OPENROUTER_BASE_URL,
            "",
            OPENROUTER_MODELS[0],
            "openrouter",
            "medium",
            True,
        )
    if provider == "opencode-go":
        return ConnectionConfig(
            OPENCODE_GO_BASE_URL,
            "",
            OPENCODE_GO_MODELS[0],
            "opencode-go",
            "medium",
            True,
        )
    return ConnectionConfig(
        os.environ.get("LLAMA_BASE_URL", LOCAL_BASE_URL),
        os.environ.get("LLAMA_API_KEY", "llama-cpp"),
        os.environ.get("LLAMA_MODEL", "local-model"),
        "local",
        os.environ.get("REMIE_REASONING_EFFORT", "medium"),
        False,
    )


def default_config() -> ConnectionConfig:
    return ConnectionConfig(
        base_url=os.environ.get("LLAMA_BASE_URL", LOCAL_BASE_URL),
        api_key=os.environ.get("LLAMA_API_KEY", "llama-cpp"),
        model=os.environ.get("LLAMA_MODEL", "local-model"),
        provider="local",
        reasoning_effort=os.environ.get("REMIE_REASONING_EFFORT", "medium"),
        verify_ssl=False,
    )


class ConfigStore:
    """Load and save provider profiles at an explicitly supplied path."""

    def __init__(
        self,
        config_dir: Path = CONFIG_DIR,
        config_file: Path | None = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.config_file = (
            Path(config_file) if config_file else self.config_dir / "config.json"
        )

    def _read(self) -> dict:
        try:
            value = json.loads(self.config_file.read_text())
            return value if isinstance(value, dict) else {}
        except OSError, json.JSONDecodeError:
            return {}

    def load(self) -> ConnectionConfig:
        data = self._read()
        if not data:
            return default_config()
        providers = data.get("providers")
        if isinstance(providers, dict):
            active_provider = data.get("active_provider") or data.get("provider")
            if active_provider in SUPPORTED_PROVIDERS:
                profile = providers.get(active_provider, {})
                defaults = provider_defaults(active_provider)
                return ConnectionConfig(
                    profile.get("base_url", defaults.base_url),
                    profile.get("api_key", defaults.api_key),
                    profile.get("model", defaults.model),
                    active_provider,
                    profile.get("reasoning_effort", defaults.reasoning_effort),
                    bool(profile.get("verify_ssl", defaults.verify_ssl)),
                )
        base_url = data.get("base_url", "")
        provider = data.get(
            "provider",
            "opencode-go" if base_url.rstrip("/") == OPENCODE_GO_BASE_URL else "local",
        )
        if provider not in SUPPORTED_PROVIDERS:
            return provider_defaults("local")
        return ConnectionConfig(
            base_url=base_url,
            api_key=data.get("api_key", ""),
            model=data.get("model", ""),
            provider=provider,
            reasoning_effort=data.get("reasoning_effort", "medium"),
            verify_ssl=bool(data.get("verify_ssl", False)),
        )

    def load_profiles(self) -> dict[str, ConnectionConfig]:
        data = self._read()
        if not data:
            active = default_config()
            return {
                provider: provider_defaults(provider)
                for provider in SUPPORTED_PROVIDERS
            } | {active.provider: active}

        stored = data.get("providers")
        if isinstance(stored, dict):
            profiles = {
                provider: provider_defaults(provider)
                for provider in SUPPORTED_PROVIDERS
            }
            for provider in SUPPORTED_PROVIDERS:
                profile = stored.get(provider)
                if not isinstance(profile, dict):
                    continue
                defaults = profiles[provider]
                profiles[provider] = ConnectionConfig(
                    profile.get("base_url", defaults.base_url),
                    profile.get("api_key", defaults.api_key),
                    profile.get("model", defaults.model),
                    provider,
                    profile.get("reasoning_effort", defaults.reasoning_effort),
                    bool(profile.get("verify_ssl", defaults.verify_ssl)),
                )
            return profiles

        legacy = self.load()
        return {
            provider: provider_defaults(provider) for provider in SUPPORTED_PROVIDERS
        } | {legacy.provider: legacy}

    def save_profiles(
        self,
        profiles: dict[str, ConnectionConfig],
        active_provider: str,
    ) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "active_provider": active_provider,
            STATUS_ANIMATION_CONFIG_KEY: self.load_status_animation_enabled(),
            "providers": {
                provider: {
                    "base_url": config.base_url,
                    "api_key": config.api_key,
                    "model": config.model,
                    "reasoning_effort": config.reasoning_effort,
                    "verify_ssl": config.verify_ssl,
                }
                for provider, config in profiles.items()
                if provider in SUPPORTED_PROVIDERS
            },
        }
        self.config_file.write_text(json.dumps(payload, indent=2))

    def save(self, config: ConnectionConfig) -> None:
        profiles = self.load_profiles()
        profiles[config.provider] = config
        self.save_profiles(profiles, config.provider)

    def load_status_animation_enabled(self) -> bool:
        return bool(self._read().get(STATUS_ANIMATION_CONFIG_KEY, True))

    def save_status_animation_enabled(self, enabled: bool) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = self._read()
        data[STATUS_ANIMATION_CONFIG_KEY] = bool(enabled)
        self.config_file.write_text(json.dumps(data, indent=2))
