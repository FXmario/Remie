"""Dedicated modal for switching models on the active provider."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Select
from textual.widgets.option_list import Option

from remie.tui.effort_slider import EffortSlider
from remie import codex_auth
from remie.agent import (
    CODEX_MODELS,
    OPENROUTER_MODELS,
    OPENCODE_GO_MODELS,
    fetch_codex_models,
    fetch_opencode_go_models,
    fetch_openrouter_models,
    get_config,
    load_provider_configs,
    save_provider_configs,
    set_active_connection,
    supports_reasoning_effort,
)
from remie.model_names import ModelInfo
from remie.tui.constants import REASONING_EFFORTS
from remie.tui.contracts import is_agent_app
from remie.tui.helpers import _model_option
from remie.tui.screens.connection_services import ConnectionServices
from remie.tui.widgets import ModelBadge


class ModelScreen(ModalScreen):
    """Search and activate a model without reopening connection settings."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    CSS = """
    ModelScreen {
        align: center middle;
    }

    #model-dialog {
        width: 64;
        height: 26;
        max-width: 94%;
        max-height: 86%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #model-dialog-title {
        height: 1;
        text-style: bold;
    }

    #model-provider-label {
        height: 2;
        padding-top: 1;
        color: $text-muted;
    }

    #model-picker-search {
        margin-bottom: 1;
    }

    #model-picker-list {
        height: 1fr;
        margin-bottom: 1;
    }

    #model-reasoning-label {
        height: 1;
        color: $text-muted;
    }

    #model-reasoning-select {
        width: 100%;
        margin-bottom: 1;
    }

    #model-picker-actions {
        height: 3;
        align: right middle;
    }

    #model-picker-actions Button {
        margin-left: 1;
    }
    """

    def __init__(self, services: ConnectionServices | None = None) -> None:
        super().__init__()
        self._init_state(services)

    def _init_state(self, services: ConnectionServices | None = None) -> None:
        self._services = services or ConnectionServices(
            fetch_opencode_models=fetch_opencode_go_models,
            fetch_codex_models=fetch_codex_models,
            fetch_openrouter_models=fetch_openrouter_models,
            save_profiles=save_provider_configs,
        )
        self._config = get_config()
        self._models: list[tuple[Text, str]] = []

    @staticmethod
    def _provider_name(provider: str) -> str:
        return {
            "local": "Local",
            "opencode-go": "OpenCode Go",
            "codex": "Codex (ChatGPT)",
            "openrouter": "OpenRouter",
        }.get(provider, provider)

    def _fallback_models(self) -> list[str]:
        models = {
            "codex": CODEX_MODELS,
            "openrouter": OPENROUTER_MODELS,
            "opencode-go": OPENCODE_GO_MODELS,
        }.get(self._config.provider, [self._config.model])
        result = list(models)
        if self._config.model and self._config.model not in result:
            result.insert(0, self._config.model)
        return result

    def compose(self) -> ComposeResult:
        is_local = self._config.provider == "local"
        with Vertical(id="model-dialog"):
            yield Label("Select a model", id="model-dialog-title")
            yield Label(
                f"Provider: {self._provider_name(self._config.provider)}",
                id="model-provider-label",
            )
            yield Input(
                self._config.model if is_local else "",
                placeholder=(
                    "Enter the local model name"
                    if is_local
                    else "Filter models by name or id…"
                ),
                id="model-picker-search",
            )
            yield OptionList(id="model-picker-list")
            yield Label("Reasoning effort", id="model-reasoning-label")
            yield EffortSlider(
                value=(
                    self._config.reasoning_effort
                    if self._config.reasoning_effort in REASONING_EFFORTS
                    else "medium"
                ),
                id="model-reasoning-select",
            )
            with Horizontal(id="model-picker-actions"):
                yield Button("Cancel", id="model-picker-cancel")
                yield Button(
                    "Use model", variant="primary", id="model-picker-submit"
                )

    def on_mount(self) -> None:
        self._set_models(self._fallback_models())
        search = self.query_one("#model-picker-search", Input)
        standalone = self.parent is None or self.parent.__class__.__name__ != "TabPane"
        if standalone:
            search.focus()
        self._update_reasoning_control()
        if standalone and self._config.provider != "local":
            self.run_worker(self._load_live_models(), exclusive=False)

    def _set_models(self, models: "list[str | ModelInfo]") -> None:
        rows: list[tuple[Text, str]] = []
        seen: set[str] = set()
        for model in models:
            label, model_id = _model_option(model)
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            rows.append((label, model_id))
        if self._config.model and self._config.model not in seen:
            rows.insert(0, _model_option(self._config.model))
        self._models = rows
        self._apply_filter(self.query_one("#model-picker-search", Input).value)

    def _apply_filter(self, query: str) -> None:
        model_list = self.query_one("#model-picker-list", OptionList)
        if self._config.provider == "local":
            visible = self._models
        else:
            needle = query.strip().casefold()
            visible = [
                (label, model_id)
                for label, model_id in self._models
                if not needle
                or needle in model_id.casefold()
                or needle in label.plain.casefold()
            ]
        model_list.set_options(
            [Option(label, id=model_id) for label, model_id in visible]
        )
        if not visible:
            return
        selected_id = (
            self._config.model
            if any(model_id == self._config.model for _, model_id in visible)
            else visible[0][1]
        )
        model_list.highlighted = model_list.get_option_index(selected_id)
        self._update_reasoning_control(selected_id)

    def _update_reasoning_control(self, model: str | None = None) -> None:
        """Show reasoning effort only when the highlighted model supports it."""
        if not self.is_mounted:
            return
        selected = model or self._selected_model()
        supported = bool(selected) and supports_reasoning_effort(
            selected, self._config.provider
        )
        self.query_one("#model-reasoning-label", Label).display = supported
        self.query_one("#model-reasoning-select", EffortSlider).display = supported

    async def _load_live_models(self) -> None:
        try:
            if self._config.provider == "opencode-go":
                if not self._config.api_key:
                    return
                models = await self._services.fetch_opencode_models(
                    self._config.api_key
                )
            elif self._config.provider == "codex":
                if not codex_auth.is_signed_in():
                    return
                models = await self._services.fetch_codex_models()
            elif self._config.provider == "openrouter":
                models = await self._services.fetch_openrouter_models()
            else:
                return
        except Exception:
            # The bundled list remains usable when a catalog request fails.
            return
        if models and self.is_running:
            self._set_models(models)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-picker-search":
            event.stop()
            self._apply_filter(event.value or "")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model-picker-search":
            event.stop()
            self._select_model()

    def _selected_model(self) -> str:
        if self._config.provider == "local":
            return self.query_one("#model-picker-search", Input).value.strip()
        option = self.query_one("#model-picker-list", OptionList).highlighted_option
        return str(option.id) if option is not None and option.id is not None else ""

    def _select_model(self) -> None:
        model = self._selected_model()
        if not model:
            self.notify("Choose a model first", severity="warning")
            return
        reasoning = self.query_one("#model-reasoning-select", EffortSlider).value
        effort = reasoning if isinstance(reasoning, str) else self._config.reasoning_effort
        if not supports_reasoning_effort(model, self._config.provider):
            effort = "off"
        config = set_active_connection(
            self._config.base_url,
            self._config.api_key,
            model,
            provider=self._config.provider,
            reasoning_effort=effort,
            verify_ssl=self._config.verify_ssl,
        )
        profiles = load_provider_configs()
        profiles[config.provider] = config
        self._services.save_profiles(profiles, config.provider)
        app = self.app
        if is_agent_app(app):
            app.query_one(ModelBadge).update_config(config)
        self.dismiss()
        self.app.notify(
            f"Switched to {_model_option(model)[0].plain}", title="Model updated"
        )

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_list.id == "model-picker-list":
            option_id = event.option.id if event.option is not None else None
            self._update_reasoning_control(
                str(option_id) if option_id is not None else None
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "model-picker-list":
            event.stop()
            self._select_model()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "model-picker-cancel":
            self.dismiss()
        elif event.button.id == "model-picker-submit":
            self._select_model()
