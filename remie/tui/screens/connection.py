"""Connection/provider/model picker modal."""

import asyncio

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Switch

from remie import codex_auth
from remie.agent import (
    CODEX_BACKEND_BASE,
    CODEX_MODELS,
    OPENROUTER_MODELS,
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_MODELS,
    ConnectionConfig,
    configure_openai,
    fetch_codex_models,
    fetch_openrouter_models,
    get_config,
    load_provider_configs,
    save_provider_configs,
    supports_reasoning_effort,
)
from remie.model_names import ModelInfo
from remie.tui.constants import REASONING_EFFORTS, PROVIDER_BASE_URLS
from remie.tui.helpers import _coerce_model_info, _model_option
from remie.tui.widgets import ModelBadge

import remie.tui as _tui_pkg


class ConnectionScreen(ModalScreen):
    """Modal to select a provider and connect to the LLM API."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._stashed_effort: str | None = None
        self._profiles = load_provider_configs()
        current = get_config()
        self._profiles[current.provider] = current
        self._active_provider = current.provider
        # Master option lists per Select id; the visible (filtered) options
        # are derived from these whenever a search input changes.
        self._select_masters: dict[str, list[tuple[Text, str]]] = {}
        # Value tokens for filter-driven Select syncs: when a search filter
        # rebuilds options, the value we programmatically assign is recorded
        # here so the resulting (queued) Select.Changed is recognized as a
        # programmatic sync instead of a user action — deterministic no matter
        # when the queued message runs.
        self._select_tokens: dict[str, str | None] = {}

    CSS = """
    ConnectionScreen {
        align: center middle;
    }

    #connection-dialog {
        width: 60;
        height: 24;
        max-height: 90%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #connection-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    #connection-dialog Label {
        margin-top: 1;
    }

    #connection-dialog .row {
        height: 3;
        width: 100%;
        align: center middle;
    }

    #connection-dialog .filter-row {
        height: 3;
        width: 100%;
    }

    #connection-dialog .filter-row Select {
        width: 1fr;
        margin-right: 1;
    }

    #connection-dialog .filter-row Input {
        width: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        current = self._profiles[self._active_provider]
        with Vertical(id="connection-dialog"):
            with VerticalScroll(id="connection-scroll"):
                yield Label("Connection", id="dialog-title")
                with Horizontal(id="provider-filter-row", classes="filter-row"):
                    yield Select(
                        [
                            ("Local (llama.cpp)", "local"),
                            ("OpenCode Go", "opencode-go"),
                            ("Codex (ChatGPT Plus/Pro)", "codex"),
                            ("OpenRouter", "openrouter"),
                        ],
                        value=self._active_provider,
                        id="provider-select",
                        prompt="Choose provider...",
                    )
                    yield Input(
                        placeholder="Filter providers…",
                        id="provider-search",
                    )
                yield Label("Base URL", id="base-url-label")
                yield Input(
                    current.base_url,
                    placeholder="http://localhost:7070/v1",
                    id="base-url-input",
                )
                yield Label("API Key", id="api-key-label")
                yield Input(
                    current.api_key,
                    password=True,
                    placeholder="API key",
                    id="api-key-input",
                )
                yield Label("ChatGPT account", id="codex-account-label")
                yield Horizontal(
                    Button(
                        "Sign in with ChatGPT", variant="primary", id="codex-signin-button"
                    ),
                    Button("Sign out", id="codex-signout-button"),
                    classes="row",
                )
                yield Label("Model")
                if current.provider == "codex":
                    model_list = list(CODEX_MODELS)
                elif current.provider == "openrouter":
                    model_list = list(OPENROUTER_MODELS)
                else:
                    model_list = list(OPENCODE_GO_MODELS)
                if (
                    current.provider != "local"
                    and current.model
                    and current.model not in model_list
                ):
                    model_list = [current.model] + model_list
                with Horizontal(id="model-filter-row", classes="filter-row"):
                    yield Select(
                        [_model_option(model) for model in model_list],
                        value=(
                            current.model
                            if current.model in model_list
                            else model_list[0]
                        ),
                        id="model-select",
                        prompt="Select model...",
                    )
                    yield Input(
                        placeholder="Filter models (name or id)…",
                        id="model-search",
                    )
                yield Input(
                    current.model if current.provider == "local" else "",
                    placeholder="Enter the local model name",
                    id="local-model-input",
                )
                yield Label("Reasoning effort", id="reasoning-effort-label")
                with Horizontal(id="reasoning-filter-row", classes="filter-row"):
                    yield Select(
                        [(effort.title(), effort) for effort in REASONING_EFFORTS],
                        value=current.reasoning_effort
                        if current.reasoning_effort in REASONING_EFFORTS
                        else "medium",
                        id="reasoning-effort-select",
                        prompt="Select reasoning effort...",
                    )
                    yield Input(
                        placeholder="Filter efforts…",
                        id="reasoning-search",
                    )
                yield Label("Verify local SSL certificates", id="verify-ssl-label")
                yield Switch(
                    value=current.verify_ssl,
                    id="verify-ssl-switch",
                    animate=False,
                )
            with Horizontal(classes="row"):
                yield Button("Submit", variant="primary", id="submit-button")
                yield Button("Cancel", id="cancel-button")

    def on_mount(self) -> None:
        provider = self.query_one("#provider-select", Select).value
        # Seed the master option lists for the searchable dropdowns.
        self._store_options(
            "provider-select",
            [
                ("Local (llama.cpp)", "local"),
                ("OpenCode Go", "opencode-go"),
                ("Codex (ChatGPT Plus/Pro)", "codex"),
                ("OpenRouter", "openrouter"),
            ],
        )
        self._store_options("reasoning-effort-select", list(REASONING_EFFORTS))
        self._set_provider_fields(provider)
        self._update_reasoning_fields()
        if provider == "local":
            self.query_one("#api-key-input", Input).focus()
        if provider == "opencode-go":
            api_key = self.query_one("#api-key-input", Input).value.strip()
            if api_key:
                self.run_worker(
                    self._refresh_models(api_key, str(provider)), exclusive=False
                )
        if provider == "codex":
            self._refresh_codex_models()
            self.run_worker(self._prefetch_codex_models(), exclusive=False)
        if provider == "openrouter":
            self._refresh_openrouter_models()
            self.run_worker(self._prefetch_openrouter_models(), exclusive=False)

    def _store_options(
        self, select_id: str, models: "list[str | ModelInfo | tuple[Text, str]]"
    ) -> None:
        """Remember the full option list for a Select; filtering narrows a
        copy of it. Accepts raw ids, ModelInfo rows, or prebuilt options."""
        master: list[tuple[Text, str]] = []
        for model in models:
            if (
                isinstance(model, tuple)
                and len(model) == 2
                and isinstance(model[1], str)
            ):
                master.append(model)
            else:
                master.append(_model_option(model))
        self._select_masters[select_id] = master

    def _apply_select_filter(self, select_id: str, query: str) -> None:
        """Narrow a Select to master entries matching the query (matches both
        the pretty label and the raw value); keeps the current selection when
        it still matches."""
        master = self._select_masters.get(select_id)
        if master is None:
            return
        select = self.query_one(f"#{select_id}", Select)
        previous = select.value
        q = query.strip().lower()
        if q:
            filtered = [
                (label, value)
                for label, value in master
                if q in value.lower() or q in label.plain.lower()
            ]
        else:
            filtered = master
        # Programmatic value syncs during filtering must not trigger the
        # screen's Select.Changed side effects; record the assigned value as
        # this Select's token so the queued Changed is recognized.
        select.set_options(filtered)
        values = [value for _, value in filtered]
        assigned: str | None = None
        if isinstance(previous, str) and previous in values:
            assigned = previous
        elif values:
            assigned = values[0]
        self._select_tokens[select_id] = assigned
        if assigned is not None:
            select.value = assigned

    def _search_input_for(self, select_id: str) -> Input | None:
        search_ids = {
            "provider-select": "provider-search",
            "model-select": "model-search",
            "reasoning-effort-select": "reasoning-search",
        }
        search_id = search_ids.get(select_id)
        if not search_id:
            return None
        try:
            return self.query_one(f"#{search_id}", Input)
        except Exception:
            return None

    def _current_query(self, select_id: str) -> str:
        search_input = self._search_input_for(select_id)
        if search_input is None:
            return ""
        return search_input.value or ""

    def _set_model_options(self, models: "list[str | ModelInfo]") -> None:
        """Replace the model dropdown's master list and re-apply any active
        filter, keeping the profile's model selected when it survives."""
        self._store_options("model-select", models)
        self._apply_select_filter("model-select", self._current_query("model-select"))

    def _codex_account_text(self) -> str:
        auth = codex_auth.load_auth()
        if auth is None:
            return "Not signed in — your ChatGPT Plus/Pro plan signs in via the browser."
        return f"Signed in: {codex_auth.account_summary(auth)}"

    def _update_codex_account_label(self) -> None:
        self.query_one("#codex-account-label", Label).update(
            f"ChatGPT account — {self._codex_account_text()}"
        )

    def _refresh_codex_models(self) -> None:
        profile = self._profiles.get("codex")
        fallback = list(CODEX_MODELS)
        if profile is not None and profile.model and profile.model not in fallback:
            fallback.insert(0, profile.model)
        current_value = (
            profile.model
            if profile is not None and profile.model in fallback
            else fallback[0]
        )
        self._set_model_options(fallback)
        select = self.query_one("#model-select", Select)
        values = [value for _, value in self._select_masters.get("model-select", [])]
        if current_value in values:
            select.value = current_value

    async def _prefetch_codex_models(self) -> None:
        """Replace the bundled Codex list with the account's live models."""
        if not codex_auth.is_signed_in():
            return
        try:
            models = await fetch_codex_models()
        except Exception:
            return
        if not models or not self.is_running:
            return
        select = self.query_one("#model-select", Select)
        previously_selected = str(select.value)
        options = [model for model in models]
        if previously_selected and previously_selected not in [
            _coerce_model_info(model).id for model in options
        ]:
            options.insert(0, previously_selected)
        self._set_model_options(options)
        ids = [_coerce_model_info(model).id for model in options]
        select.value = (
            previously_selected if previously_selected in ids else ids[0]
        )
        self._update_codex_account_label()

    async def _run_codex_signin(self, button: Button) -> None:
        button.disabled = True
        button.label = "Waiting for browser..."
        self.notify(
            "Opening the browser to sign in to ChatGPT. Complete the sign-in in "
            "the browser tab; Remie continues automatically.",
            title="ChatGPT sign-in",
            timeout=10,
        )

        def show_login_url(url: str) -> None:
            self.app.call_from_thread(
                self.notify,
                f"If the browser did not open, visit:\n{url}",
                title="ChatGPT sign-in URL",
                severity="information",
                timeout=20,
            )

        try:
            auth = await asyncio.to_thread(
                lambda: asyncio.run(codex_auth.login(on_login_url=show_login_url))
            )
        except codex_auth.CodexAuthError as error:
            self.notify(str(error), title="ChatGPT sign-in failed", severity="error")
        except Exception as error:  # noqa: BLE001 — surface any failure to the user
            self.notify(
                f"{type(error).__name__}: {error}",
                title="ChatGPT sign-in failed",
                severity="error",
            )
        else:
            self._profiles["codex"] = ConnectionConfig(
                CODEX_BACKEND_BASE,
                "",
                str(self.query_one("#model-select", Select).value),
                "codex",
                self.query_one("#reasoning-effort-select", Select).value
                if isinstance(
                    self.query_one("#reasoning-effort-select", Select).value, str
                )
                else "medium",
                True,
            )
            self.notify(
                f"Signed in as {codex_auth.account_summary(auth)}",
                title="ChatGPT connected",
            )
            await self._prefetch_codex_models()
        finally:
            if self.is_running:
                button = self.query_one("#codex-signin-button", Button)
                button.disabled = False
                button.label = "Sign in with ChatGPT"
            self._update_codex_account_label()

    def _sign_out_codex(self) -> None:
        removed = codex_auth.clear_auth()
        message = (
            "Signed out of ChatGPT; tokens were removed."
            if removed
            else "No stored ChatGPT tokens found."
        )
        self.notify(message, title="Codex (ChatGPT)")
        self._update_codex_account_label()

    def _refresh_openrouter_models(self) -> None:
        profile = self._profiles.get("openrouter")
        fallback = list(OPENROUTER_MODELS)
        if profile is not None and profile.model and profile.model not in fallback:
            fallback.insert(0, profile.model)
        current_value = (
            profile.model
            if profile is not None and profile.model in fallback
            else fallback[0]
        )
        self._set_model_options(fallback)
        select = self.query_one("#model-select", Select)
        values = [value for _, value in self._select_masters.get("model-select", [])]
        if current_value in values:
            select.value = current_value

    async def _prefetch_openrouter_models(self) -> None:
        """Replace the bundled OpenRouter list with the live catalog (public
        endpoint, works before an API key is entered)."""
        try:
            import remie.tui as _tui_pkg

            models = await _tui_pkg.fetch_openrouter_models()
        except Exception:
            return
        if not models or not self.is_running:
            return
        select = self.query_one("#model-select", Select)
        previously_selected = str(select.value)
        ids = [_coerce_model_info(model).id for model in models]
        if previously_selected and previously_selected in ids:
            options: "list[str | ModelInfo]" = models
        else:
            options = (
                [previously_selected] + models if previously_selected else models
            )
            ids.insert(0, previously_selected)
        self._set_model_options(options)
        select.value = (
            previously_selected if previously_selected in ids else ids[0]
        )
        self._update_reasoning_fields()

    def _set_provider_fields(self, provider: object) -> None:
        is_local = provider == "local"
        is_codex = provider == "codex"
        has_provider = provider in {"local", "opencode-go", "codex", "openrouter"}
        base_url_input = self.query_one("#base-url-input", Input)
        base_url_label = self.query_one("#base-url-label", Label)
        base_url_input.display = is_local
        base_url_label.display = is_local
        base_url_input.disabled = not is_local
        api_key_input = self.query_one("#api-key-input", Input)
        api_key_input.display = not is_codex
        api_key_input.disabled = is_codex
        api_key_label = self.query_one("#api-key-label", Label)
        api_key_label.display = not is_codex
        model_select = self.query_one("#model-select", Select)
        model_filter_row = self.query_one("#model-filter-row", Horizontal)
        local_model_input = self.query_one("#local-model-input", Input)
        model_filter_row.display = has_provider and not is_local
        model_select.display = has_provider and not is_local
        model_select.disabled = not has_provider or is_local
        local_model_input.display = is_local
        local_model_input.disabled = not is_local
        reasoning_select = self.query_one("#reasoning-effort-select", Select)
        reasoning_label = self.query_one("#reasoning-effort-label", Label)
        reasoning_select.display = has_provider
        reasoning_label.display = has_provider
        verify_label = self.query_one("#verify-ssl-label", Label)
        verify_switch = self.query_one("#verify-ssl-switch", Switch)
        verify_label.display = is_local
        verify_switch.display = is_local
        verify_switch.disabled = not is_local
        account_label = self.query_one("#codex-account-label", Label)
        signin_button = self.query_one("#codex-signin-button", Button)
        signout_button = self.query_one("#codex-signout-button", Button)
        account_label.display = is_codex
        signin_button.display = is_codex
        signout_button.display = is_codex
        signin_button.disabled = not is_codex
        signout_button.disabled = not is_codex
        if is_codex:
            self._update_codex_account_label()
        # Search inputs mirror their dropdown's visibility.
        model_search = self.query_one("#model-search", Input)
        model_search.display = has_provider and not is_local
        provider_search = self.query_one("#provider-search", Input)
        provider_search.display = True
        reasoning_search = self.query_one("#reasoning-search", Input)
        reasoning_search.display = has_provider
        reasoning_search.disabled = not has_provider

    _SEARCH_TARGETS = {
        "provider-search": "provider-select",
        "model-search": "model-select",
        "reasoning-search": "reasoning-effort-select",
    }

    def on_input_changed(self, event: Input.Changed) -> None:
        select_id = self._SEARCH_TARGETS.get(event.input.id or "")
        if select_id:
            event.stop()
            self._apply_select_filter(select_id, event.value or "")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit modal fields without leaking them to the chat input."""
        event.stop()
        if (event.input.id or "") in self._SEARCH_TARGETS:
            # Enter inside a filter box jumps to its dropdown instead of
            # submitting the whole form.
            select_id = self._SEARCH_TARGETS[event.input.id or ""]
            try:
                self.query_one(f"#{select_id}", Select).focus()
            except Exception:
                pass
            return
        self._connect()

    def _capture_profile(self) -> None:
        """Keep edits made to the current provider before switching away."""
        provider = self._active_provider
        model = (
            self.query_one("#local-model-input", Input).value.strip()
            if provider == "local"
            else self._selected_model()
        )
        effort = self.query_one("#reasoning-effort-select", Select).value
        if not isinstance(effort, str):
            effort = "medium"
        self._profiles[provider] = ConnectionConfig(
            self.query_one("#base-url-input", Input).value.strip()
            if provider == "local"
            else PROVIDER_BASE_URLS[provider],
            self.query_one("#api-key-input", Input).value.strip(),
            model,
            provider,
            effort,
            self.query_one("#verify-ssl-switch", Switch).value
            if provider == "local"
            else True,
        )

    def _apply_profile(self, provider: str) -> None:
        profile = self._profiles[provider]
        self.query_one("#base-url-input", Input).value = profile.base_url
        self.query_one("#api-key-input", Input).value = profile.api_key
        self.query_one("#local-model-input", Input).value = profile.model
        reasoning = self.query_one("#reasoning-effort-select", Select)
        reasoning.value = profile.reasoning_effort
        self.query_one("#verify-ssl-switch", Switch).value = profile.verify_ssl

    def _update_reasoning_fields(self, selected_model: str | None = None) -> None:
        """Enable or fade the reasoning-effort picker for the selected model.

        Models that don't accept `reasoning_effort` get the effort snapped to
        "off" and the control disabled (dimmed); the prior effort is stashed
        and restored when a supported model is selected again.
        """
        model = selected_model or self._selected_model()
        provider = self.query_one("#provider-select", Select).value
        if provider not in {"local", "opencode-go", "codex", "openrouter"}:
            return
        supported = supports_reasoning_effort(model, provider)
        select = self.query_one("#reasoning-effort-select", Select)
        label = self.query_one("#reasoning-effort-label", Label)
        if supported:
            select.disabled = False
            label.disabled = False
            if self._stashed_effort is not None:
                select.value = self._stashed_effort
                self._stashed_effort = None
        else:
            if select.value != "off":
                self._stashed_effort = select.value
            select.value = "off"
            select.disabled = True
            label.disabled = True

    async def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""
        if self._select_tokens.get(select_id) == event.value:
            # Programmatic sync from the search filter, not a user action.
            return
        self._select_tokens[select_id] = (
            event.value if isinstance(event.value, str) else None
        )
        if select_id == "provider-select":
            self._capture_profile()
            self._active_provider = str(event.value)
            self._apply_profile(self._active_provider)
            self._set_provider_fields(event.value)
            if event.value == "codex":
                self._refresh_codex_models()
                self.run_worker(self._prefetch_codex_models(), exclusive=False)
            elif event.value == "openrouter":
                self._refresh_openrouter_models()
                self.run_worker(self._prefetch_openrouter_models(), exclusive=False)
            elif event.value in PROVIDER_BASE_URLS:
                self.query_one("#base-url-input", Input).value = PROVIDER_BASE_URLS[
                    event.value
                ]
                fallback_models: "list[str | ModelInfo]" = list(OPENCODE_GO_MODELS)
                profile = self._profiles[self._active_provider]
                if profile.model and profile.model not in fallback_models:
                    fallback_models.insert(0, profile.model)
                self._set_model_options(fallback_models)
                model_select = self.query_one("#model-select", Select)
                model_select.value = profile.model or fallback_models[0]
                api_key = self.query_one("#api-key-input", Input).value.strip()
                if api_key:
                    await self._refresh_models(api_key, str(event.value))
        if event.select.id in {"provider-select", "model-select"}:
            selected_model = (
                str(event.value) if event.select.id == "model-select" else None
            )
            self._update_reasoning_fields(selected_model)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss()
            return
        if event.button.id == "codex-signin-button":
            await self._run_codex_signin(event.button)
        elif event.button.id == "codex-signout-button":
            self._sign_out_codex()
        elif event.button.id == "submit-button":
            self._connect()

    async def _refresh_models(self, api_key: str, provider: str = "opencode-go") -> None:
        select = self.query_one("#model-select", Select)
        previously_selected = select.value
        select.loading = True
        models: "list[str | ModelInfo]" = await _tui_pkg.fetch_opencode_go_models(
            api_key
        )
        select.loading = False
        self._set_model_options(models)
        # Keep the user's selection when it is still offered by the live list;
        # only fall back to the first model when it is gone.
        ids = [_coerce_model_info(model).id for model in models]
        if isinstance(previously_selected, str) and previously_selected in ids:
            select.value = previously_selected
        elif ids:
            select.value = ids[0]
        self._update_reasoning_fields()

    def _connect(self) -> None:
        provider = self.query_one("#provider-select", Select).value
        if provider not in {"local", "opencode-go", "codex", "openrouter"}:
            self.notify("Choose a provider first", severity="warning")
            return
        if provider == "codex":
            base_url = CODEX_BACKEND_BASE
            api_key = ""
            model = self._selected_model()
        elif provider != "local":
            base_url = PROVIDER_BASE_URLS.get(str(provider), OPENCODE_GO_BASE_URL)
            api_key = self.query_one("#api-key-input", Input).value.strip()
            model = self._selected_model()
            if not api_key:
                self.notify(
                    f"Enter your {str(provider).replace('-', ' ').title()} API key",
                    title="Missing API key",
                    severity="error",
                )
                return
        else:
            base_url = self.query_one("#base-url-input", Input).value.strip()
            api_key = self.query_one("#api-key-input", Input).value.strip()
            model = self._selected_model()
            if not model:
                self.notify("Enter the local model name", severity="error")
                return
        verify_ssl = (
            self.query_one("#verify-ssl-switch", Switch).value
            if provider == "local"
            else True
        )
        effort = self.query_one("#reasoning-effort-select", Select).value
        if not isinstance(effort, str):
            effort = "medium"
        if not supports_reasoning_effort(model, str(provider)):
            effort = "off"
        if provider == "codex" and not codex_auth.is_signed_in():
            self.notify(
                "Sign in with ChatGPT before connecting.",
                title="Not signed in",
                severity="error",
            )
            return
        config = configure_openai(
            base_url,
            api_key,
            model,
            provider=str(provider),
            reasoning_effort=effort,
            verify_ssl=verify_ssl,
        )
        import remie.tui as _tui_pkg

        self._profiles[str(provider)] = config
        _tui_pkg.save_provider_configs(self._profiles, str(provider))
        app = self.app
        if isinstance(app, AgentApp):
            app.query_one(ModelBadge).update_config(config)
        self.dismiss()
        from remie.agent import get_model_info

        self.app.notify(
            f"Connected to {get_model_info(model).resolved_display()}",
            title="Connection updated",
        )

    def _selected_model(self) -> str:
        provider = self.query_one("#provider-select", Select).value
        if provider == "local":
            return self.query_one("#local-model-input", Input).value.strip()
        value = self.query_one("#model-select", Select).value
        if isinstance(value, str):
            return value
        fallbacks = {
            "codex": CODEX_MODELS,
            "openrouter": OPENROUTER_MODELS,
        }
        return fallbacks.get(str(provider), OPENCODE_GO_MODELS)[0]


from remie.tui import _agent_app_registry as _registry

_registry.register_module(__name__)
