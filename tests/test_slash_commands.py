import asyncio

import pytest
from textual.widgets import Input, OptionList

import remie.agent as agent
import remie.tui.app as tui_app
from remie.agent import get_config
from remie.tui import (
    AgentApp,
    ChatScreen,
    ConnectionScreen,
    MemoryScreen,
    ModelBadge,
    ModelScreen,
    PromptTextArea,
    SlashCommandPopup,
    StatusIndicator,
)
from remie.tui.slash_commands import (
    is_slash_command_token,
    resolve_slash_command,
    slash_command_matches,
)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    monkeypatch.setattr(agent, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(agent, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setattr(agent, "_config", agent._default_config())

    async def no_prefetch(_api_key):
        return ["kimi-k3"]

    monkeypatch.setattr(tui_app, "fetch_opencode_go_models", no_prefetch)


def test_slash_command_registry_filters_and_resolves_trailing_slash():
    assert [command.name for command in slash_command_matches("/")] == [
        "memories",
        "chats",
        "connect",
        "models",
    ]
    assert [command.name for command in slash_command_matches("/mo")] == ["models"]
    assert slash_command_matches("explain /models") == ()
    assert slash_command_matches("/models please") == ()
    assert resolve_slash_command("/connect/").name == "connect"
    assert resolve_slash_command("/unknown") is None
    assert is_slash_command_token("/unknown") is True


def test_slash_popup_highlights_first_and_enter_runs_selection():
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one(PromptTextArea)
            prompt.focus()
            await pilot.press("/")
            await pilot.pause()

            popup = app.query_one(SlashCommandPopup)
            assert popup.display is True
            assert popup.highlighted_command.name == "memories"

            await pilot.press("down")
            assert popup.highlighted_command.name == "chats"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

    asyncio.run(exercise())


def test_tab_completes_and_auto_runs_exact_command():
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one(PromptTextArea)
            prompt.focus()
            for character in "/mo":
                await pilot.press(character)
            await pilot.pause()

            popup = app.query_one(SlashCommandPopup)
            assert [option.id for option in popup.options] == ["models"]
            await pilot.press("tab")
            await pilot.pause()
            assert isinstance(app.screen, ModelScreen)
            assert prompt.text == ""

    asyncio.run(exercise())


def test_slash_popup_keeps_status_aligned_with_prompt():
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one(PromptTextArea)
            prompt.focus()
            await pilot.press("/")
            await pilot.pause()

            status = app.query_one(StatusIndicator)
            assert status.region.bottom == prompt.region.bottom

    asyncio.run(exercise())


def test_mouse_hover_changes_highlighted_command():
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one(PromptTextArea)
            prompt.focus()
            await pilot.press("/")
            await pilot.pause()

            popup = app.query_one(SlashCommandPopup)
            await pilot.hover(popup, offset=(3, 3))
            await pilot.pause()
            assert popup.highlighted_command.name == "connect"

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("command", "screen_type"),
    [
        ("/memories", MemoryScreen),
        ("/chats", ChatScreen),
        ("/connect", ConnectionScreen),
        ("/models", ModelScreen),
    ],
)
def test_slash_commands_open_their_popups_without_reaching_model(
    command, screen_type
):
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptTextArea)
            prompt.focus()
            initial_conversation = list(app.conversation)

            for character in command:
                await pilot.press(character)
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, screen_type)
            assert prompt.text == ""
            assert app.conversation == initial_conversation
            assert app._input_queue.empty()
            assert command not in app._prompt_history

    asyncio.run(exercise())


def test_busy_agent_rejects_slash_command_without_queueing():
    async def exercise():
        app = AgentApp()
        notifications = []
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, **kwargs: notifications.append((message, kwargs)),
        )
        try:
            async with app.run_test() as pilot:
                app._agent_running = True
                app.on_prompt_submitted(tui_app.PromptSubmitted("/models"))
                await pilot.pause()
                assert len(app.screen_stack) == 1
                assert app._input_queue.empty()
                assert notifications[-1][1]["title"] == "Agent busy"
        finally:
            monkeypatch.undo()

    asyncio.run(exercise())


def test_model_popup_switches_local_model_and_updates_badge(monkeypatch):
    previous = get_config()
    agent.configure_openai(
        "http://localhost:7070/v1",
        "key",
        "old-local-model",
        provider="local",
        reasoning_effort="off",
    )
    saved = []
    monkeypatch.setattr(
        "remie.tui.screens.models.save_provider_configs",
        lambda profiles, active: saved.append((profiles, active)),
    )

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            app.on_prompt_submitted(tui_app.PromptSubmitted("/models"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelScreen)
            assert screen.query_one("#model-provider-label").render().plain.endswith(
                "Local"
            )

            model_input = screen.query_one("#model-picker-search", Input)
            model_input.value = "new-local-model"
            screen.query_one("#model-picker-submit").press()
            await pilot.pause()

            assert get_config().model == "new-local-model"
            assert "New Local Model" in app.query_one(ModelBadge).render().plain
            assert saved and saved[-1][1] == "local"
            assert isinstance(app.screen, tui_app.AgentScreen)

    try:
        asyncio.run(exercise())
    finally:
        agent.configure_openai(
            previous.base_url,
            previous.api_key,
            previous.model,
            previous.provider,
            previous.reasoning_effort,
            previous.verify_ssl,
        )


def test_model_popup_filters_remote_model_catalog(monkeypatch):
    agent.configure_openai(
        agent.OPENROUTER_BASE_URL,
        "key",
        "openai/gpt-5.6",
        provider="openrouter",
        reasoning_effort="medium",
        verify_ssl=True,
    )

    async def fake_models():
        return ["vendor/alpha", "vendor/beta"]

    monkeypatch.setattr(
        "remie.tui.screens.models.fetch_openrouter_models", fake_models
    )

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            app.on_prompt_submitted(tui_app.PromptSubmitted("/models"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelScreen)
            model_list = screen.query_one("#model-picker-list", OptionList)
            for _ in range(20):
                if any(option.id == "vendor/alpha" for option in model_list.options):
                    break
                await pilot.pause()

            screen.query_one("#model-picker-search", Input).value = "beta"
            await pilot.pause()
            assert [option.id for option in model_list.options] == ["vendor/beta"]

    asyncio.run(exercise())
