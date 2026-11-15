import asyncio

from textual.widgets import Button, Checkbox, TextArea
from remie.tui.app import AgentApp
from remie.tui.screens.memory import MemoryScreen
from remie.storage.memory import read_memory, memory_path


def test_memory_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("REMIE_HOME", str(tmp_path))

    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(120, 45)) as pilot:
            context = app._tab_prompt_context()["memory_scope"]
            app.push_screen(MemoryScreen(context))
            await pilot.pause()
            screen = app.screen

            async def press(name):
                await screen.on_button_pressed(Button.Pressed(screen.query_one(name, Button)))
                await pilot.pause()

            editor = screen.query_one(TextArea)
            editor.load_text("first note")
            await press("#memory-save")
            assert read_memory("project", context) == "first note"
            screen.query_one(Checkbox).value = True
            await pilot.pause()
            await press("#memory-edit")
            editor.load_text("updated note")
            await press("#memory-save")
            assert read_memory("project", context) == "updated note"
            editor.load_text("second note")
            await press("#memory-save")
            for checkbox in screen.query(Checkbox):
                checkbox.value = True
            await pilot.pause()
            await press("#memory-delete")
            assert read_memory("project", context)
            await press("#memory-delete")
            assert read_memory("project", context) == ""

            editor.load_text("stale write")
            path = memory_path("project", context)
            path.write_text("external change")
            await press("#memory-save")
            assert path.read_text() == "external change"

    asyncio.run(exercise())
