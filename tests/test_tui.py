import asyncio

import httpx

import chaldea.tui as tui
from chaldea.tui import AgentApp


def test_connection_error_shows_toast_and_keeps_app_running(monkeypatch):
    async def exercise():
        async def failed_stream(_conversation):
            raise httpx.ConnectError("connection refused")
            yield ""

        monkeypatch.setattr(tui, "stream_llm_call", failed_stream)

        app = AgentApp()
        notifications = []
        monkeypatch.setattr(
            app,
            "notify",
            lambda *args, **kwargs: notifications.append((args, kwargs)),
        )
        async with app.run_test() as pilot:
            worker = app.run_agent_turn()
            await worker.wait()
            await pilot.pause()

            assert app.is_running
            assert app.query_one("#prompt").disabled is False
            assert len(notifications) == 1
            assert notifications[0][1]["severity"] == "error"

    asyncio.run(exercise())
