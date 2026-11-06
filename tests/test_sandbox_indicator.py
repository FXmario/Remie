import asyncio

import pytest
from textual.app import App

from remie.tui.widgets import SandboxHeader, SandboxIndicator


@pytest.mark.parametrize('mode,state,text', [
    (None, 'enabled', 'Sandbox ON'),
    ('on', 'enabled', 'Sandbox ON'),
    ('OFF', 'disabled', 'Sandbox OFF'),
    ('bad', 'invalid', 'Sandbox INVALID'),
])
def test_indicator_configuration(monkeypatch, mode, state, text):
    monkeypatch.setattr('remie.tui.widgets.sys.platform', 'linux')
    if mode is None:
        monkeypatch.delenv('REMIE_SANDBOX', raising=False)
    else:
        monkeypatch.setenv('REMIE_SANDBOX', mode)
    indicator = SandboxIndicator()
    indicator.refresh_status()
    assert indicator.has_class(state)
    assert text in str(indicator.content)


def test_unavailable(monkeypatch):
    monkeypatch.setattr('remie.tui.widgets.sys.platform', 'darwin')
    monkeypatch.setenv('REMIE_SANDBOX', 'on')
    indicator = SandboxIndicator()
    indicator.refresh_status()
    assert indicator.has_class('unavailable')


def test_narrow_header(monkeypatch):
    monkeypatch.setenv('REMIE_SANDBOX', 'on')

    class HeaderApp(App):
        def compose(self):
            yield SandboxHeader()

    async def exercise():
        async with HeaderApp().run_test(size=(40, 12)) as pilot:
            badge = pilot.app.query_one(SandboxIndicator)
            assert badge.region.width > 0
            assert badge.region.right <= 40
            assert badge.region.y == 0
    asyncio.run(exercise())
