"""Unit tests for the model display-name heuristics."""

from rich.text import Text

from remie.model_names import ModelInfo, prettify_model_id, prettify_model_name


def test_prettifies_screenshot_ids():
    cases = {
        "stealth/ox-alpha": ("Ox Alpha", "Stealth", False),
        "~z-ai/glm-latest": ("GLM Latest", "Z.AI", False),
        "z-ai/glm-5.3": ("GLM 5.3", "Z.AI", False),
        "dots-studio/dots-3-note-preview:free": (
            "Dots 3 Note Preview",
            "Dots Studio",
            True,
        ),
    }
    for model_id, (display, vendor, free) in cases.items():
        info = prettify_model_id(model_id)
        assert info.id == model_id
        assert info.display == display, model_id
        assert info.vendor == vendor, model_id
        assert info.free is free, model_id


def test_prettifies_common_vendor_prefixes():
    assert prettify_model_id("meta/muse-spark-1.2").vendor == "Meta"
    assert prettify_model_id("deepseek/deepseek-v4").display.startswith("DeepSeek")
    assert prettify_model_id("moonshotai/kimi-k3").vendor == "Moonshot AI"
    assert prettify_model_id("minimax-m3").display == "MiniMax M3"


def test_version_tokens_keep_case():
    info = prettify_model_id("tencent/hy-mt2-1.8b")
    assert info.display == "HY MT2 1.8B"
    assert prettify_model_name("qwen3.8") == "Qwen3.8"
    assert prettify_model_name("grok-4.5") == "Grok 4.5"


def test_ids_without_vendor_get_no_vendor_label():
    info = prettify_model_id("kimi-k3")
    assert info.display == "Kimi K3"
    assert info.vendor == ""


def test_resolved_display_falls_back_to_heuristic():
    info = ModelInfo(id="openai/gpt-x")
    assert info.resolved_display() != ""
    named = ModelInfo(id="x", display="Fancy Name")
    assert named.resolved_display() == "Fancy Name"


def test_model_option_builds_rich_label_with_badges():
    from remie.tui import _model_option

    label, value = _model_option(
        ModelInfo(id="z-ai/glm-5.3", display="GLM 5.3", vendor="Z.AI", free=True)
    )
    assert isinstance(label, Text)
    assert value == "z-ai/glm-5.3"
    plain = label.plain
    assert "GLM 5.3" in plain
    assert "Z.AI" in plain
    assert "Free" in plain


def test_model_option_accepts_raw_string_ids():
    from remie.tui import _model_option

    label, value = _model_option("stealth/ox-alpha")
    assert value == "stealth/ox-alpha"
    assert "Ox Alpha" in label.plain
    assert "Stealth" in label.plain
