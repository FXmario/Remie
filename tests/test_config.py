"""Provider configuration is controlled by saved profiles, not legacy env vars."""

from remie.config import ConfigStore, default_config, provider_defaults


def test_retired_environment_variables_do_not_change_profiles(monkeypatch, tmp_path):
    for name, value in {
        "LLAMA_BASE_URL": "https://ignored.example/v1",
        "LLAMA_API_KEY": "ignored-key",
        "LLAMA_MODEL": "ignored-model",
        "REMIE_REASONING_EFFORT": "high",
    }.items():
        monkeypatch.setenv(name, value)

    defaults = provider_defaults("local")
    assert defaults.base_url == "http://localhost:7070/v1"
    assert defaults.api_key == "llama-cpp"
    assert defaults.model == "local-model"
    assert defaults.reasoning_effort == "medium"
    assert default_config() == defaults

    store = ConfigStore(tmp_path)
    assert store.load() == defaults
    assert store.load_profiles()["local"] == defaults
    for profile in store.load_profiles().values():
        assert profile.reasoning_effort == "medium"

    defaults.base_url = "http://localhost:8080/v1"
    defaults.api_key = "saved-key"
    defaults.model = "saved-model"
    defaults.reasoning_effort = "low"
    store.save(defaults)
    assert store.load() == defaults
    assert store.load_profiles()["local"] == defaults
