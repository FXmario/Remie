from remie.tools.common import _migrate_project_state, _project_id, _project_root


def test_project_root_prefers_nearest_git_root(tmp_path):
    root = tmp_path / "repo"
    nested = root / "src" / "package"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()

    assert _project_root(nested) == root.resolve()


def test_project_root_falls_back_to_start_directory(tmp_path):
    project = tmp_path / "not-a-repository"
    project.mkdir()

    assert _project_root(project) == project.resolve()


def test_project_ids_distinguish_same_named_directories(tmp_path):
    first = tmp_path / "one" / "app"
    second = tmp_path / "two" / "app"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    assert _project_id(first).startswith("app-")
    assert _project_id(first) != _project_id(second)


def test_migrates_and_removes_legacy_project_state(tmp_path):
    legacy = tmp_path / "project" / ".remie"
    destination = tmp_path / "home" / ".remie" / "projects" / "project-id"
    (legacy / "chats").mkdir(parents=True)
    (legacy / "chats" / "chat.json").write_text('{"messages": []}', encoding="utf-8")
    (legacy / "active_memory").write_text("memory-id", encoding="utf-8")

    assert _migrate_project_state(legacy, destination) is True
    assert not legacy.exists()
    assert (destination / "chats" / "chat.json").read_text(encoding="utf-8") == '{"messages": []}'
    assert (destination / "active_memory").read_text(encoding="utf-8") == "memory-id"
    assert (destination / ".migrated-from-project-dir").is_file()


def test_migration_conflict_keeps_legacy_state(tmp_path):
    legacy = tmp_path / "project" / ".remie"
    destination = tmp_path / "state" / "project-id"
    legacy.mkdir(parents=True)
    destination.mkdir(parents=True)
    (legacy / "active_memory").write_text("old", encoding="utf-8")
    (destination / "active_memory").write_text("new", encoding="utf-8")

    assert _migrate_project_state(legacy, destination) is False
    assert legacy.is_dir()
    assert (legacy / "active_memory").read_text(encoding="utf-8") == "old"
    assert (destination / "active_memory").read_text(encoding="utf-8") == "new"


def test_remie_dir_uses_configured_home_and_migrates(tmp_path, monkeypatch):
    import remie.tools as tools
    from remie.tools.common import _remie_dir

    project = tmp_path / "repo"
    project.mkdir()
    (project / ".git").mkdir()
    legacy = project / ".remie"
    legacy.mkdir()
    (legacy / "active_memory").write_text("abc", encoding="utf-8")
    state_home = tmp_path / "central-state"
    monkeypatch.chdir(project)
    monkeypatch.setenv("REMIE_HOME", str(state_home))
    monkeypatch.setattr(tools, "_remie_dir", _remie_dir)

    result = _remie_dir()

    assert result == state_home / "projects" / _project_id(project)
    assert (result / "active_memory").read_text(encoding="utf-8") == "abc"
    assert not legacy.exists()
