from pathlib import Path

from ui.settings_store import DesktopSettings, SettingsStore


def test_settings_store_round_trip(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    settings = DesktopSettings(
        source_path="docs",
        db_path="custom_db",
        ollama_model="deepseek-r1:7b",
        top_k=9,
    )

    store.save(settings)
    loaded = store.load()

    assert loaded.source_path == "docs"
    assert loaded.db_path == "custom_db"
    assert loaded.ollama_model == "deepseek-r1:7b"
    assert loaded.top_k == 9
