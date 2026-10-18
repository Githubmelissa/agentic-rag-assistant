from pathlib import Path

import pytest
from pydantic import ValidationError

from assistant import __version__
from assistant.config import PROJECT_ROOT, Settings


def make_settings(**overrides: object) -> Settings:
    # _env_file=None: tests must not depend on a developer's local .env
    return Settings(_env_file=None, **overrides)


def test_version_is_set() -> None:
    assert __version__


def test_defaults() -> None:
    s = make_settings()
    assert s.chunk_size == 500
    assert s.chunk_overlap == 80
    assert s.llm_provider == "mock"


def test_relative_paths_resolve_to_project_root() -> None:
    s = make_settings(data_dir=Path("data"))
    assert s.data_dir == PROJECT_ROOT / "data"
    assert s.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert s.sales_db_path.name == "sales.db"


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "300")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    s = make_settings()
    assert s.chunk_size == 300
    assert s.llm_provider == "ollama"


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_api_key_hidden_from_repr() -> None:
    s = make_settings(llm_api_key="secret-value")
    assert "secret-value" not in repr(s)


def test_index_dir_per_chunk_size() -> None:
    s = make_settings()
    assert s.index_dir_for(300).name == "cs300"
    assert s.index_dir_for(300).parent == s.index_dir
