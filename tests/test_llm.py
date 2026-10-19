from typing import Any

import httpx
import pytest

from assistant.llm import (
    GROQ_BASE_URL,
    LLMError,
    MockLLM,
    OllamaLLM,
    OpenAICompatibleLLM,
    make_llm,
)
from tests.conftest import make_settings as settings


class FakeResponse:
    def __init__(self, status_code: int, data: dict[str, Any]) -> None:
        self.status_code = status_code
        self._data = data
        self.text = str(data)

    def json(self) -> dict[str, Any]:
        return self._data


def test_mock_cites_first_label() -> None:
    r = MockLLM().complete("sys", "Context:\n[doc p.3]\nsome text\n\nQuestion: q")
    assert "[doc p.3]" in r.text


def test_mock_refuses_without_context() -> None:
    assert "could not find" in MockLLM().complete("sys", "no labels").text


def test_make_llm_selects_provider() -> None:
    assert isinstance(make_llm(settings()), MockLLM)
    assert isinstance(make_llm(settings(llm_provider="ollama", llm_model="m")), OllamaLLM)
    groq = make_llm(settings(llm_provider="groq", llm_model="m", llm_api_key="k"))
    assert isinstance(groq, OpenAICompatibleLLM)
    assert groq.url == GROQ_BASE_URL + "/chat/completions"


def test_make_llm_errors_are_explicit() -> None:
    with pytest.raises(LLMError, match="LLM_MODEL"):
        make_llm(settings(llm_provider="ollama"))
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        make_llm(settings(llm_provider="groq", llm_model="m"))
    with pytest.raises(LLMError, match="LLM_BASE_URL"):
        make_llm(settings(llm_provider="openai_compatible", llm_model="m", llm_api_key="k"))


def test_ollama_request_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        sent.update(url=url, json=json)
        return FakeResponse(
            200, {"message": {"content": " hi "}, "prompt_eval_count": 10, "eval_count": 2}
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OllamaLLM("qwen", "http://localhost:11434/", 0.0, 100, 8192, 5)
    r = llm.complete("system", "user")
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["json"]["options"]["num_ctx"] == 8192  # context window must be set
    assert (r.text, r.input_tokens, r.output_tokens) == ("hi", 10, 2)


def test_ollama_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(404, {}))
    with pytest.raises(LLMError, match="ollama pull"):
        OllamaLLM("qwen", "http://x", 0.0, 10, 10, 5).complete("s", "u")


def test_ollama_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*a: Any, **k: Any) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(LLMError, match="Is Ollama installed"):
        OllamaLLM("qwen", "http://x", 0.0, 10, 10, 5).complete("s", "u")


def test_api_error_never_leaks_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(401, {"error": "bad key"}))
    llm = OpenAICompatibleLLM("m", "https://api.example", "SECRET-KEY", 0.0, 10, 5)
    with pytest.raises(LLMError) as err:
        llm.complete("s", "u")
    assert "SECRET-KEY" not in str(err.value)


def test_ollama_out_of_memory_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"error": "model runner has unexpectedly stopped, this may be due to resource limits"}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(500, body))
    with pytest.raises(LLMError, match="LLM_NUM_CTX"):
        OllamaLLM("qwen", "http://x", 0.0, 10, 10, 5).complete("s", "u")
