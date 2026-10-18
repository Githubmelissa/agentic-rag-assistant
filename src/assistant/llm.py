"""Provider-agnostic LLM wrapper.

Every provider implements one method, `complete(system, user)`, and returns an
`LLMResponse` with the text, token counts and latency. The rest of the code never
depends on a specific provider, so switching is one line in `.env`:

- mock:              deterministic fake answers, no network (tests and CI)
- ollama:            local model through Ollama's native /api/chat endpoint
- groq:              free hosted tier, OpenAI-compatible API
- openai_compatible: any other OpenAI-compatible endpoint (set LLM_BASE_URL)

We call the HTTP APIs directly with `httpx` rather than adding a vendor SDK per
provider: one small dependency, and the request/response format is visible in the code.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from assistant.config import Settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot be reached or returns an error."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None  # None when the provider does not report usage
    output_tokens: int | None
    latency_s: float


class LLM(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


class MockLLM:
    """Fake LLM for tests and CI. Cites the first context label it finds, or refuses."""

    model = "mock"
    _label = re.compile(r"\[[^\[\]]+ p\.\d+\]")

    def complete(self, system: str, user: str) -> LLMResponse:
        match = self._label.search(user)
        text = (
            f"According to the documents, this is the answer {match.group(0)}."
            if match
            else "I could not find this in the documents."
        )
        return LLMResponse(text, self.model, len(user.split()), len(text.split()), 0.0)


class OllamaLLM:
    """Local model served by Ollama (https://ollama.com)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        num_ctx: int,
        timeout_s: float,
    ) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/api/chat"
        self.options = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx}
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": self.options,
        }
        t0 = time.perf_counter()
        try:
            r = httpx.post(self.url, json=payload, timeout=self.timeout_s)
        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.url}. Is Ollama installed and running?"
            ) from e
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        if r.status_code != 200:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        data = r.json()
        return LLMResponse(
            text=data["message"]["content"].strip(),
            model=self.model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_s=time.perf_counter() - t0,
        )


class OpenAICompatibleLLM:
    """Hosted model behind an OpenAI-compatible /chat/completions endpoint (e.g. Groq)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> None:
        if not api_key:
            raise LLMError("LLM_API_KEY is empty. Put your key in the .env file.")
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        t0 = time.perf_counter()
        try:
            r = httpx.post(self.url, json=payload, headers=self.headers, timeout=self.timeout_s)
        except httpx.HTTPError as e:
            raise LLMError(f"Cannot reach {self.url}: {e}") from e
        if r.status_code != 200:
            # Never include request headers in the message: they contain the API key.
            raise LLMError(f"LLM API error {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usage") or {}
        return LLMResponse(
            text=data["choices"][0]["message"]["content"].strip(),
            model=self.model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_s=time.perf_counter() - t0,
        )


def make_llm(settings: Settings) -> LLM:
    """Build the LLM chosen in the settings (LLM_PROVIDER)."""
    s = settings
    if s.llm_provider == "mock":
        return MockLLM()
    if not s.llm_model:
        raise LLMError("LLM_MODEL is empty. Set it in the .env file.")
    if s.llm_provider == "ollama":
        return OllamaLLM(
            s.llm_model,
            s.ollama_base_url,
            s.llm_temperature,
            s.llm_max_tokens,
            s.llm_num_ctx,
            s.llm_timeout_seconds,
        )
    base_url = GROQ_BASE_URL if s.llm_provider == "groq" else s.llm_base_url
    if not base_url:
        raise LLMError("LLM_BASE_URL is empty (needed for provider openai_compatible).")
    return OpenAICompatibleLLM(
        s.llm_model,
        base_url,
        s.llm_api_key,
        s.llm_temperature,
        s.llm_max_tokens,
        s.llm_timeout_seconds,
    )
