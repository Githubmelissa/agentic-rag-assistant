"""HTTP API around the agent: `POST /ask` and `GET /health`.

    uvicorn assistant.api:app --reload

Design choices (see docs/decisions.md):
- The agent is built once, at startup: loading the embedding and reranking models takes
  several seconds and a few hundred MB of RAM, so doing it per request would add that to
  every answer.
- If the indexes or the LLM are missing, the app still starts: `/health` reports what is
  wrong and `/ask` answers 503. A container that dies at boot tells the user nothing.
- One question at a time, guarded by a lock. A 3B model on a laptop CPU has no spare
  capacity: serialising gives predictable latency instead of several slow answers at once.
  A second question while one is running gets 429 rather than waiting for minutes.
- The endpoints are plain `def`, so FastAPI runs them in a worker thread and `/health`
  stays responsive while an answer is being produced.
- No authentication: this is a local demo. Putting it on a network would need at least an
  API key and a rate limit (listed in the README's limitations).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from assistant.agent.graph import Agent, AgentConfig, AgentResult
from assistant.config import Settings, get_settings
from assistant.llm import LLMError, make_llm
from assistant.retrieval.retriever import Retriever

BUSY_WAIT_SECONDS = 1.0  # how long a request waits for the one in front of it
MAX_ROWS_IN_RESPONSE = 100  # the SQL guard already limits rows; this is a second belt


class AskRequest(BaseModel):
    # extra="forbid": a misspelled field is an error, not silence.
    # str_strip_whitespace: "   " is an empty question, not a 3-character one.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=500)


class CitationOut(BaseModel):
    source: str
    page: int  # PDF page number
    page_label: str  # printed page number, when the PDF has one
    snippet: str


class StepOut(BaseModel):
    node: str
    latency_s: float


class AskResponse(BaseModel):
    question: str
    route: str
    answer: str
    citations: list[CitationOut]
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    steps: list[StepOut]
    latency_s: float
    trace_id: str


class HealthResponse(BaseModel):
    status: str  # "ok" or "not ready"
    detail: str | None
    llm: str
    chunk_size: int
    chunks_indexed: int | None
    sales_db: bool
    uptime_s: float


def to_response(result: AgentResult) -> AskResponse:
    return AskResponse(
        question=result.question,
        route=result.route,
        answer=result.answer,
        citations=[
            CitationOut(source=c.source, page=c.page, page_label=c.page_label, snippet=c.snippet)
            for c in result.citations
        ],
        sql=result.sql,
        columns=result.columns,
        rows=[list(r) for r in result.rows[:MAX_ROWS_IN_RESPONSE]],
        row_count=len(result.rows),
        steps=[StepOut(node=s["node"], latency_s=s["latency_s"]) for s in result.trace],
        latency_s=round(result.total_latency_s, 2),
        trace_id=result.trace_id,
    )


class Service:
    """Holds the agent and serialises questions. Built once per process."""

    def __init__(self, settings: Settings, agent: Agent | None = None) -> None:
        self.settings = settings
        self.agent = agent
        self.retriever: Retriever | None = getattr(agent, "retriever", None)
        self.error: str | None = None
        self.lock = Lock()
        self.started_at = time.monotonic()

    def start(self) -> None:
        """Build the agent. Any failure is kept as a message instead of killing the app."""
        if self.agent is not None:
            return
        s = self.settings
        try:
            llm = make_llm(s)
            self.retriever = Retriever.from_settings(s)
            config = AgentConfig(
                top_k=s.answer_top_k,
                sql_max_retries=s.sql_max_retries,
                sql_retry_on_empty=s.sql_retry_on_empty,
                sql_max_rows=s.sql_row_limit,
                sql_timeout_s=s.sql_timeout_seconds,
            )
            self.agent = Agent(self.retriever, llm, s.sales_db_path, config, log_dir=s.log_dir)
        except Exception as e:  # noqa: BLE001 - reported by /health, not swallowed
            self.error = f"{type(e).__name__}: {e}"

    def ask(self, question: str) -> AskResponse:
        if self.agent is None:
            raise HTTPException(503, self.error or "the assistant is not ready yet")
        if not self.lock.acquire(timeout=BUSY_WAIT_SECONDS):
            raise HTTPException(429, "busy: this assistant answers one question at a time")
        try:
            return to_response(self.agent.ask(question))
        except LLMError as e:
            raise HTTPException(503, f"the language model could not be reached: {e}") from e
        finally:
            self.lock.release()

    def health(self) -> HealthResponse:
        s = self.settings
        return HealthResponse(
            status="ok" if self.agent is not None else "not ready",
            detail=self.error,
            llm=f"{s.llm_provider}:{s.llm_model}" if s.llm_model else s.llm_provider,
            chunk_size=s.chunk_size,
            chunks_indexed=len(self.retriever.chunks) if self.retriever else None,
            sales_db=s.sales_db_path.exists(),
            uptime_s=round(time.monotonic() - self.started_at, 1),
        )


def create_app(settings: Settings | None = None, agent: Agent | None = None) -> FastAPI:
    """`agent` is injected by the tests, so they never load a model."""
    service = Service(settings or get_settings(), agent)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        service.start()  # a few seconds: the models are loaded here, not per request
        yield

    app = FastAPI(
        title="Agentic RAG assistant",
        description="Ask a question about the 2025 company reports or the gift-shop sales.",
        version="0.0.1",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return service.health()

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return service.ask(request.question)

    return app


app = create_app()
