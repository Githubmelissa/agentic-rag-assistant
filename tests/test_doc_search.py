from pathlib import Path

import pytest

from assistant.agent.prompts import NOT_FOUND
from assistant.ingest.chunking import Chunk
from assistant.llm import LLMResponse, MockLLM
from assistant.retrieval.bm25 import BM25Index
from assistant.retrieval.dense import DenseIndex
from assistant.retrieval.retriever import RetrievedChunk, Retriever
from assistant.tools.doc_search import answer_question, build_context, extract_citations
from tests.conftest import FakeEmbedder


class FixedLLM:
    model = "fixed"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, system: str, user: str) -> LLMResponse:
        return LLMResponse(self.text, self.model, 1, 1, 0.0)


class NoRerank:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)


CHUNKS = [
    Chunk("k:p47:c0", "kering_urd", 47, "45", "Group revenue was 14,675 million euros.", 6),
    Chunk("h:p10:c0", "hermes_urd", 10, "", "Hermes employs many craftspeople.", 4),
    Chunk("l:p5:c0", "lvmh_ar", 5, "", "LVMH opened stores in Asia.", 5),
    Chunk("l:p9:c0", "lvmh_ar", 9, "", "Wine and spirits sales slowed.", 5),
]
RESULTS = [RetrievedChunk(c, 1.0, i + 1) for i, c in enumerate(CHUNKS[:2])]


@pytest.fixture
def retriever(tmp_path: Path, fake_embedder: FakeEmbedder) -> Retriever:
    dense = DenseIndex.build(CHUNKS, tmp_path, fake_embedder)
    return Retriever(CHUNKS, dense, BM25Index(CHUNKS), NoRerank())


def test_context_labels_every_passage() -> None:
    context = build_context(RESULTS)
    assert "[kering_urd p.47]\nGroup revenue" in context
    assert "[hermes_urd p.10]" in context


def test_extract_valid_invalid_and_grouped_citations() -> None:
    answer = (
        "Revenue was 14,675 [kering_urd p.47]. Staff grew [hermes_urd p.10; kering_urd p.47]. "
        "Profit rose [kering_urd p.99]."
    )
    valid, invalid = extract_citations(answer, RESULTS)
    assert [(c.source, c.page) for c in valid] == [("kering_urd", 47), ("hermes_urd", 10)]
    assert valid[0].page_label == "45"
    assert invalid == ["kering_urd p.99"]  # not given as context: flagged


def test_answer_question_with_mock_llm(retriever: Retriever) -> None:
    a = answer_question("group revenue", retriever, MockLLM(), mode="bm25", k=2)
    assert a.found
    assert a.citations and a.citations[0].source == "kering_urd"
    assert a.invalid_citations == []
    assert a.retrieval_mode == "bm25"


def test_not_found_answer_has_no_citations(retriever: Retriever) -> None:
    a = answer_question("CEO salary", retriever, FixedLLM(NOT_FOUND), mode="bm25")
    assert not a.found
    assert a.citations == []


def test_contradictory_answer_flagged(retriever: Retriever) -> None:
    llm = FixedLLM("Revenue was 99 billion [made_up_report p.1].")
    a = answer_question("revenue", retriever, llm, mode="dense")
    assert a.found
    assert a.citations == []
    assert a.invalid_citations == ["made_up_report p.1"]


def test_citation_matching_ignores_case() -> None:
    valid, invalid = extract_citations("Revenue [Kering_URD p.47].", RESULTS)
    assert [(c.source, c.page) for c in valid] == [("kering_urd", 47)]
    assert invalid == []


def test_answer_without_citation_is_flagged(retriever: Retriever) -> None:
    a = answer_question("revenue", retriever, FixedLLM("Revenue was 721."), mode="dense")
    assert a.found and a.uncited


def test_refusal_is_not_flagged_as_uncited(retriever: Retriever) -> None:
    a = answer_question("CEO salary", retriever, FixedLLM(NOT_FOUND), mode="dense")
    assert not a.uncited
