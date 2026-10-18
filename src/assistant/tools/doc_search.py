"""Document RAG: retrieve passages, ask the LLM to answer from them, check citations.

How we limit invented facts ("hallucinations"):
- the prompt allows only the retrieved context and asks for a fixed refusal sentence when
  the answer is missing;
- temperature 0 (most deterministic output);
- every citation in the answer is checked against the passages actually given to the
  model; citations to anything else are reported as `invalid_citations`.
These reduce, but do not eliminate, wrong answers: faithfulness is measured in Stage 6.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from assistant.agent.prompts import DOC_QA_SYSTEM, DOC_QA_USER, NOT_FOUND
from assistant.ingest.chunking import Chunk
from assistant.llm import LLM, LLMResponse
from assistant.retrieval.retriever import RetrievedChunk, Retriever

# Matches labels such as [kering_urd_2025 p.47]; also inside groups like [a p.1; b p.2].
_CITATION = re.compile(r"([A-Za-z0-9_\-]+) p\.(\d+)")
_BRACKETS = re.compile(r"\[([^\[\]]+)\]")
SNIPPET_CHARS = 300


def label(chunk: Chunk) -> str:
    return f"[{chunk.source} p.{chunk.page}]"


@dataclass(frozen=True)
class Citation:
    source: str
    page: int  # PDF page number
    page_label: str  # printed page number, if the PDF defines one
    snippet: str


@dataclass
class DocAnswer:
    question: str
    answer: str
    found: bool  # False when the model used the "not found" sentence
    citations: list[Citation]
    invalid_citations: list[str]  # cited labels that were NOT in the context
    retrieval_mode: str
    retrieved: list[RetrievedChunk] = field(repr=False)
    llm: LLMResponse = field(repr=False)

    @property
    def uncited(self) -> bool:
        """True for an answer that states something but cites no valid passage."""
        return self.found and not self.citations


def build_context(results: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"{label(r.chunk)}\n{r.chunk.text}" for r in results)


def extract_citations(
    answer: str, results: list[RetrievedChunk]
) -> tuple[list[Citation], list[str]]:
    """Split the labels cited in the answer into valid (given as context) and invalid.

    Matching ignores letter case: small models sometimes write "Kering_urd_2025".
    """
    by_key = {(r.chunk.source.lower(), r.chunk.page): r.chunk for r in results}
    valid: dict[tuple[str, int], Citation] = {}
    invalid: list[str] = []
    for group in _BRACKETS.findall(answer):
        for source, page in _CITATION.findall(group):
            key = (source.lower(), int(page))
            chunk = by_key.get(key)
            if chunk is None:
                if f"{source} p.{page}" not in invalid:
                    invalid.append(f"{source} p.{page}")
            elif key not in valid:
                valid[key] = Citation(
                    chunk.source, chunk.page, chunk.page_label, chunk.text[:SNIPPET_CHARS]
                )
    return list(valid.values()), invalid


def answer_question(
    question: str, retriever: Retriever, llm: LLM, mode: str = "hybrid_rerank", k: int = 5
) -> DocAnswer:
    results = retriever.retrieve(question, mode=mode, k=k)
    response = llm.complete(
        DOC_QA_SYSTEM, DOC_QA_USER.format(context=build_context(results), question=question)
    )
    text = response.text
    found = NOT_FOUND.rstrip(".").lower() not in text.lower()
    citations, invalid = extract_citations(text, results) if found else ([], [])
    return DocAnswer(question, text, found, citations, invalid, mode, results, response)
