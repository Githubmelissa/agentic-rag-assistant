import pytest

from assistant.ingest.chunking import chunk_page, count_words, split_sentences
from assistant.ingest.loaders import Page


def sentence(i: int, n_words: int = 10) -> str:
    """A sentence of exactly n_words words, recognisable by its number."""
    return f"S{i} " + " ".join(["word"] * (n_words - 2)) + " end."


def page_of(*blocks: str) -> Page:
    return Page(source="doc", page=7, page_label="5", blocks=list(blocks))


def test_split_sentences() -> None:
    text = "Revenue was €17.2 billion. It fell 12%. The U.S. market grew."
    assert split_sentences(text) == [
        "Revenue was €17.2 billion.",
        "It fell 12%.",
        "The U.S. market grew.",
    ]


def test_short_page_is_one_chunk_with_metadata() -> None:
    chunks = chunk_page(page_of(sentence(1)), chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.chunk_id, c.source, c.page, c.page_label) == ("doc:p7:c0", "doc", 7, "5")
    assert c.n_tokens == 10


def test_chunks_respect_size_and_never_cut_sentences() -> None:
    text = " ".join(sentence(i) for i in range(20))  # 200 words
    chunks = chunk_page(page_of(text), chunk_size=35, chunk_overlap=0)
    assert all(c.n_tokens <= 35 for c in chunks)
    for c in chunks:
        assert c.text.endswith("end.")  # every chunk ends on a sentence boundary
    # Without overlap, every sentence appears exactly once.
    joined = " ".join(c.text for c in chunks)
    assert all(joined.count(f"S{i} ") == 1 for i in range(20))


def test_overlap_repeats_last_sentence() -> None:
    text = " ".join(sentence(i) for i in range(6))  # 60 words
    chunks = chunk_page(page_of(text), chunk_size=30, chunk_overlap=10)
    # chunk 0 = S0 S1 S2; chunk 1 starts with S2 (the 10-token overlap)
    assert chunks[0].text.split()[-10] == "S2"
    assert chunks[1].text.startswith("S2 ")


def test_overlong_sentence_is_split_into_windows() -> None:
    long_sentence = " ".join(f"w{i}" for i in range(100)) + "."
    chunks = chunk_page(page_of(long_sentence), chunk_size=30, chunk_overlap=0)
    assert all(c.n_tokens <= 30 for c in chunks)
    assert sum(c.n_tokens for c in chunks) == 100


def test_chunk_ids_are_unique_and_ordered() -> None:
    text = " ".join(sentence(i) for i in range(10))
    chunks = chunk_page(page_of(text), chunk_size=25, chunk_overlap=5)
    assert [c.chunk_id for c in chunks] == [f"doc:p7:c{i}" for i in range(len(chunks))]


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_page(page_of("x."), chunk_size=10, chunk_overlap=10)


def test_count_words() -> None:
    assert count_words("  two   words ") == 2
