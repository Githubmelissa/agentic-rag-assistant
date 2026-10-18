import json
from pathlib import Path

from assistant.ingest.__main__ import MANIFEST_FILE, build
from assistant.ingest.chunking import Chunk, count_words
from assistant.ingest.store import load_chunks, save_chunks
from assistant.retrieval.bm25 import BM25Index, tokenize
from assistant.retrieval.dense import DenseIndex
from tests.conftest import FakeEmbedder, make_pdf


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(cid, "doc", 1, "", text, count_words(text))


CHUNKS = [
    chunk("a", "Kering revenue reached 17.2 billion euros in 2025."),
    chunk("b", "Hermes opened new leather workshops in France."),
    chunk("c", "Greenhouse gas emissions fell thanks to renewable energy."),
]


def test_tokenize_keeps_numbers_and_drops_stopwords() -> None:
    assert tokenize("The revenue was 17.2 billion in 2025") == [
        "revenue",
        "17.2",
        "billion",
        "2025",
    ]


def test_bm25_ranks_keyword_match_first() -> None:
    results = BM25Index(CHUNKS).search("leather workshops", k=3)
    assert results[0][0] == "b"
    assert all(score > 0 for _, score in results)


def test_bm25_returns_nothing_for_unknown_words() -> None:
    assert BM25Index(CHUNKS).search("xylophone", k=3) == []


def test_chunks_roundtrip(tmp_path: Path) -> None:
    save_chunks(CHUNKS, tmp_path)
    assert load_chunks(tmp_path) == CHUNKS


def test_dense_index_build_open_search(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    DenseIndex.build(CHUNKS, tmp_path, fake_embedder)
    index = DenseIndex.open(tmp_path, fake_embedder)  # reopened from disk
    assert index.count() == 3
    results = index.search("renewable energy emissions", k=2)
    assert results[0][0] == "c"
    assert len(results) == 2


def test_dense_rebuild_replaces_old_collection(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    DenseIndex.build(CHUNKS, tmp_path, fake_embedder)
    index = DenseIndex.build(CHUNKS[:1], tmp_path, fake_embedder)
    assert index.count() == 1


def test_build_end_to_end(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(
        raw / "acme.pdf",
        ["Acme sells luxury handbags. " * 20, "Acme cut its greenhouse gas emissions by half."],
    )
    index_dir = tmp_path / "idx"
    manifest = build(raw, index_dir, 50, 10, count_words, fake_embedder, "fake")
    assert manifest["n_chunks"] == len(load_chunks(index_dir))
    assert json.loads((index_dir / MANIFEST_FILE).read_text())["chunk_size"] == 50
    assert (
        DenseIndex.open(index_dir, fake_embedder)
        .search("emissions", k=1)[0][0]
        .startswith("acme:p2")
    )
