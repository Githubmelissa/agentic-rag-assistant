"""Save and load chunks as JSON Lines: the single source of truth for both indexes."""

from __future__ import annotations

import json
from pathlib import Path

from assistant.ingest.chunking import Chunk

CHUNKS_FILE = "chunks.jsonl"


def save_chunks(chunks: list[Chunk], index_dir: Path) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / CHUNKS_FILE
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_chunks(index_dir: Path) -> list[Chunk]:
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the indexes first: python -m assistant.ingest"
        )
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
