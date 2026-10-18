"""Dense (semantic) index: chunk embeddings stored in a persistent Chroma collection.

We compute embeddings ourselves and pass them to Chroma (`embedding_function=None`), so
Chroma never downloads its own default model and we control exactly which model is used.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from assistant.ingest.chunking import Chunk
from assistant.retrieval.embeddings import Embedder

COLLECTION = "chunks"
CHROMA_SUBDIR = "chroma"
ADD_BATCH = 1000  # Chroma limits the size of a single insert


def _client(index_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(index_dir / CHROMA_SUBDIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


class DenseIndex:
    def __init__(self, collection: chromadb.Collection, embedder: Embedder) -> None:
        self.collection = collection
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk], index_dir: Path, embedder: Embedder) -> DenseIndex:
        """(Re)create the collection from scratch and add all chunks."""
        client = _client(index_dir)
        if COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION)
        collection = client.create_collection(
            COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
        vectors = embedder.embed_documents([c.text for c in chunks])
        for start in range(0, len(chunks), ADD_BATCH):
            batch = chunks[start : start + ADD_BATCH]
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[start : start + ADD_BATCH],
                documents=[c.text for c in batch],
                metadatas=[
                    {"source": c.source, "page": c.page, "page_label": c.page_label} for c in batch
                ],
            )
        return cls(collection, embedder)

    @classmethod
    def open(cls, index_dir: Path, embedder: Embedder) -> DenseIndex:
        collection = _client(index_dir).get_collection(COLLECTION, embedding_function=None)
        return cls(collection, embedder)

    def count(self) -> int:
        return self.collection.count()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the k most similar chunks as (chunk_id, cosine similarity), best first."""
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, self.count()),
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0] if result["distances"] else []
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances, strict=True)]
