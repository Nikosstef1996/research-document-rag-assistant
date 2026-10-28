"""A tiny in-memory semantic search index using normalized vector dot products."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from app.documents import Chunk


class Embedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, query: str) -> np.ndarray: ...


@dataclass(frozen=True)
class Match:
    chunk_id: str
    source: str
    text: str
    score: float
    page: int | None = None


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Cannot normalize a zero embedding vector")
    return x / norm


class SearchIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("No documents to index")
        self.chunks = chunks
        self.embedder = embedder
        raw = embedder.encode_documents([c.text for c in chunks])
        if raw.ndim != 2 or raw.shape[0] != len(chunks):
            raise ValueError("Embedder returned unexpected document shape")
        self.vectors = normalized(raw)

    def search(self, query: str, top_k: int = 3) -> list[Match]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        q = normalized(self.embedder.encode_query(query).reshape(1, -1))[0]
        if q.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query embedding dimension mismatch")
        # Cosine similarity equals dot product for unit-length embeddings.
        scores = self.vectors @ q
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [Match(self.chunks[i].chunk_id, self.chunks[i].source,
                      self.chunks[i].text, float(scores[i])) for i in indices]
