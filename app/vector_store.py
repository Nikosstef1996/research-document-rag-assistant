"""Persist normalized Sentence Transformer embeddings in an on-disk Chroma collection."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import numpy as np

from app.documents import Chunk
from app.retrieval import Match

COLLECTION_NAME = "research_document_chunks"


class Embedder(Protocol):
    model_id: str
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, query: str) -> np.ndarray: ...


def client_for(db_dir: Path):
    import chromadb  # optional heavy dependency, only used when indexing/querying
    return chromadb.PersistentClient(path=str(db_dir))


def build_index(chunks: list[Chunk], db_dir: Path, embedder: Embedder) -> int:
    """Replace the entire local collection with freshly generated embeddings."""
    if not chunks:
        raise ValueError("No chunks to index")
    vectors = np.asarray(embedder.encode_documents([c.text for c in chunks]), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding model returned the wrong number of vectors")
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite embeddings")
    client = client_for(db_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:  # Older Chroma versions may use ValueError for a missing collection.
        pass
    except Exception as exc:
        # Chroma 1.x uses NotFoundError. Import only on this error path so
        # lightweight tests can mock the vector store without installing Chroma.
        from chromadb.errors import NotFoundError
        if not isinstance(exc, NotFoundError):
            raise
    collection = client.get_or_create_collection(COLLECTION_NAME,
                                metadata={"hnsw:space": "cosine"}, embedding_function=None)
    # Chroma may impose a maximum write batch size; keep batches small.
    for start in range(0, len(chunks), 100):
        batch = chunks[start:start+100]
        collection.upsert(ids=[c.chunk_id for c in batch],
                          documents=[c.text for c in batch],
                          metadatas=[{"source": c.source, "page": c.page if c.page else 0,
                                      "embedding_model": embedder.model_id} for c in batch],
                          embeddings=vectors[start:start+100].tolist())
    return collection.count()


class PersistentRetriever:
    def __init__(self, db_dir: Path, embedder: Embedder):
        self.embedder = embedder
        self.collection = client_for(db_dir).get_collection(COLLECTION_NAME,
                                                             embedding_function=None)
        if self.collection.count() == 0:
            raise ValueError("Collection is empty. Run scripts.index_documents.")
        # Prevent comparing embeddings from different models.
        metadata = self.collection.get(limit=1, include=["metadatas"])["metadatas"]
        if metadata and metadata[0].get("embedding_model") != embedder.model_id:
            raise ValueError("Stored model differs; rebuild index with scripts.index_documents")

    def search(self, query: str, top_k: int = 3) -> list[Match]:
        if not query.strip() or top_k < 1:
            raise ValueError("Provide a query and positive top_k")
        embedding = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise ValueError("Invalid query embedding")
        result = self.collection.query(query_embeddings=[embedding.tolist()],
                                       n_results=min(top_k, self.collection.count()),
                                       include=["documents", "metadatas", "distances"])
        matches = []
        for chunk_id, content, meta, distance in zip(
                result["ids"][0],result["documents"][0],result["metadatas"][0],result["distances"][0]):
            matches.append(Match(chunk_id=chunk_id,source=meta["source"],text=content,
                                 score=float(1.0-distance), page=(meta.get("page") or None)))
        return matches
