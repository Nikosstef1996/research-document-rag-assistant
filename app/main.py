"""Research document RAG service: persistent vector search + optional local LLM."""
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.embeddings import LocalEmbedder, MODEL_ID
from app.ollama_client import LLMUnavailable, generate_answer, DEFAULT_MODEL
from app.vector_store import PersistentRetriever

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "chroma_db"
app = FastAPI(title="Research Document RAG Assistant", version="0.2.0")


class Query(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=8)

    @field_validator("query")
    @classmethod
    def valid_query(cls, text: str) -> str:
        value = text.strip()
        if len(value) < 2:
            raise ValueError("Query needs two non-whitespace characters")
        return value


class Source(BaseModel):
    chunk_id: str
    source: str
    page: int | None
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[Source]


class AnswerResponse(SearchResponse):
    answer: str
    llm_model: str


@lru_cache(maxsize=1)
def get_retriever():
    return PersistentRetriever(DB_DIR, LocalEmbedder())


def search_docs(query: str, top_k: int):
    try:
        return get_retriever().search(query, top_k)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=f"Index not ready: {exc}. Run python -m scripts.index_documents") from exc
    except Exception as exc:
        # Chroma reports a missing collection via its own error types.
        if "NotFoundError" in type(exc).__name__:
            raise HTTPException(status_code=503,
                detail="Index missing. Run python -m scripts.index_documents") from exc
        raise


@app.get("/health")
def health():
    return {"status": "ok", "generation": "optional-local-ollama"}


@app.post("/search", response_model=SearchResponse)
def search(payload: Query):
    matches = search_docs(payload.query, payload.top_k)
    return SearchResponse(query=payload.query, results=[Source(**vars(m)) for m in matches])


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: Query):
    matches = search_docs(payload.query, payload.top_k)
    try:
        answer = generate_answer(payload.query, matches)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AnswerResponse(query=payload.query, answer=answer, llm_model=DEFAULT_MODEL,
                          results=[Source(**vars(m)) for m in matches])
