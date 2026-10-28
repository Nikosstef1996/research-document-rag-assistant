# Research Document RAG Assistant

A **local-first Retrieval-Augmented Generation (RAG) application** for searching research documents and generating answers based on retrieved source passages. Built with Python, Sentence Transformers, ChromaDB, Ollama, and FastAPI. No cloud account or paid inference API is required.

## Features

- **Document ingestion:** Extracts text from `.txt` files and text-based PDFs, retaining PDF page references.
- **Semantic search:** Encodes document chunks and queries using `sentence-transformers/all-MiniLM-L6-v2` and ranks passages by cosine similarity.
- **Persistent vector storage:** Stores embeddings in a local ChromaDB collection, reusable across runs.
- **Source-grounded answers:** Sends retrieved passages to a locally running `qwen2.5:3b` model through Ollama, prompting it to cite supporting excerpts.
- **REST API:** Provides document search and question answering through FastAPI and Pydantic validation.
- **Automated tests and CI:** Includes pytest tests and a GitHub Actions workflow.

## Architecture

```text
TXT / text-based PDF documents
          |
          v
 Text extraction + chunking
          |
          v
 Sentence Transformer embeddings
          |
          v
  Persistent ChromaDB index
          ^
          |  semantic retrieval
          |
 User query --> embedding model
          |
          +--> POST /search --> ranked source passages
          |
          +--> POST /ask --> retrieved passages + query
                                  |
                                  v
                            Ollama / Qwen2.5
                                  |
                                  v
                          Answer + source passages
```

## Requirements

- Python **3.11** (recommended for this project)
- Internet access for the initial Python package and embedding-model downloads
- Optional: [Ollama](https://ollama.com/download) and `qwen2.5:3b` for locally generated answers

## Quick start (Windows CMD)

Run the commands from the project root (the directory containing `requirements.txt`):

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 1. Index the documents

```bat
.venv\Scripts\python.exe -m scripts.index_documents
```

The project includes five short example `.txt` documents in `data/sample_docs/`. You can add your own permitted-to-use TXT or text-based PDF files to `data/user_docs/`, then rerun the indexing command. Rebuilding **replaces** the existing document collection. The generated `data/chroma_db/` index is not committed to Git.

### 2. Try semantic search (no LLM required)

```bat
.venv\Scripts\python.exe -m scripts.demo_ask --query "How does conformal prediction quantify uncertainty?" --top-k 3 --search-only
```

This prints the highest-ranked passages, their source files, available page references, and similarity scores.

### 3. Enable local answer generation (optional)

Install and start [Ollama](https://ollama.com/download), then download the model:

```bat
ollama pull qwen2.5:3b
```

Run a retrieval-augmented question:

```bat
.venv\Scripts\python.exe -m scripts.demo_ask --query "How does conformal prediction quantify uncertainty?" --top-k 3
```

Ollama runs independently of the Python virtual environment, with its local API at `http://127.0.0.1:11434` by default. The application requests the model through that API; it does **not** require an AWS account or cloud model endpoint.

### 4. Start the API

```bat
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8001
```

Open [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs) for interactive API documentation.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | API health response |
| `/search` | `POST` | Return retrieved passages; no local LLM needed |
| `/ask` | `POST` | Generate an answer from retrieved passages; requires Ollama |

Example request body for `POST /ask`:

```json
{
  "query": "What does conformal prediction guarantee?",
  "top_k": 3
}
```

The API returns the answer and the retrieved supporting passages, including source metadata. If the index or LLM is unavailable, the application returns an error rather than silently inventing an answer.

## Testing and continuous integration

Run the unit tests:

```bat
.venv\Scripts\python.exe -m pytest -q
```

The test suite mocks external model and database dependencies; it does not constitute an end-to-end benchmark of RAG answer quality. The included `.github/workflows/ci.yml` runs lightweight tests on pushes and pull requests.

## Project structure

```text
app/                 FastAPI routes, extraction, embeddings, retrieval, LLM client
scripts/             Indexing and CLI demos
tests/               Unit tests
data/sample_docs/     Five example text documents
data/user_docs/       Optional local documents (Git-ignored)
data/chroma_db/       Generated local vector index (Git-ignored)
.github/workflows/    CI configuration
```

## Limitations and future work

This repository demonstrates a functional **local RAG architecture**, not a production-ready research assistant. The included documents are short teaching examples rather than independently validated scientific references. Generated answers can still contain unsupported claims, even with citation instructions; users should inspect the returned passages. PDF extraction is limited to text-based PDFs (no OCR), and the default embedding model primarily targets English text. The API does not currently implement authentication, request throttling, multi-user isolation, or a formal retrieval/answer-quality evaluation benchmark. Planned improvements include an evaluation dataset, retrieval metrics (e.g., Recall@k), citation-grounding checks, and multilingual retrieval.

**Privacy and publishing:** `.venv/`, `.env`, `data/chroma_db/`, and files placed in `data/user_docs/` are excluded by `.gitignore`. Review staged files before publishing, and do not upload credentials, proprietary documents, or sensitive information.

## References

- [Sentence Transformers documentation](https://sbert.net/)
- [Chroma documentation](https://docs.trychroma.com/)
- [Ollama API documentation](https://docs.ollama.com/api)
- [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
