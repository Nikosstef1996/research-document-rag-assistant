"""Rebuild the local Chroma index from bundled sample docs + private user docs."""
from pathlib import Path
from app.documents import load_corpus
from app.embeddings import LocalEmbedder
from app.vector_store import build_index

ROOT = Path(__file__).resolve().parents[1]


def main():
    chunks = load_corpus([ROOT / "data/sample_docs",ROOT / "data/user_docs"])
    print(f"Extracted {len(chunks)} chunks; loading embedding model...")
    embedder = LocalEmbedder()
    count = build_index(chunks, ROOT / "data/chroma_db",embedder)
    print(f"Indexed {count} chunks in data/chroma_db; source pages retained for PDF files.")
    print("Re-run after adding or changing documents, then restart the FastAPI server.")


if __name__ == "__main__":
    main()
