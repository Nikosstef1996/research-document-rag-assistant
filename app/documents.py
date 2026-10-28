"""Load .txt and machine-readable .pdf documents with page-level provenance."""
from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    page: int | None = None  # 1-based PDF page number, None for .txt


def split_words(text: str, size: int = 90, overlap: int = 20) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    return [" ".join(words[start:start + size])
            for start in range(0, len(words), step) if start == 0 or start + overlap < len(words)]


def load_chunks(directory: Path, size: int = 90, overlap: int = 20) -> list[Chunk]:
    """Load supported files in one directory; text-based PDFs only (no OCR)."""
    if not directory.is_dir():
        raise FileNotFoundError(f"No document directory: {directory}")
    chunks: list[Chunk] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".txt", ".pdf"):
            continue
        if path.suffix.lower() == '.txt':
            pages = [(None, path.read_text(encoding='utf-8'))]
        else:
            import pymupdf  # import only if needed
            with pymupdf.open(path) as pdf:
                pages = [(i + 1, page.get_text("text", sort=True)) for i, page in enumerate(pdf)]
        for page, raw in pages:
            for i, passage in enumerate(split_words(raw, size, overlap)):
                # Deterministic IDs, including the file path so sample/user names don't collide.
                digest = hashlib.sha256(f"{path.resolve()}|{page}|{i}".encode()).hexdigest()[:20]
                chunks.append(Chunk(chunk_id=f"chunk-{digest}", source=path.name,
                                    text=passage, page=page))
    return chunks


def load_corpus(directories: list[Path], size: int = 90, overlap: int = 20) -> list[Chunk]:
    chunks = []
    for directory in directories:
        chunks.extend(load_chunks(directory, size=size, overlap=overlap))
    if not chunks:
        raise ValueError("No extractable text found. Scanned-image PDFs require OCR (not included).")
    return chunks
