"""Simple demonstration of semantic similarity (downloads model on first run)."""
import numpy as np
from app.embeddings import LocalEmbedder
from app.retrieval import normalized


def main():
    sentences = [
        "Predicting tomorrow's electricity consumption",
        "Forecasting next-day power demand",
        "A Docker container packages an application",
    ]
    vectors = normalized(LocalEmbedder().encode_documents(sentences))
    print("Embedding array shape:", vectors.shape)
    scores = vectors @ vectors[0]
    for s, score in zip(sentences, scores):
        print(f"{score:.3f}  {s}")
    print("Closer cosine scores indicate more similar vectors; this does not guarantee factual relevance.")


if __name__ == "__main__":
    main()
