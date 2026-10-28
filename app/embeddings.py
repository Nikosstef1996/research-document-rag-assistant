"""Local neural embeddings. The model is downloaded once, then cached by Hugging Face."""
import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    def __init__(self, model_id: str = MODEL_ID):
        from sentence_transformers import SentenceTransformer  # lazy heavy import
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device="cpu")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode_document(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self.model.encode_query(
            query, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False), dtype=np.float32)
