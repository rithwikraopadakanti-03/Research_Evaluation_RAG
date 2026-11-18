import numpy as np
from app.core.config import get_settings


class SentenceTransformerEmbedder:
    """Lazy wrapper so importing the app never downloads a model."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None

    def encode(self, texts):
        if not texts:
            return np.empty((0, 0), dtype="float32")
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(
            self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )


_factory = None
_cache = {}


def set_embedder_factory(factory):
    """Replace how embedders are built (used by tests and offline smoke runs)."""
    global _factory
    _factory = factory
    _cache.clear()


def get_embedder(model_name=None):
    name = model_name or get_settings().embedding_model
    if name not in _cache:
        _cache[name] = (_factory or SentenceTransformerEmbedder)(name)
    return _cache[name]
