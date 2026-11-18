from app.core.config import get_settings


class CrossEncoderReranker:
    """Re-scores candidate chunks with a cross-encoder (loaded lazily)."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None

    def rerank(self, query, items, top_k):
        if not items:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, it["chunk"].text) for it in items])
        ranked = sorted(zip(items, scores), key=lambda p: float(p[1]), reverse=True)[:top_k]
        return [{"chunk": it["chunk"], "score": float(s)} for it, s in ranked]


_factory = None
_cache = {}


def set_reranker_factory(factory):
    global _factory
    _factory = factory
    _cache.clear()


def get_reranker(model_name=None):
    name = model_name or get_settings().rerank_model
    if name not in _cache:
        _cache[name] = (_factory or CrossEncoderReranker)(name)
    return _cache[name]
