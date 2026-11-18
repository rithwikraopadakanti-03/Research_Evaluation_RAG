import re
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

class BM25Index:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.engine = BM25Okapi([tokenize(c.text) for c in self.chunks]) if self.chunks else None
    def search(self, query, top_k=5):
        if self.engine is None:
            return []
        scores = self.engine.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [{"chunk": self.chunks[i], "score": float(scores[i])}
                for i in order[:top_k] if scores[i] > 0]
