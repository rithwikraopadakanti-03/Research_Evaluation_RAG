import time

from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.index import DenseIndex

RETRIEVERS = ("dense", "bm25", "hybrid")


class RetrievalEngine:
    """Builds the dense and BM25 indexes once; any retriever can reuse them."""

    def __init__(self, chunks, embedder):
        self.chunks = list(chunks)
        start = time.perf_counter()
        self.dense = DenseIndex(embedder)
        self.dense.build(self.chunks)
        self.bm25 = BM25Index(self.chunks)
        self.build_seconds = time.perf_counter() - start

    def retrieve(self, query, retriever="hybrid", n=5):
        if retriever == "dense":
            return self.dense.search(query, n)
        if retriever == "bm25":
            return self.bm25.search(query, n)
        if retriever == "hybrid":
            return reciprocal_rank_fusion(
                [self.dense.search(query, n * 2), self.bm25.search(query, n * 2)], top_k=n
            )
        raise ValueError(f"Unknown retriever: {retriever}")
