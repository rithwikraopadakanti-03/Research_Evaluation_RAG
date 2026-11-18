import faiss


class DenseIndex:
    def __init__(self, embedder):
        self.embedder, self.chunks, self.index = embedder, [], None

    def build(self, chunks):
        self.chunks = list(chunks)
        if self.chunks:
            vectors = self.embedder.encode([c.text for c in self.chunks])
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)

    def search(self, query, top_k=5):
        if self.index is None:
            return []
        scores, ids = self.index.search(self.embedder.encode([query]), min(top_k, len(self.chunks)))
        return [{"chunk": self.chunks[int(i)], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]
