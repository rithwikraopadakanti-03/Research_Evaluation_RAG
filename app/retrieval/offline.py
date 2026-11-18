"""Deterministic stand-ins for the neural models.

Used by the unit tests and for smoke-testing the pipeline offline. Numbers produced with these
are NOT representative of real embedding or cross-encoder quality.
"""
import hashlib
import re

import numpy as np


class HashingEmbedder:
    def __init__(self, model_name="hashing", dim=256):
        self.model_name, self.dim, self.calls = model_name, dim, 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class OverlapReranker:
    def __init__(self, model_name="overlap"):
        self.model_name = model_name

    def rerank(self, query, items, top_k):
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = [(len(q & set(re.findall(r"[a-z0-9]+", it["chunk"].text.lower()))), it) for it in items]
        scored.sort(key=lambda p: p[0], reverse=True)
        return [{"chunk": it["chunk"], "score": float(s)} for s, it in scored[:top_k]]
