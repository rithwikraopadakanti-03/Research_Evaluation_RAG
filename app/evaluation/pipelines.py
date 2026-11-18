from itertools import product
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.core.config import get_settings

MAX_GRID_SIZE = 12
Retriever = Literal["dense", "bm25", "hybrid"]
Reranker = Literal["none", "cross_encoder"]


class PipelineConfig(BaseModel):
    name: Optional[str] = None
    chunk_size: int = Field(default=700, ge=20, le=3000)
    chunk_overlap: int = Field(default=100, ge=0)
    embedding_model: str = ""
    retriever: Retriever = "hybrid"
    reranker: Reranker = "none"
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _finish(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.embedding_model:
            self.embedding_model = get_settings().embedding_model
        if not self.name:
            rerank = "+rerank" if self.reranker != "none" else ""
            self.name = f"{self.retriever}{rerank}, chunk={self.chunk_size}, overlap={self.chunk_overlap}, k={self.top_k}"
        return self

    @property
    def index_key(self):
        return (self.chunk_size, self.chunk_overlap, self.embedding_model)


class GridSpec(BaseModel):
    chunk_sizes: list[int] = Field(default_factory=lambda: [700])
    chunk_overlap: int = Field(default=100, ge=0)
    retrievers: list[Retriever] = Field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    rerankers: list[Reranker] = Field(default_factory=lambda: ["none"])
    embedding_models: list[str] = Field(default_factory=lambda: [""])
    top_k: int = Field(default=5, ge=1, le=20)


def expand_grid(spec: GridSpec) -> list[PipelineConfig]:
    combos = list(product(spec.embedding_models, spec.chunk_sizes, spec.retrievers, spec.rerankers))
    if not combos:
        raise ValueError("Grid is empty")
    if len(combos) > MAX_GRID_SIZE:
        raise ValueError(f"Grid has {len(combos)} pipelines; the maximum is {MAX_GRID_SIZE}")
    return [
        PipelineConfig(chunk_size=size, chunk_overlap=min(spec.chunk_overlap, size - 1),
                       embedding_model=model, retriever=ret, reranker=rr, top_k=spec.top_k)
        for model, size, ret, rr in combos
    ]
