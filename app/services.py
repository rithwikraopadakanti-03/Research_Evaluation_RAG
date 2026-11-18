"""Shared orchestration used by both the FastAPI app and the Streamlit dashboard."""
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.evaluation.pipelines import expand_grid, PipelineConfig
from app.evaluation.runner import BenchmarkRunner
from app.generation.answer import AnswerGenerator
from app.ingestion.chunking import split_text
from app.ingestion.loaders import content_doc_id, load_document, safe_filename, save_upload, UploadTooLarge
from app.retrieval.embeddings import get_embedder
from app.retrieval.engine import RetrievalEngine
from app.retrieval.reranker import get_reranker


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def ingest_document(repo, filename: str, content: bytes) -> dict:
    """Validate, store and register a document. Returns {doc_id, filename}."""
    cfg = get_settings()
    doc_id, display, path = save_upload(filename, content, cfg.uploads_dir, cfg.max_upload_bytes)
    repo.add_document(doc_id, display, path, _now())
    return {"doc_id": doc_id, "filename": display}


def delete_document(repo, doc_id: str) -> bool:
    """Remove a document from the DB and delete its file. Returns True if found."""
    docs = repo.list_documents()
    target = next((d for d in docs if d["doc_id"] == doc_id), None)
    if not target:
        return False
    path = Path(target["path"])
    if path.exists():
        path.unlink()
    repo.delete_document(doc_id)
    return True


# ---------------------------------------------------------------------------
# Dataset ingestion
# ---------------------------------------------------------------------------

def ingest_dataset(repo, filename: str, content: bytes) -> dict:
    """Validate, store and register a benchmark dataset. Returns {dataset_id, filename, rows}."""
    cfg = get_settings()
    if not (filename or "").lower().endswith(".jsonl"):
        raise ValueError("Datasets must be .jsonl files")
    if len(content) > cfg.max_upload_bytes:
        raise UploadTooLarge(f"File exceeds the {cfg.max_upload_mb} MB limit")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Dataset must be UTF-8 text")
    rows = BenchmarkRunner.read_dataset_text(text)  # validates before anything is stored
    dataset_id = content_doc_id(content)
    path = cfg.datasets_dir / f"{dataset_id}.jsonl"
    cfg.datasets_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    name = safe_filename(Path(filename).stem) + ".jsonl"
    repo.add_dataset(dataset_id, name, path, len(rows), _now())
    return {"dataset_id": dataset_id, "filename": name, "rows": len(rows)}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(repo) -> list[dict]:
    """Return a list of {doc_id, filename, text} for all stored documents."""
    docs = []
    for d in repo.list_documents():
        path = Path(d["path"])
        if path.exists():
            docs.append({"doc_id": d["doc_id"], "filename": d["filename"], "text": load_document(path)})
    return docs


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def execute_benchmark(repo, request) -> dict:
    """Run a full benchmark and return the result dict."""
    dataset = repo.get_dataset(request.dataset_id)
    if not dataset:
        raise LookupError(f"Dataset not found: {request.dataset_id}")
    corpus = load_corpus(repo)
    if not corpus:
        raise ValueError("Upload at least one document before benchmarking.")
    pipelines = request.pipelines or expand_grid(request.grid)
    rows = BenchmarkRunner.read_dataset(dataset["path"])
    runner = BenchmarkRunner(corpus, repo)
    return runner.run(rows, pipelines, request.metric, request.use_judge, request.dataset_id)


# ---------------------------------------------------------------------------
# Live single-query execution
# ---------------------------------------------------------------------------

def query_live(
    repo,
    question: str,
    retriever: str = "hybrid",
    top_k: int = 5,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    embedding_model: str = "",
) -> dict:
    """Execute a single live query against the indexed corpus.

    Returns a dict with:
        chunks  : list of {rank, source, score, text, chunk_id}
        answer  : generated answer string
        latency_ms : total wall-clock time in milliseconds
    """
    import time

    cfg = get_settings()
    corpus = load_corpus(repo)
    if not corpus:
        raise ValueError("Upload at least one document first.")

    em = embedding_model or cfg.embedding_model
    all_chunks = []
    for doc in corpus:
        all_chunks.extend(split_text(doc["text"], doc["doc_id"], doc["filename"], chunk_size, chunk_overlap))

    if not all_chunks:
        raise ValueError("Documents contain no indexable text.")

    embedder = get_embedder(em)
    engine = RetrievalEngine(all_chunks, embedder)

    reranker = None
    n = top_k
    if retriever == "hybrid+rerank":
        retriever = "hybrid"
        reranker = get_reranker(cfg.rerank_model)
        n = max(top_k, cfg.rerank_candidates)

    t0 = time.perf_counter()
    found = engine.retrieve(question, retriever, n)
    if reranker:
        found = reranker.rerank(question, found, top_k)
    generator = AnswerGenerator()
    answer = generator.generate(question, found)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    chunks_out = [
        {
            "rank": i,
            "source": r["chunk"].source,
            "score": round(r["score"], 4),
            "text": r["chunk"].text,
            "chunk_id": r["chunk"].chunk_id,
            "word_count": r["chunk"].word_count,
        }
        for i, r in enumerate(found[:top_k], 1)
    ]
    return {"chunks": chunks_out, "answer": answer, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# Chunk preview for a single document
# ---------------------------------------------------------------------------

def get_chunk_preview(
    repo,
    doc_id: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    max_chunks: int = 10,
) -> list[dict]:
    """Return the first *max_chunks* chunks of a document for preview purposes."""
    docs = repo.list_documents()
    target = next((d for d in docs if d["doc_id"] == doc_id), None)
    if not target:
        raise LookupError(f"Document not found: {doc_id}")
    path = Path(target["path"])
    if not path.exists():
        raise FileNotFoundError(f"File missing on disk: {path}")
    text = load_document(path)
    chunks = split_text(text, doc_id, target["filename"], chunk_size, chunk_overlap)
    return [
        {
            "index": c.chunk_index,
            "chunk_id": c.chunk_id,
            "word_count": c.word_count,
            "preview": c.text[:200] + ("…" if len(c.text) > 200 else ""),
        }
        for c in chunks[:max_chunks]
    ]
