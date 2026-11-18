import csv
import io
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from app import services
from app.api.schemas import BenchmarkRequest
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ingestion.loaders import UploadTooLarge
from app.storage.repository import Repository

cfg = get_settings()
configure_logging()
log = logging.getLogger(__name__)
repo = Repository()

app = FastAPI(
    title=cfg.app_name,
    version="2.0.0",
    description=(
        "RAG Research & Evaluation Framework API — "
        "upload documents, run pipeline benchmarks, compare retrieval strategies."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_limited(file: UploadFile) -> bytes:
    data = await file.read(cfg.max_upload_bytes + 1)
    if len(data) > cfg.max_upload_bytes:
        raise HTTPException(413, f"File exceeds the {cfg.max_upload_mb} MB limit")
    return data


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Detailed health check with system status."""
    docs = repo.list_documents()
    datasets = repo.list_datasets()
    runs = repo.list_runs()
    return {
        "status": "ok",
        "app": cfg.app_name,
        "version": "2.0.0",
        "llm_mode": cfg.llm_mode,
        "embedding_model": cfg.embedding_model,
        "documents": len(docs),
        "datasets": len(datasets),
        "runs": len(runs),
    }


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@app.post("/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, TXT, or Markdown document to the knowledge base."""
    content = await _read_limited(file)
    try:
        return services.ingest_document(repo, file.filename or "", content)
    except UploadTooLarge as e:
        raise HTTPException(413, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/documents", tags=["Documents"])
def documents():
    """List all uploaded documents."""
    return [{k: v for k, v in d.items() if k != "path"} for d in repo.list_documents()]


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    """Delete a document and remove it from the knowledge base."""
    if not services.delete_document(repo, doc_id):
        raise HTTPException(404, f"Document not found: {doc_id}")
    return {"deleted": doc_id}


@app.get("/documents/{doc_id}/chunks", tags=["Documents"])
def document_chunks(doc_id: str, chunk_size: int = 700, chunk_overlap: int = 100, max_chunks: int = 10):
    """Preview how a document will be chunked."""
    try:
        return services.get_chunk_preview(repo, doc_id, chunk_size, chunk_overlap, max_chunks)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@app.post("/datasets/upload", tags=["Datasets"])
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a JSONL benchmark dataset."""
    content = await _read_limited(file)
    try:
        return services.ingest_dataset(repo, file.filename or "", content)
    except UploadTooLarge as e:
        raise HTTPException(413, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/datasets", tags=["Datasets"])
def datasets():
    """List all uploaded benchmark datasets."""
    return repo.list_datasets()


@app.delete("/datasets/{dataset_id}", tags=["Datasets"])
def delete_dataset(dataset_id: str):
    """Delete a benchmark dataset."""
    if not repo.delete_dataset(dataset_id):
        raise HTTPException(404, f"Dataset not found: {dataset_id}")
    return {"deleted": dataset_id}


# ---------------------------------------------------------------------------
# Live Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retriever: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")
    top_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=700, ge=50, le=3000)
    chunk_overlap: int = Field(default=100, ge=0)
    embedding_model: Optional[str] = None


@app.post("/query", tags=["Live Query"])
def live_query(request: QueryRequest):
    """Run a single question against the current knowledge base and return chunks + answer."""
    try:
        return services.query_live(
            repo,
            question=request.question,
            retriever=request.retriever,
            top_k=request.top_k,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model or "",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        log.exception("Live query failed")
        raise HTTPException(500, "Internal error during query")


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------

@app.post("/benchmark/run", tags=["Benchmark"])
def benchmark(request: BenchmarkRequest):
    """Execute a full pipeline benchmark grid and return the leaderboard."""
    try:
        return services.execute_benchmark(repo, request)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        log.exception("Benchmark failed")
        raise HTTPException(500, "Internal error while running the benchmark")


@app.get("/benchmark/runs", tags=["Benchmark"])
def runs():
    """List all past benchmark runs."""
    return repo.list_runs()


@app.get("/benchmark/runs/{run_id}", tags=["Benchmark"])
def run_detail(run_id: str):
    """Retrieve the full result of a benchmark run."""
    result = repo.get_run(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return result


@app.delete("/benchmark/runs/{run_id}", tags=["Benchmark"])
def delete_run(run_id: str):
    """Delete a benchmark run."""
    if not repo.delete_run(run_id):
        raise HTTPException(404, "Run not found")
    return {"deleted": run_id}


@app.get("/benchmark/runs/{run_id}/export", tags=["Benchmark"])
def export(run_id: str, format: str = "csv"):
    """Export a benchmark run as CSV or JSON."""
    result = repo.get_run(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    if format == "json":
        return JSONResponse(result)
    if format != "csv":
        raise HTTPException(400, "format must be csv or json")
    output = io.StringIO()
    rows = result["traces"]
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={run_id}.csv"},
    )
