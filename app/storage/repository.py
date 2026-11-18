"""SQLite-backed repository for documents, datasets and benchmark runs."""
import json
from pathlib import Path

from sqlalchemy import create_engine, text

from app.core.config import get_settings


class Repository:
    def __init__(self):
        url = get_settings().database_url
        if url.startswith("sqlite:///"):
            Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, future=True)
        with self.engine.begin() as c:
            c.execute(text(
                "CREATE TABLE IF NOT EXISTS benchmark_runs "
                "(run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL)"
            ))
            c.execute(text(
                "CREATE TABLE IF NOT EXISTS documents "
                "(doc_id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            ))
            c.execute(text(
                "CREATE TABLE IF NOT EXISTS datasets "
                "(dataset_id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL, "
                "rows INTEGER NOT NULL, created_at TEXT NOT NULL)"
            ))

    # ------------------------------------------------------------------
    # Benchmark runs
    # ------------------------------------------------------------------

    def save_run(self, result: dict) -> None:
        with self.engine.begin() as c:
            c.execute(
                text("INSERT OR REPLACE INTO benchmark_runs VALUES (:id,:created,:payload)"),
                {"id": result["run_id"], "created": result["created_at"], "payload": json.dumps(result)},
            )

    def get_run(self, run_id: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(
                text("SELECT payload FROM benchmark_runs WHERE run_id=:id"), {"id": run_id}
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_runs(self) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(
                text("SELECT run_id,created_at FROM benchmark_runs ORDER BY created_at DESC")
            ).fetchall()
        return [{"run_id": r[0], "created_at": r[1]} for r in rows]

    def delete_run(self, run_id: str) -> bool:
        with self.engine.begin() as c:
            result = c.execute(
                text("DELETE FROM benchmark_runs WHERE run_id=:id"), {"id": run_id}
            )
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, filename: str, path, created_at: str) -> None:
        with self.engine.begin() as c:
            c.execute(
                text("INSERT OR REPLACE INTO documents VALUES (:id,:name,:path,:created)"),
                {"id": doc_id, "name": filename, "path": str(path), "created": created_at},
            )

    def list_documents(self) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(
                text("SELECT doc_id,filename,path,created_at FROM documents ORDER BY created_at DESC")
            ).fetchall()
        return [{"doc_id": r[0], "filename": r[1], "path": r[2], "created_at": r[3]} for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        with self.engine.begin() as c:
            result = c.execute(
                text("DELETE FROM documents WHERE doc_id=:id"), {"id": doc_id}
            )
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def add_dataset(self, dataset_id: str, filename: str, path, rows: int, created_at: str) -> None:
        with self.engine.begin() as c:
            c.execute(
                text("INSERT OR REPLACE INTO datasets VALUES (:id,:name,:path,:rows,:created)"),
                {"id": dataset_id, "name": filename, "path": str(path), "rows": rows, "created": created_at},
            )

    def get_dataset(self, dataset_id: str) -> dict | None:
        with self.engine.connect() as c:
            r = c.execute(
                text("SELECT dataset_id,filename,path,rows,created_at FROM datasets WHERE dataset_id=:id"),
                {"id": dataset_id},
            ).fetchone()
        return {"dataset_id": r[0], "filename": r[1], "path": r[2], "rows": r[3], "created_at": r[4]} if r else None

    def list_datasets(self) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(
                text("SELECT dataset_id,filename,rows,created_at FROM datasets ORDER BY created_at DESC")
            ).fetchall()
        return [{"dataset_id": r[0], "filename": r[1], "rows": r[2], "created_at": r[3]} for r in rows]

    def delete_dataset(self, dataset_id: str) -> bool:
        with self.engine.begin() as c:
            result = c.execute(
                text("DELETE FROM datasets WHERE dataset_id=:id"), {"id": dataset_id}
            )
        return result.rowcount > 0
