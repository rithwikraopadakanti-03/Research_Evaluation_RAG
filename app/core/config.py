from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Research & Evaluation Framework"
    app_env: str = "local"
    database_url: str = "sqlite:///./artifacts/rag_eval.db"
    artifact_dir: Path = Path("./artifacts")
    data_dir: Path = Path("./data")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    chunk_size: int = 700
    chunk_overlap: int = 100
    default_top_k: int = 5
    max_upload_mb: int = 20
    llm_mode: str = "demo"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    log_level: str = "INFO"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    def prepare_dirs(self):
        for folder in (self.artifact_dir, self.data_dir, self.uploads_dir, self.datasets_dir):
            folder.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings():
    s = Settings()
    s.prepare_dirs()
    return s
