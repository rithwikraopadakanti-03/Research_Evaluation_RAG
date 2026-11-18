import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


class UploadTooLarge(ValueError):
    """Raised when an uploaded file exceeds the configured size limit."""


def content_doc_id(content: bytes) -> str:
    """Stable identifier derived from file bytes (same file -> same id)."""
    return hashlib.sha1(content).hexdigest()[:12]


def safe_filename(name: str) -> str:
    base = Path(name or "upload").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return (cleaned or "upload")[:100]


def save_upload(filename, content, destination, max_bytes, allowed=SUPPORTED):
    """Validate and store an upload. Returns (doc_id, display_name, path).

    Files are stored as <doc_id>_<sanitised name> so equal names never overwrite.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise ValueError(f"Unsupported file type: {suffix or 'none'}")
    if not content:
        raise ValueError("Uploaded file is empty")
    if len(content) > max_bytes:
        raise UploadTooLarge(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
    doc_id = content_doc_id(content)
    display = f"{safe_filename(Path(filename).stem)}{suffix}"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{doc_id}_{display}"
    target.write_bytes(content)
    return doc_id, display, target


def load_document(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(f"[Page {i}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages, 1))
    return path.read_text(encoding="utf-8", errors="replace")
