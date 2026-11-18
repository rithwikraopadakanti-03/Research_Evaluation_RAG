"""Document chunking utilities.

Provides two strategies:
- split_text: fast word-boundary chunking (original behaviour, now with word_count field)
- split_text_sentences: sentence-aware chunking that never cuts mid-sentence
"""
import hashlib
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single text chunk extracted from a document."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    source: str
    word_count: int = 0  # number of words in this chunk's body


# ---------------------------------------------------------------------------
# Word-boundary chunker (original strategy)
# ---------------------------------------------------------------------------

def split_text(text: str, doc_id: str, source: str, chunk_size: int = 700, overlap: int = 100) -> list[Chunk]:
    """Split *text* into overlapping word-boundary chunks.

    Args:
        text: Raw document text.
        doc_id: Stable document identifier.
        source: Human-readable source label (e.g. filename).
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        Ordered list of :class:`Chunk` objects.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    words = re.findall(r"\S+", text)
    step = chunk_size - overlap
    output: list[Chunk] = []
    for idx, start in enumerate(range(0, len(words), step)):
        word_slice = words[start : start + chunk_size]
        body = " ".join(word_slice).strip()
        if body:
            digest = hashlib.sha1(f"{doc_id}:{idx}:{body}".encode()).hexdigest()[:12]
            output.append(
                Chunk(
                    chunk_id=f"chunk-{digest}",
                    doc_id=doc_id,
                    text=body,
                    chunk_index=idx,
                    source=source,
                    word_count=len(word_slice),
                )
            )
        if start + chunk_size >= len(words):
            break
    return output


# ---------------------------------------------------------------------------
# Sentence-aware chunker (new strategy)
# ---------------------------------------------------------------------------

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Naively split *text* into sentences on sentence-ending punctuation."""
    sentences = _SENT_END.split(text.strip())
    # Keep non-empty sentences only
    return [s.strip() for s in sentences if s.strip()]


def split_text_sentences(
    text: str,
    doc_id: str,
    source: str,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[Chunk]:
    """Split *text* into chunks at sentence boundaries.

    Accumulates sentences until the next sentence would push the chunk over
    *chunk_size* words, then starts a new chunk.  An overlap of *overlap*
    words is prepended to the next chunk by back-tracking over previous
    sentences.  This preserves sentence integrity — no sentence is split in
    the middle.

    Args:
        text: Raw document text.
        doc_id: Stable document identifier.
        source: Human-readable source label (e.g. filename).
        chunk_size: Target maximum words per chunk.
        overlap: Approximate word overlap between consecutive chunks.

    Returns:
        Ordered list of :class:`Chunk` objects.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    output: list[Chunk] = []
    idx = 0
    sent_idx = 0  # current sentence pointer

    while sent_idx < len(sentences):
        current_words: list[str] = []
        chunk_sentences: list[str] = []

        # Build up a chunk sentence-by-sentence
        while sent_idx < len(sentences):
            sent_words = sentences[sent_idx].split()
            if current_words and len(current_words) + len(sent_words) > chunk_size:
                # This sentence would exceed the limit — stop here
                break
            current_words.extend(sent_words)
            chunk_sentences.append(sentences[sent_idx])
            sent_idx += 1

        body = " ".join(current_words).strip()
        if body:
            digest = hashlib.sha1(f"{doc_id}:s{idx}:{body}".encode()).hexdigest()[:12]
            output.append(
                Chunk(
                    chunk_id=f"chunk-{digest}",
                    doc_id=doc_id,
                    text=body,
                    chunk_index=idx,
                    source=source,
                    word_count=len(current_words),
                )
            )
            idx += 1

        # Back-track overlap: re-wind sent_idx to include overlapping words
        if overlap > 0 and sent_idx < len(sentences):
            overlap_words = 0
            back = len(chunk_sentences) - 1
            while back >= 0 and overlap_words < overlap:
                overlap_words += len(chunk_sentences[back].split())
                back -= 1
            # Rewind sent_idx by the number of overlap sentences
            overlap_sents = len(chunk_sentences) - 1 - back
            sent_idx = max(sent_idx - overlap_sents, sent_idx - len(chunk_sentences) + 1)

    return output
