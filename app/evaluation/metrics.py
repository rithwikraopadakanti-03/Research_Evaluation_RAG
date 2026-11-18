"""Retrieval and answer quality metrics.

All functions are pure and dependency-free (only stdlib).  They return ``None``
when the information needed to compute a metric is absent (e.g. no labels),
so callers can distinguish "not applicable" from a zero score.
"""
import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def tokens(s: str) -> list[str]:
    """Tokenise *s* into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def normalize(s: str) -> str:
    """Lower-case and collapse whitespace so passages can be matched inside chunks."""
    return " ".join(re.findall(r"\S+", (s or "").lower()))


# ---------------------------------------------------------------------------
# Label presence check
# ---------------------------------------------------------------------------

def has_labels(relevant_passages=None, relevant_chunk_ids=None) -> bool:
    """Return True when at least one relevance label is present."""
    return bool(relevant_passages) or bool(relevant_chunk_ids)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def retrieval_metrics(
    retrieved_chunks,
    relevant_passages=None,
    relevant_chunk_ids=None,
    k: int = 5,
) -> dict:
    """Compute retrieval quality metrics for a single query.

    A retrieved chunk is considered relevant when it contains any labelled
    passage (normalised substring match) or its ``chunk_id`` is in
    *relevant_chunk_ids*.  Recall is the share of labelled items covered by
    the top-*k* results.

    Returns ``None`` for every metric when the row carries no relevance labels.

    Metrics returned
    ----------------
    hit_at_k        : 1.0 if any relevant chunk appears in top-k, else 0.0
    recall_at_k     : fraction of labelled items covered
    mrr             : Mean Reciprocal Rank (rank of first relevant chunk)
    precision_at_k  : fraction of top-k results that are relevant
    ndcg_at_k       : Normalised Discounted Cumulative Gain at k
    """
    passages = [normalize(p) for p in (relevant_passages or []) if normalize(p)]
    ids = set(relevant_chunk_ids or [])
    if not passages and not ids:
        return {
            "hit_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "precision_at_k": None,
            "ndcg_at_k": None,
        }

    ranked = list(retrieved_chunks[:k])
    texts = [normalize(c.text) for c in ranked]

    # Relevance flag per retrieved chunk
    relevant_flags = [
        any(p in t for p in passages) or c.chunk_id in ids
        for c, t in zip(ranked, texts)
    ]

    # Coverage: how many labelled items appear in top-k
    covered = sum(any(p in t for t in texts) for p in passages) + len(
        ids & {c.chunk_id for c in ranked}
    )
    total = len(passages) + len(ids)

    first = next((rank for rank, flag in enumerate(relevant_flags, 1) if flag), None)

    return {
        "hit_at_k": float(first is not None),
        "recall_at_k": covered / total,
        "mrr": 1.0 / first if first else 0.0,
        "precision_at_k": sum(relevant_flags) / max(k, 1),
        "ndcg_at_k": ndcg_at_k(retrieved_chunks, relevant_passages, relevant_chunk_ids, k),
    }


# ---------------------------------------------------------------------------
# NDCG
# ---------------------------------------------------------------------------

def ndcg_at_k(
    retrieved_chunks,
    relevant_passages=None,
    relevant_chunk_ids=None,
    k: int = 5,
) -> float | None:
    """Normalised Discounted Cumulative Gain at *k*.

    Binary relevance: a chunk is relevant (gain = 1) if it matches any
    labelled passage or its chunk_id is listed; irrelevant chunks score 0.

    Returns ``None`` when no labels are present.
    """
    passages = [normalize(p) for p in (relevant_passages or []) if normalize(p)]
    ids = set(relevant_chunk_ids or [])
    if not passages and not ids:
        return None

    ranked = list(retrieved_chunks[:k])

    def is_relevant(chunk) -> int:
        t = normalize(chunk.text)
        return int(any(p in t for p in passages) or chunk.chunk_id in ids)

    gains = [is_relevant(c) for c in ranked]

    def dcg(gains_list: list[int]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains_list))

    actual_dcg = dcg(gains)
    # Ideal: sort gains descending (all relevant first)
    ideal_dcg = dcg(sorted(gains, reverse=True))
    if ideal_dcg == 0.0:
        return 0.0
    return round(actual_dcg / ideal_dcg, 6)


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between *prediction* and *reference* strings."""
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def semantic_similarity_to_context(answer: str, context: str) -> float:
    """Lightweight proxy for semantic coverage: token Jaccard similarity.

    Measures how much of the answer vocabulary is present in the context.
    No external dependencies — uses simple token overlap.  Values range
    [0, 1]: 1.0 means every answer token appears in the context.
    """
    a_tokens = set(tokens(answer))
    c_tokens = set(tokens(context))
    if not a_tokens:
        return 0.0
    intersection = a_tokens & c_tokens
    # Asymmetric Jaccard: answer tokens covered by context
    return len(intersection) / len(a_tokens)


def answer_metrics(answer: str, ground_truth: str, context: str) -> dict:
    """Cheap lexical answer metrics. These are heuristics, not proof of correctness.

    Metrics returned
    ----------------
    answer_f1               : token F1 against *ground_truth* (None when absent)
    lexical_context_overlap : fraction of answer tokens present in context
    semantic_sim_to_context : asymmetric token-Jaccard coverage (alias for above, more descriptive)
    """
    a_set = set(tokens(answer))
    c_set = set(tokens(context))
    return {
        "answer_f1": token_f1(answer, ground_truth) if ground_truth else None,
        "lexical_context_overlap": len(a_set & c_set) / max(len(a_set), 1),
        "semantic_sim_to_context": semantic_similarity_to_context(answer, context),
    }
