"""Helpers for turning LLM output into verified benchmark rows."""
import json
import re

from app.evaluation.metrics import normalize

PROMPT = (
    "From the passage below write ONE question a user could ask, a short reference answer, "
    "and the exact answer-bearing phrase copied verbatim from the passage (at most 12 words). "
    'Reply with strict JSON: {"question": "...", "answer": "...", "passage": "..."}.\n\nPassage:\n'
)


def parse_example(reply_text, chunk_text):
    """Return a dataset row, or None when the LLM output is malformed or the phrase is not verbatim."""
    match = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", reply_text or ""), re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question, answer, passage = (str(data.get(k, "")).strip() for k in ("question", "answer", "passage"))
    if not (question and answer and passage):
        return None
    if normalize(passage) not in normalize(chunk_text) or len(passage.split()) > 12:
        return None
    return {"question": question, "ground_truth": answer, "relevant_passages": [passage]}
