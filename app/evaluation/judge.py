import json
import logging
import re

from openai import OpenAI

from app.core.config import get_settings

log = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You grade a RAG answer using ONLY the supplied context. "
    'Reply with strict JSON: {"faithfulness": 1-5, "answer_relevance": 1-5, "reason": "<one sentence>"}. '
    "faithfulness: 5 = every claim is supported by the context, 1 = mostly unsupported. "
    "answer_relevance: 5 = fully answers the question, 1 = does not answer it. No text outside the JSON."
)
EMPTY = {"faithfulness": None, "answer_relevance": None, "judge_reason": None}


def parse_judge_json(text):
    """Parse the judge reply, tolerating code fences. Raises ValueError when invalid."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge reply")
    data = json.loads(match.group(0))
    scores = {}
    for key in ("faithfulness", "answer_relevance"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
            raise ValueError(f"{key} must be a number from 1 to 5")
        scores[key] = float(value)
    return {**scores, "judge_reason": str(data.get("reason", ""))[:300]}


class LLMJudge:
    """LLM-as-judge. Returns None scores (never zeros) when disabled or when the call fails."""

    def __init__(self, client=None, model=None):
        cfg = get_settings()
        configured = cfg.llm_mode == "openai_compatible" and bool(cfg.llm_api_key and cfg.llm_model)
        self.enabled = client is not None or configured
        self.model = model or cfg.llm_model or "judge-model"
        self.client = client
        if self.client is None and configured:
            self.client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url or None)

    def score(self, question, context, answer):
        if not self.enabled or self.client is None:
            return dict(EMPTY)
        prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
        for attempt in range(2):
            try:
                reply = self.client.chat.completions.create(
                    model=self.model, temperature=0,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                )
                return parse_judge_json(reply.choices[0].message.content)
            except Exception as exc:  # network errors and bad JSON both get one retry
                log.warning("Judge attempt %s failed: %s", attempt + 1, exc)
        return dict(EMPTY)
