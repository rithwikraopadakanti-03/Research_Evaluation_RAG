import re

from openai import OpenAI

from app.core.config import get_settings

SYSTEM = ("Answer only from the supplied context. If it is insufficient, say so. "
          "Cite supporting chunks using [chunk_id]. Do not invent facts.")


class AnswerGenerator:
    def __init__(self, client=None):
        cfg = get_settings()
        self.cfg = cfg
        self.use_llm = client is not None or (
            cfg.llm_mode == "openai_compatible" and bool(cfg.llm_api_key and cfg.llm_model))
        self.client = client
        if self.client is None and self.use_llm:
            self.client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url or None)

    @property
    def mode(self):
        """'llm' when a real model writes answers, else 'demo_extractive' (top-1 chunk sentence)."""
        return "llm" if self.use_llm else "demo_extractive"

    def generate(self, question, retrieved):
        if self.use_llm:
            context = "\n\n".join(f"[{x['chunk'].chunk_id}] {x['chunk'].text}" for x in retrieved)
            result = self.client.chat.completions.create(
                model=self.cfg.llm_model or "model", temperature=0.1,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
            return result.choices[0].message.content or ""
        if not retrieved:
            return "I do not have enough information in the indexed documents to answer."
        sentences = re.split(r"(?<=[.!?])\s+", retrieved[0]["chunk"].text)
        q = set(re.findall(r"\w+", question.lower()))
        best = max(sentences, key=lambda s: len(q & set(re.findall(r"\w+", s.lower()))), default="")
        return f"{best.strip()} [{retrieved[0]['chunk'].chunk_id}]"
