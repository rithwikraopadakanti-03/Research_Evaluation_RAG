import json
import time
import uuid
from statistics import mean

from app.core.config import get_settings
from app.evaluation.judge import LLMJudge
from app.evaluation.metrics import answer_metrics, retrieval_metrics
from app.generation.answer import AnswerGenerator
from app.ingestion.chunking import split_text
from app.retrieval.embeddings import get_embedder
from app.retrieval.engine import RetrievalEngine
from app.retrieval.reranker import get_reranker
from app.storage.repository import Repository

# metric -> True when higher is better
LEADERBOARD_METRICS = {
    "mrr": True, "recall_at_k": True, "hit_at_k": True, "precision_at_k": True,
    "answer_f1": True, "faithfulness": True, "answer_relevance": True,
    "retrieval_ms": False, "total_ms": False,
}
SUMMARY_METRICS = ["hit_at_k", "recall_at_k", "mrr", "precision_at_k", "answer_f1",
                   "lexical_context_overlap", "faithfulness", "answer_relevance"]


def rank_pipelines(summaries, metric):
    """Sort by the chosen metric; ties go to higher recall, then lower search latency."""
    higher = LEADERBOARD_METRICS[metric]

    def key(s):
        value = s.get(f"avg_{metric}")
        recall = s.get("avg_recall_at_k")
        search_ms = (s.get("avg_retrieval_ms") or 0) + (s.get("avg_rerank_ms") or 0)
        primary = 0 if value is None else (-value if higher else value)
        return (value is None, primary, 0 if recall is None else -recall, search_ms)

    return sorted(summaries, key=key)


class BenchmarkRunner:
    def __init__(self, documents, repository=None, generator=None, judge=None):
        """documents: list of dicts with doc_id, filename, text."""
        self.documents = list(documents)
        self.repo = repository or Repository()
        self.generator = generator or AnswerGenerator()
        self.judge = judge or LLMJudge()

    @staticmethod
    def read_dataset_text(text):
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL line {n}: {e}")
            if not isinstance(row, dict):
                raise ValueError(f"Line {n} must be a JSON object")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"Line {n} needs a non-empty question")
            passages = row.get("relevant_passages", [])
            if not isinstance(passages, list) or not all(isinstance(p, str) for p in passages):
                raise ValueError(f"Line {n}: relevant_passages must be a list of strings")
            rows.append(row)
        if not rows:
            raise ValueError("Dataset is empty")
        return rows

    @classmethod
    def read_dataset(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.read_dataset_text(f.read())

    def _chunks_for(self, pipeline):
        out = []
        for doc in self.documents:
            out.extend(split_text(doc["text"], doc["doc_id"], doc["filename"],
                                  pipeline.chunk_size, pipeline.chunk_overlap))
        return out

    def run(self, dataset_rows, pipelines, metric="mrr", use_judge=False, dataset_id=None):
        if metric not in LEADERBOARD_METRICS:
            raise ValueError(f"Unknown leaderboard metric: {metric}")
        if not pipelines:
            raise ValueError("At least one pipeline is required")
        cfg = get_settings()
        run_id = str(uuid.uuid4())
        engines, traces, summaries = {}, [], []
        for pipe in pipelines:
            reused = pipe.index_key in engines
            if not reused:
                chunks = self._chunks_for(pipe)
                if not chunks:
                    raise ValueError("The uploaded documents contain no text to index")
                engines[pipe.index_key] = RetrievalEngine(chunks, get_embedder(pipe.embedding_model))
            engine = engines[pipe.index_key]
            reranker = get_reranker(cfg.rerank_model) if pipe.reranker == "cross_encoder" else None
            n = max(pipe.top_k, cfg.rerank_candidates) if reranker else pipe.top_k
            # Warm-up so one-off model loading is not counted in latency.
            warm = engine.retrieve(dataset_rows[0]["question"], pipe.retriever, n)
            if reranker:
                reranker.rerank(dataset_rows[0]["question"], warm, pipe.top_k)
            group = []
            for row in dataset_rows:
                t0 = time.perf_counter()
                found = engine.retrieve(row["question"], pipe.retriever, n)
                t1 = time.perf_counter()
                if reranker:
                    found = reranker.rerank(row["question"], found, pipe.top_k)
                t2 = time.perf_counter()
                answer = self.generator.generate(row["question"], found)
                t3 = time.perf_counter()
                context = "\n".join(x["chunk"].text for x in found)
                judged = self.judge.score(row["question"], context, answer) if use_judge else \
                    {"faithfulness": None, "answer_relevance": None, "judge_reason": None}
                chunks_found = [x["chunk"] for x in found]
                trace = {
                    "run_id": run_id, "pipeline": pipe.name, "question": row["question"], "answer": answer,
                    "retrieved_chunk_ids": [c.chunk_id for c in chunks_found],
                    "retrieved_sources": list(dict.fromkeys(c.source for c in chunks_found)),
                    "retrieval_ms": round((t1 - t0) * 1000, 3), "rerank_ms": round((t2 - t1) * 1000, 3),
                    "generation_ms": round((t3 - t2) * 1000, 3), "total_ms": round((t3 - t0) * 1000, 3),
                    **retrieval_metrics(chunks_found, row.get("relevant_passages"),
                                        row.get("relevant_chunk_ids"), pipe.top_k),
                    **answer_metrics(answer, row.get("ground_truth", ""), context),
                    "faithfulness": judged["faithfulness"], "answer_relevance": judged["answer_relevance"],
                    "judge_reason": judged["judge_reason"],
                }
                group.append(trace)
            traces.extend(group)
            summary = {
                "pipeline": pipe.name, "config": pipe.model_dump(), "queries": len(group),
                "n_chunks": len(engine.chunks), "index_build_s": round(engine.build_seconds, 3),
                "index_reused": reused,
            }
            for lat in ("retrieval_ms", "rerank_ms", "generation_ms", "total_ms"):
                summary[f"avg_{lat}"] = round(mean(t[lat] for t in group), 3)
            for m in SUMMARY_METRICS:
                vals = [t[m] for t in group if t.get(m) is not None]
                summary[f"avg_{m}"] = round(mean(vals), 4) if vals else None
            summaries.append(summary)
        ranked = rank_pipelines(summaries, metric)
        leaderboard = [{"rank": i, **s} for i, s in enumerate(ranked, 1)]
        top_value = leaderboard[0].get(f"avg_{metric}")
        best = {"pipeline": leaderboard[0]["pipeline"], "metric": metric, "value": top_value} \
            if top_value is not None else None
        result = {
            "run_id": run_id, "dataset_id": dataset_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metric": metric, "answer_mode": self.generator.mode,
            "judge_mode": "llm" if (use_judge and self.judge.enabled) else "off",
            "summaries": summaries, "leaderboard": leaderboard, "best_pipeline": best, "traces": traces,
        }
        self.repo.save_run(result)
        return result
