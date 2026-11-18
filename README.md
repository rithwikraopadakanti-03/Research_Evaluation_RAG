# RAG Research & Evaluation Framework

A local-first tool for comparing RAG pipelines. Upload documents and a labelled question set, run a grid of
configurations (chunk size, overlap, embedding model, retriever, reranker, top-k), and get a leaderboard that
names the best pipeline for **your** data, with retrieval quality, answer quality, and latency side by side.

Stack: Python, FastAPI, Streamlit, FAISS, BM25, sentence-transformers, SQLite.

## What it compares
| Dimension | Options |
|---|---|
| Retriever | dense (FAISS), BM25, hybrid (Reciprocal Rank Fusion) |
| Reranker | none, cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Chunking | any chunk size and overlap (in words) |
| Embedding model | any sentence-transformers model |

Indexes are built once per (chunk size, overlap, embedding model) and reused by every retriever in the grid.

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                        # Windows: Copy-Item .env.example .env
python scripts/create_sample_data.py
uvicorn app.api.main:app --reload                           # API docs: http://127.0.0.1:8000/docs
streamlit run dashboard/Home.py                             # dashboard: http://localhost:8501
```
Docker: `docker compose up --build` starts the API (8000) and the dashboard (8501).

## Run a benchmark from the command line
```bash
python scripts/create_sample_data.py
python scripts/run_benchmark.py            # add --rerank to include the cross-encoder
```
This writes `docs/RESULTS.md` from the actual run. The README contains no hand-typed numbers; results live in that
generated file. The bundled sample corpus is tiny (two short files), so use it as a functional check and benchmark
your own documents for real conclusions.

## Benchmark dataset (`.jsonl`, one object per line)
```json
{"question": "What is BM25?", "ground_truth": "A lexical ranking algorithm.", "relevant_passages": ["BM25 is a lexical retrieval algorithm"]}
```
- `relevant_passages`: short phrases copied from your documents. A retrieved chunk counts as relevant if it contains
  one (case and whitespace insensitive). Keep each phrase to roughly the overlap length (about 10 words or fewer)
  so chunk boundaries cannot split it. This keeps labels valid when you change the chunk size.
- `relevant_chunk_ids` is still accepted but changes with chunking, so prefer passages.
- `ground_truth` is optional and only used for answer F1.
- Draft a dataset with an LLM: `python scripts/generate_testset.py --n 20` (keeps only verbatim phrases; review the CSV).

## Metrics
- Retrieval: Hit@K, Recall@K, MRR, Precision@K (need `relevant_passages`).
- Answer: token F1 (needs `ground_truth`), lexical context overlap (heuristic), LLM-judge faithfulness and answer
  relevance (1-5, only when `LLM_MODE=openai_compatible`; otherwise empty, never zero).
- Latency per pipeline: retrieval, rerank, generation, and total, measured after a warm-up query. Index build time is
  reported separately.
- Leaderboard: rank by any metric; ties go to higher recall, then lower search latency. `best_pipeline` reports which
  metric it won on.

## LLM configuration
Default `LLM_MODE=demo` writes extractive answers from the top chunk, so the whole pipeline runs without a key. In
that mode answer metrics only reflect the top-1 chunk, and results say `answer_mode: demo_extractive`. For real answer
evaluation set `LLM_MODE=openai_compatible`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env` (Groq and other
OpenAI-compatible providers work). Never commit `.env`.
Check the judge against your own ratings: put rows in `data/judge_calibration.csv`
(see `data/judge_calibration.example.csv`) and run `python scripts/calibrate_judge.py`.

## API
- `GET /health`
- `POST /documents/upload`, `GET /documents`
- `POST /datasets/upload`, `GET /datasets`
- `POST /benchmark/run` with `dataset_id` and either `pipelines` (explicit list) or `grid` (expanded, max 12 pipelines)
- `GET /benchmark/runs`, `GET /benchmark/runs/{run_id}`, `GET /benchmark/runs/{run_id}/export?format=csv|json`

Uploads are limited by `MAX_UPLOAD_MB` (default 20), stored as `<hash>_<sanitised name>`, and benchmarks only read
datasets registered through the upload endpoint, never arbitrary server paths.

## Tests
`pytest -q` (CI runs it on every push). Tests use deterministic stand-in models, so no model download is needed.

## Limitations
- Small or auto-generated question sets give noisy rankings; use 50+ questions from your real documents.
- Passage labels can miss a relevant chunk that answers the question with different wording.
- Lexical overlap is a heuristic. The LLM judge must be validated against human ratings before you trust it.
- Chunking is word-based, and the dense index is rebuilt in memory for each run (no persistent vector store).
- No authentication or multi-user isolation: run it locally or behind your own access control.
