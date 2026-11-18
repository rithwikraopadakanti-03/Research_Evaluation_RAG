"""RAG Research & Evaluation Lab — Industry-grade Streamlit dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import services
from app.api.schemas import BenchmarkRequest
from app.core.config import get_settings
from app.evaluation.pipelines import GridSpec, MAX_GRID_SIZE, PipelineConfig, expand_grid
from app.evaluation.runner import LEADERBOARD_METRICS
from app.ingestion.loaders import UploadTooLarge
from app.storage.repository import Repository

# ---------------------------------------------------------------------------
# Page config & global state
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Research Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = get_settings()
repo = Repository()

# ---------------------------------------------------------------------------
# CSS — clean professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 16px 20px;
        border-left: 4px solid #4f8ef7; margin-bottom: 8px;
    }
    .rank-gold   { color: #FFD700; font-size: 1.3rem; }
    .rank-silver { color: #C0C0C0; font-size: 1.3rem; }
    .rank-bronze { color: #CD7F32; font-size: 1.3rem; }
    .chunk-card  {
        background: #f1f3f5; border-radius: 8px; padding: 12px 16px;
        margin-bottom: 8px; border-left: 3px solid #6c757d;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown("## 🧪")
with col_title:
    st.markdown("## RAG Research & Evaluation Lab")
    st.caption(
        "Upload documents · Run pipeline benchmarks · Compare retrieval strategies · Find the best RAG configuration"
    )
st.divider()

# ---------------------------------------------------------------------------
# Leaderboard display columns
# ---------------------------------------------------------------------------
LEADERBOARD_COLS = [
    "rank", "pipeline",
    "avg_hit_at_k", "avg_recall_at_k", "avg_mrr", "avg_ndcg_at_k",
    "avg_precision_at_k", "avg_answer_f1",
    "avg_faithfulness", "avg_answer_relevance",
    "avg_retrieval_ms", "avg_total_ms",
    "n_chunks", "index_build_s",
]

METRIC_LABELS = {
    "avg_hit_at_k": "Hit@K",
    "avg_recall_at_k": "Recall@K",
    "avg_mrr": "MRR",
    "avg_ndcg_at_k": "NDCG@K",
    "avg_precision_at_k": "Precision@K",
    "avg_answer_f1": "Answer F1",
    "avg_faithfulness": "Faithfulness",
    "avg_answer_relevance": "Answer Relevance",
    "avg_retrieval_ms": "Retrieval (ms)",
    "avg_total_ms": "Total (ms)",
}

RANK_MEDAL = {1: "🥇", 2: "🥈", 3: "🥉"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_rank(r):
    return RANK_MEDAL.get(r, str(r))


def _color_delta(val):
    if val is None:
        return ""
    if val > 0:
        return "color: green"
    if val < 0:
        return "color: red"
    return ""


def show_leaderboard(result: dict) -> pd.DataFrame:
    best = result.get("best_pipeline")
    judge = result.get("judge_mode", "off")

    c1, c2 = st.columns([3, 1])
    with c1:
        if best:
            st.success(
                f"🏆 **Best pipeline** by **{best['metric'].upper()}**: "
                f"`{best['pipeline']}` → score **{best['value']:.4f}**"
            )
        else:
            st.warning(
                "No retrieval labels found — add `relevant_passages` to the dataset to rank by retrieval quality."
            )
    with c2:
        st.info(
            f"Answer: **{result.get('answer_mode', '?')}**  \n"
            f"Judge: **{judge}**"
        )

    frame = pd.DataFrame(result["leaderboard"])
    display_cols = [c for c in LEADERBOARD_COLS if c in frame.columns]
    styled = frame[display_cols].copy()
    if "rank" in styled.columns:
        styled["rank"] = styled["rank"].apply(_fmt_rank)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    return frame


def _radar_chart(frame: pd.DataFrame, top_n: int = 3) -> go.Figure | None:
    radar_metrics = [
        "avg_hit_at_k", "avg_recall_at_k", "avg_mrr",
        "avg_precision_at_k", "avg_answer_f1",
    ]
    cols_present = [c for c in radar_metrics if c in frame.columns]
    if not cols_present:
        return None
    top = frame.head(top_n).copy()
    fig = go.Figure()
    for _, row in top.iterrows():
        vals = [row.get(c) or 0 for c in cols_present]
        vals.append(vals[0])
        labels = [METRIC_LABELS.get(c, c) for c in cols_present]
        labels.append(labels[0])
        fig.add_trace(go.Scatterpolar(r=vals, theta=labels, fill="toself", name=str(row["pipeline"])[:40]))
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
        title=f"Top-{top_n} Pipelines — Radar Comparison",
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — experiment controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Experiment Controls")

    mode = st.radio("Pipeline mode", ["Grid search", "Single pipeline"], horizontal=True)

    with st.expander("📐 Retrieval settings", expanded=True):
        top_k = st.slider("Top K", 1, 15, cfg.default_top_k)
        if mode == "Grid search":
            # NOTE: `sizes` holds the chunk sizes chosen here. Do not reuse this name elsewhere
            # in the script, or the later assignment silently replaces your selection.
            sizes = st.multiselect("Chunk sizes (words)", [60, 120, 200, 400, 700, 1000], [200, 700])
            overlap = st.slider("Overlap (words)", 0, 200, min(cfg.chunk_overlap, 100), 10)
            retrievers = st.multiselect("Retrievers", ["dense", "bm25", "hybrid"], ["dense", "bm25", "hybrid"])
            rerankers = st.multiselect("Rerankers", ["none", "cross_encoder"], ["none"])
        else:
            size = st.slider("Chunk size (words)", 50, 1200, cfg.chunk_size, 50)
            overlap_s = st.slider("Overlap (words)", 0, 300, min(cfg.chunk_overlap, 199), 10)
            retriever = st.selectbox("Retriever", ["hybrid", "dense", "bm25"])
            reranker = st.selectbox("Reranker", ["none", "cross_encoder"])

    with st.expander("🤖 Model settings", expanded=False):
        if mode == "Grid search":
            embedding = st.text_input("Embedding model", cfg.embedding_model)
        else:
            embedding_s = st.text_input("Embedding model", cfg.embedding_model)

    metric = st.selectbox("Rank leaderboard by", list(LEADERBOARD_METRICS))
    use_judge = st.checkbox("Use LLM judge (requires LLM_MODE=openai_compatible)")

    st.divider()
    st.caption(f"**Generation:** `{cfg.llm_mode}`")
    st.caption(f"**Env:** `{cfg.app_env}`")
    st.caption("v2.0.0 · RAG Research Lab")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📚 Knowledge Base",
    "🚀 Run Experiment",
    "📊 Results",
    "🔍 Compare Runs",
    "💬 Live Query",
    "ℹ️ About",
])

# ===========================================================================
# TAB 1 — Knowledge Base
# ===========================================================================
with t1:
    st.subheader("📚 Knowledge Base")

    # --- Quick load sample data ---
    sample_corpus_dir = Path(__file__).resolve().parents[1] / "data" / "sample_corpus"
    sample_dataset_path = Path(__file__).resolve().parents[1] / "data" / "sample_questions.jsonl"

    sample_col1, sample_col2 = st.columns([2, 3])
    with sample_col1:
        if st.button("⚡ Load Sample Data", type="secondary", help="Loads the built-in RAG corpus and Q&A dataset"):
            loaded_docs, loaded_ds = 0, 0
            if sample_corpus_dir.exists():
                for f in sample_corpus_dir.iterdir():
                    if f.suffix in {".md", ".txt", ".pdf"}:
                        try:
                            services.ingest_document(repo, f.name, f.read_bytes())
                            loaded_docs += 1
                        except Exception:
                            pass
            if sample_dataset_path.exists():
                try:
                    services.ingest_dataset(repo, sample_dataset_path.name, sample_dataset_path.read_bytes())
                    loaded_ds += 1
                except Exception:
                    pass
            st.success(f"Loaded {loaded_docs} document(s) and {loaded_ds} dataset(s).")
            st.rerun()

    # --- Upload documents ---
    st.markdown("#### Source Documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("💾 Save Documents", disabled=not uploads):
        for f in uploads:
            try:
                services.ingest_document(repo, f.name, f.getvalue())
                st.success(f"✅ {f.name}")
            except (ValueError, UploadTooLarge) as e:
                st.error(f"❌ {f.name}: {e}")
        st.rerun()

    # --- Document list ---
    docs = repo.list_documents()
    st.metric("Total documents", len(docs))
    if docs:
        df_docs = pd.DataFrame(docs)
        # File sizes on disk (named file_sizes_kb so it can never clash with the chunk-size selection)
        file_sizes_kb = []
        for d in docs:
            p = Path(d["path"])
            file_sizes_kb.append(round(p.stat().st_size / 1024, 1) if p.exists() else 0)
        df_docs["size_kb"] = file_sizes_kb
        st.dataframe(
            df_docs[["filename", "size_kb", "created_at", "doc_id"]].rename(columns={
                "filename": "File", "size_kb": "Size (KB)", "created_at": "Uploaded", "doc_id": "ID"
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Delete controls
        col_del1, col_del2 = st.columns([2, 2])
        with col_del1:
            del_id = st.selectbox("Select doc to delete", [d["doc_id"] for d in docs],
                                  format_func=lambda x: next(d["filename"] for d in docs if d["doc_id"] == x))
        with col_del2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete Selected Document"):
                services.delete_document(repo, del_id)
                st.success("Document deleted.")
                st.rerun()

        # --- Chunk preview ---
        with st.expander("🔎 Chunk Preview — see how your document will be split"):
            prev_doc = st.selectbox("Preview document", [d["doc_id"] for d in docs],
                                    format_func=lambda x: next(d["filename"] for d in docs if d["doc_id"] == x),
                                    key="chunk_prev_doc")
            prev_size = st.slider("Chunk size for preview", 50, 1200, cfg.chunk_size, 50, key="prev_size")
            prev_overlap = st.slider("Overlap for preview", 0, 300, 100, 10, key="prev_overlap")
            try:
                previews = services.get_chunk_preview(repo, prev_doc, prev_size, prev_overlap, max_chunks=8)
                if previews:
                    df_prev = pd.DataFrame(previews).rename(columns={
                        "index": "#", "word_count": "Words", "preview": "Text Preview", "chunk_id": "Chunk ID"
                    })
                    st.dataframe(df_prev[["#", "Words", "Text Preview"]], use_container_width=True, hide_index=True)
                    st.caption(f"Showing first {len(previews)} of total chunks at chunk_size={prev_size}, overlap={prev_overlap}")
            except Exception as e:
                st.error(str(e))

    # --- Datasets ---
    st.markdown("#### Benchmark Datasets")
    st.caption(
        'One JSON object per line: `{"question": "...", "ground_truth": "...", "relevant_passages": ["..."]}`'
    )
    ds_file = st.file_uploader("Upload dataset (.jsonl)", type=["jsonl"], label_visibility="collapsed")
    if st.button("💾 Save Dataset", disabled=not ds_file):
        try:
            info = services.ingest_dataset(repo, ds_file.name, ds_file.getvalue())
            st.success(f"✅ Saved {info['rows']} questions from `{info['filename']}`.")
            st.rerun()
        except (ValueError, UploadTooLarge) as e:
            st.error(str(e))

    datasets = repo.list_datasets()
    if datasets:
        st.dataframe(
            pd.DataFrame(datasets).rename(columns={
                "filename": "File", "rows": "Questions", "created_at": "Uploaded", "dataset_id": "ID"
            })[["File", "Questions", "Uploaded"]],
            use_container_width=True,
            hide_index=True,
        )

# ===========================================================================
# TAB 2 — Run Experiment
# ===========================================================================
with t2:
    st.subheader("🚀 Run Experiment")

    docs = repo.list_documents()
    datasets = repo.list_datasets()

    # Pre-flight checks
    if not docs:
        st.warning("⚠️ No documents in the knowledge base. Upload documents on the **Knowledge Base** tab first.")
    if not datasets:
        st.warning("⚠️ No benchmark dataset loaded. Upload a `.jsonl` dataset on the **Knowledge Base** tab first.")

    labels = {
        f"{d['filename']}  ({d['rows']} questions)": d["dataset_id"]
        for d in datasets
    }
    choice = st.selectbox("Select dataset", list(labels)) if labels else None

    # Pipeline count indicator
    if mode == "Grid search" and labels:
        grid_sizes = sizes or [cfg.chunk_size]
        grid_retrievers = retrievers or ["hybrid"]
        grid_rerankers = rerankers or ["none"]
        pipeline_count = len(grid_sizes) * len(grid_retrievers) * len(grid_rerankers)
        col_pc1, col_pc2 = st.columns([1, 3])
        with col_pc1:
            st.metric("Pipelines to run", pipeline_count)
        with col_pc2:
            # The maximum check must come first, otherwise it can never be reached.
            if pipeline_count > MAX_GRID_SIZE:
                st.error(f"❌ Exceeds maximum of {MAX_GRID_SIZE} pipelines. Reduce options.")
            elif pipeline_count > 8:
                st.warning(f"⚠️ {pipeline_count} pipelines — this may take a while.")
            if overlap >= min(grid_sizes):
                st.warning(
                    f"⚠️ Overlap ({overlap}) is not smaller than your smallest chunk size ({min(grid_sizes)}). "
                    "It will be capped at chunk size − 1, which makes chunks almost identical. "
                    "Lower the overlap (10 works well for small chunks)."
                )

    can_run = bool(docs and choice)
    if st.button("▶️ Run Benchmark", type="primary", disabled=not can_run):
        with st.spinner("Running benchmarks… (first run downloads the embedding model)"):
            try:
                if mode == "Grid search":
                    request = BenchmarkRequest(
                        dataset_id=labels[choice],
                        metric=metric,
                        use_judge=use_judge,
                        grid=GridSpec(
                            chunk_sizes=sizes or [cfg.chunk_size],
                            chunk_overlap=overlap,
                            retrievers=retrievers or ["hybrid"],
                            rerankers=rerankers or ["none"],
                            embedding_models=[embedding],
                            top_k=top_k,
                        ),
                    )
                else:
                    request = BenchmarkRequest(
                        dataset_id=labels[choice],
                        metric=metric,
                        use_judge=use_judge,
                        pipelines=[
                            PipelineConfig(
                                chunk_size=size,
                                chunk_overlap=min(overlap_s, size - 1),
                                retriever=retriever,
                                reranker=reranker,
                                embedding_model=embedding_s,
                                top_k=top_k,
                            )
                        ],
                    )
                result = services.execute_benchmark(repo, request)
                st.session_state["latest_run"] = result
                st.success("✅ Benchmark complete! Open the **Results** tab.")
            except (ValueError, LookupError) as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ===========================================================================
# TAB 3 — Results
# ===========================================================================
with t3:
    st.subheader("📊 Results")

    runs_list = repo.list_runs()
    selected_run = (
        st.selectbox("Select run", [r["run_id"] for r in runs_list],
                     format_func=lambda x: f"{x[:8]}… ({next((r['created_at'][:19] for r in runs_list if r['run_id'] == x), '')})")
        if runs_list else None
    )
    result = repo.get_run(selected_run) if selected_run else st.session_state.get("latest_run")

    if result:
        frame = show_leaderboard(result)

        # --- KPI cards ---
        st.markdown("#### Key Metrics")
        kpi_cols = st.columns(4)
        for col, (metric_key, label) in zip(
            kpi_cols,
            [("avg_mrr", "Best MRR"), ("avg_hit_at_k", "Best Hit@K"),
             ("avg_recall_at_k", "Best Recall@K"), ("avg_total_ms", "Fastest (ms)")],
        ):
            if metric_key in frame.columns:
                vals = frame[metric_key].dropna()
                if not vals.empty:
                    best_val = vals.min() if "ms" in metric_key else vals.max()
                    best_pipe = frame.loc[frame[metric_key] == best_val, "pipeline"].iloc[0]
                    col.metric(label, f"{best_val:.4f}", help=f"Pipeline: {best_pipe}")

        # --- Radar chart ---
        st.markdown("#### Pipeline Comparison")
        chart_cols = st.columns([2, 1])
        with chart_cols[0]:
            radar = _radar_chart(frame)
            if radar:
                st.plotly_chart(radar, use_container_width=True)

        with chart_cols[1]:
            chart_metric = st.selectbox(
                "Bar chart metric",
                [k for k in METRIC_LABELS if k in frame.columns],
                format_func=lambda x: METRIC_LABELS.get(x, x),
            )
            chart_data = frame[["pipeline", chart_metric]].dropna()
            if not chart_data.empty:
                fig = px.bar(
                    chart_data,
                    x="pipeline",
                    y=chart_metric,
                    title=METRIC_LABELS.get(chart_metric, chart_metric),
                    color=chart_metric,
                    color_continuous_scale="Blues",
                )
                fig.update_layout(xaxis_tickangle=-30, showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)

        # --- Per-question traces ---
        with st.expander("🔬 Per-question traces"):
            traces_df = pd.DataFrame(result["traces"])
            search_q = st.text_input("Filter by question keyword", key="trace_filter")
            if search_q:
                traces_df = traces_df[traces_df["question"].str.contains(search_q, case=False, na=False)]
            st.dataframe(traces_df, use_container_width=True, hide_index=True)

            # Answer preview
            if not traces_df.empty:
                pick = st.selectbox("Preview answer for question", traces_df["question"].tolist(), key="trace_pick")
                row = traces_df[traces_df["question"] == pick].iloc[0]
                st.markdown(f"**Q:** {row['question']}")
                st.markdown(f"**A:** {row.get('answer', '—')}")
                st.caption(
                    f"MRR: {row.get('mrr', '—')} | Hit@K: {row.get('hit_at_k', '—')} | "
                    f"F1: {row.get('answer_f1', '—')} | Latency: {row.get('total_ms', '—')} ms"
                )

        # --- Exports ---
        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button(
                "⬇️ Export Traces (CSV)",
                pd.DataFrame(result["traces"]).to_csv(index=False),
                file_name=f"{result['run_id']}.csv",
                mime="text/csv",
            )
        with exp2:
            st.download_button(
                "⬇️ Export Report (JSON)",
                json.dumps(result, indent=2),
                file_name=f"{result['run_id']}.json",
                mime="application/json",
            )
    else:
        st.info("Run an experiment on the **Run Experiment** tab to see results here.")

# ===========================================================================
# TAB 4 — Compare Runs
# ===========================================================================
with t4:
    st.subheader("🔍 Compare Runs")

    ids = [r["run_id"] for r in repo.list_runs()]
    if len(ids) < 2:
        st.info("Run at least **two** experiments to compare them here.")
    else:
        def _run_label(rid):
            ts = next((r["created_at"][:19] for r in repo.list_runs() if r["run_id"] == rid), rid)
            return f"{rid[:8]}… ({ts})"

        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("Run A (baseline)", ids, 0, format_func=_run_label)
        with c2:
            b = st.selectbox("Run B (new)", ids, min(1, len(ids) - 1), format_func=_run_label)

        cmp_metrics = st.multiselect(
            "Metrics to compare",
            ["avg_mrr", "avg_recall_at_k", "avg_hit_at_k", "avg_ndcg_at_k", "avg_precision_at_k", "avg_total_ms"],
            default=["avg_mrr", "avg_recall_at_k", "avg_hit_at_k"],
        )

        if a and b and a != b:
            ra, rb = repo.get_run(a), repo.get_run(b)
            if ra and rb:
                fa = pd.DataFrame(ra.get("summaries", ra.get("leaderboard", [])))[
                    ["pipeline"] + [m for m in cmp_metrics if m in pd.DataFrame(ra.get("summaries", ra.get("leaderboard", []))).columns]
                ]
                fb = pd.DataFrame(rb.get("summaries", rb.get("leaderboard", [])))[
                    ["pipeline"] + [m for m in cmp_metrics if m in pd.DataFrame(rb.get("summaries", rb.get("leaderboard", []))).columns]
                ]
                fa = fa.rename(columns={m: f"{m}_A" for m in cmp_metrics if m in fa.columns})
                fb = fb.rename(columns={m: f"{m}_B" for m in cmp_metrics if m in fb.columns})
                merged = fa.merge(fb, on="pipeline", how="outer")

                for m in cmp_metrics:
                    col_a, col_b = f"{m}_A", f"{m}_B"
                    if col_a in merged.columns and col_b in merged.columns:
                        merged[f"Δ {METRIC_LABELS.get(m, m)}"] = (merged[col_b] - merged[col_a]).round(4)

                st.dataframe(merged, use_container_width=True, hide_index=True)
                st.caption("Positive Δ = Run B improved. Negative Δ = Run B regressed.")

                # Visual comparison
                if cmp_metrics:
                    sel_m = st.selectbox("Chart metric", cmp_metrics, format_func=lambda x: METRIC_LABELS.get(x, x))
                    col_a_name = f"{sel_m}_A"
                    col_b_name = f"{sel_m}_B"
                    if col_a_name in merged.columns and col_b_name in merged.columns:
                        fig_cmp = go.Figure()
                        fig_cmp.add_bar(x=merged["pipeline"], y=merged[col_a_name], name="Run A")
                        fig_cmp.add_bar(x=merged["pipeline"], y=merged[col_b_name], name="Run B")
                        fig_cmp.update_layout(
                            barmode="group",
                            title=f"{METRIC_LABELS.get(sel_m, sel_m)} — A vs B",
                            xaxis_tickangle=-30,
                            height=380,
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)

# ===========================================================================
# TAB 5 — Live Query
# ===========================================================================
with t5:
    st.subheader("💬 Live Query")
    st.caption("Ask a question directly against the indexed knowledge base and inspect retrieved chunks in real time.")

    docs_lq = repo.list_documents()
    if not docs_lq:
        st.warning("Upload documents in the **Knowledge Base** tab first.")
    else:
        lq_col1, lq_col2 = st.columns([3, 1])
        with lq_col1:
            question = st.text_area("Your question", placeholder="e.g. How does BM25 differ from dense retrieval?", height=80)
        with lq_col2:
            lq_retriever = st.selectbox("Retriever", ["hybrid", "dense", "bm25"], key="lq_ret")
            lq_top_k = st.slider("Top K", 1, 10, 5, key="lq_k")
            lq_chunk_size = st.slider("Chunk size", 100, 1000, cfg.chunk_size, 50, key="lq_cs")

        if st.button("🔎 Search", type="primary", disabled=not (question and question.strip())):
            with st.spinner("Searching…"):
                try:
                    result_lq = services.query_live(
                        repo,
                        question=question.strip(),
                        retriever=lq_retriever,
                        top_k=lq_top_k,
                        chunk_size=lq_chunk_size,
                        chunk_overlap=min(100, lq_chunk_size // 5),
                    )
                    st.session_state["live_result"] = result_lq
                except Exception as e:
                    st.error(str(e))

        lq_res = st.session_state.get("live_result")
        if lq_res:
            st.markdown(f"⏱️ **Latency:** {lq_res['latency_ms']} ms")
            st.markdown("#### 💡 Generated Answer")
            st.info(lq_res["answer"])

            st.markdown(f"#### 📄 Retrieved Chunks ({len(lq_res['chunks'])})")
            for chunk in lq_res["chunks"]:
                with st.expander(
                    f"#{chunk['rank']} · {chunk['source']} · {chunk['word_count']} words · score {chunk['score']:.4f}"
                ):
                    st.markdown(chunk["text"])
                    st.caption(f"Chunk ID: `{chunk['chunk_id']}`")

# ===========================================================================
# TAB 6 — About
# ===========================================================================
with t6:
    st.subheader("ℹ️ About this Lab")

    st.markdown(
        """
        **RAG Research & Evaluation Lab** is an end-to-end framework for benchmarking
        Retrieval-Augmented Generation pipelines. Upload your documents, provide a labelled
        Q&A dataset, and compare multiple retrieval strategies in one click.

        ---

        ### 🏗️ Architecture

        | Layer | Technology |
        |---|---|
        | **UI** | Streamlit |
        | **API** | FastAPI (REST) |
        | **Dense retrieval** | FAISS + Sentence Transformers |
        | **Sparse retrieval** | BM25 (rank-bm25) |
        | **Hybrid fusion** | Reciprocal Rank Fusion (RRF) |
        | **Re-ranking** | CrossEncoder (ms-marco-MiniLM) |
        | **Answer generation** | OpenAI-compatible LLM or extractive demo |
        | **LLM judge** | OpenAI-compatible API (optional) |
        | **Storage** | SQLite via SQLAlchemy |

        ---

        ### 📐 Metrics Explained

        | Metric | Description |
        |---|---|
        | **Hit@K** | Was any relevant passage in the top-K results? (0 or 1) |
        | **Recall@K** | Fraction of all relevant passages that appeared in top-K |
        | **MRR** | Mean Reciprocal Rank — rewards results near the top |
        | **NDCG@K** | Normalised Discounted Cumulative Gain — position-sensitive ranking quality |
        | **Precision@K** | Fraction of top-K results that are relevant |
        | **Answer F1** | Token overlap between generated answer and ground truth |
        | **Faithfulness** | LLM judge: is every claim in the answer supported by context? (1–5) |
        | **Answer Relevance** | LLM judge: does the answer address the question? (1–5) |
        | **Latency** | Measured separately: retrieval, re-rank, generation, and total |

        ---

        ### 🚀 Quick Start

        1. **Knowledge Base tab** → click **⚡ Load Sample Data** to try the built-in corpus
        2. **Run Experiment tab** → choose the dataset and click **▶️ Run Benchmark**
        3. **Results tab** → explore the leaderboard, radar chart, and per-question traces
        4. **Live Query tab** → ask any question directly

        ---

        ### 🔑 LLM Integration (optional)

        Edit `.env` to enable real LLM answers and AI-powered judging:

        ```env
        LLM_MODE=openai_compatible
        LLM_BASE_URL=https://api.groq.com/openai/v1   # or https://api.openai.com/v1
        LLM_API_KEY=your_api_key_here
        LLM_MODEL=openai/gpt-oss-20b                   # or gpt-4o-mini (check Groq's current model list)
        ```

        Restart both servers after editing `.env`.
        """
    )