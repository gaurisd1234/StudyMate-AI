"""StudyMate AI - Interactive Streamlit Application.

Interactive course assistant providing grounded answers with verifiable page-level citations,
score-annotated chunk inspection, per-stage latency breakdown, and retrieval mode toggles.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from studymate.answer import ABSTENTION_TEXT, answer_question  # noqa: E402
from studymate.ingest import Chunk, build_chunks  # noqa: E402
from studymate.llm import LLMClient  # noqa: E402
from studymate.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    Retriever,
    SentenceTransformerEmbedder,
)

# Page configuration
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_corpus(doc_dir: str, chunk_size: int = 250, overlap_pct: int = 20) -> list[Chunk]:
    """Ingest documents from folder and cache chunk list."""
    path = Path(doc_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Document directory '{doc_dir}' does not exist.")
    overlap_words = int(chunk_size * (overlap_pct / 100.0))
    chunks, _ = build_chunks(path, chunk_size=chunk_size, overlap=overlap_words)
    return chunks


@st.cache_resource(show_spinner=False)
def get_bm25_retriever(_chunks: list[Chunk]) -> BM25Retriever:
    """Build or retrieve cached BM25 index."""
    return BM25Retriever(_chunks)


@st.cache_resource(show_spinner=False)
def get_dense_retriever(_chunks: list[Chunk], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> DenseRetriever:
    """Build or retrieve cached dense embedding index."""
    embedder = SentenceTransformerEmbedder(model_name=model_name)
    return DenseRetriever(_chunks, embedder=embedder)


@st.cache_resource(show_spinner=False)
def get_reranker_retriever(
    _base_retriever: Retriever,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 20,
) -> RerankRetriever:
    """Instantiate cached Cross-Encoder reranker."""
    return RerankRetriever(_base_retriever, model_name=model_name, top_k_candidates=top_n)


def select_retriever(
    mode: str,
    chunks: list[Chunk],
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Retriever:
    """Construct retriever corresponding to selected retrieval mode."""
    mode_lower = mode.strip().lower()

    bm25 = get_bm25_retriever(chunks)

    if mode_lower == "bm25":
        return bm25

    dense = get_dense_retriever(chunks, model_name=dense_model)
    if mode_lower == "dense":
        return dense

    # Hybrid Reciprocal Rank Fusion (k=60)
    hybrid = HybridRetriever(bm25, dense, method="rrf", rrf_k=60)
    if mode_lower in ("hybrid", "hybrid-rrf"):
        return hybrid

    if mode_lower in ("hybrid+rerank", "hybrid-rerank", "rerank"):
        return get_reranker_retriever(hybrid, top_n=20)

    # Default fallback
    return hybrid


def main() -> None:
    st.title("📚 StudyMate AI")
    st.caption("Verifiable, page-grounded question answering over course slides and study materials.")

    # Sidebar: Indexing and Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        default_doc_dir = "data/docs" if (ROOT / "data" / "docs").is_dir() else "data/sample"
        doc_dir_input = st.text_input(
            "Document Directory",
            value=default_doc_dir,
            help="Folder containing course PDFs, markdown, or plain text notes.",
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            chunk_size = st.number_input("Chunk Size (words)", min_value=50, max_value=600, value=250, step=25)
        with col_c2:
            overlap_pct = st.number_input("Overlap (%)", min_value=0, max_value=50, value=20, step=5)

        rebuild_btn = st.button("🔄 Rebuild / Reload Index", use_container_width=True)
        if rebuild_btn:
            st.cache_resource.clear()
            st.toast("Cache cleared! Reloading corpus index...", icon="🔄")

        st.divider()

        st.subheader("🔍 Retrieval Settings")
        retrieval_mode = st.radio(
            "Retrieval Mode",
            options=["hybrid+rerank", "hybrid", "BM25", "dense"],
            index=0,
            format_func=lambda x: {
                "hybrid+rerank": "Hybrid + Cross-Encoder Rerank (Best)",
                "hybrid": "Hybrid (BM25 + Dense RRF k=60)",
                "BM25": "BM25 (Lexical Only)",
                "dense": "Dense (all-MiniLM-L6-v2)",
            }.get(x, x),
            help="Choose the retrieval strategy for finding relevant context chunks.",
        )

        top_k = st.slider("Top Context Passages (k)", min_value=1, max_value=10, value=5)

        st.divider()
        st.subheader("ℹ️ System Status")
        api_key_set = bool(os.getenv("GEMINI_API_KEY"))
        if api_key_set:
            st.success("Gemini API: Connected", icon="🟢")
        else:
            st.info("Gemini API: Not Set (Running FakeLLM mock)", icon="🟡")

    # Document Indexing
    doc_path = ROOT / doc_dir_input
    if not doc_path.exists():
        st.error(f"Document directory `{doc_path}` does not exist. Please specify a valid folder.")
        return

    try:
        with st.spinner("Building / loading chunk index..."):
            chunks = load_corpus(str(doc_path), chunk_size=int(chunk_size), overlap_pct=int(overlap_pct))
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not chunks:
        st.warning(f"No readable documents found in `{doc_dir_input}`. Add PDF or text files.")
        return

    doc_files = sorted(list({c.source for c in chunks}))
    with st.sidebar:
        st.write(f"**Indexed Documents:** {len(doc_files)}")
        st.write(f"**Total Chunks:** {len(chunks)}")
        with st.expander("View Source Files"):
            for f in doc_files:
                st.write(f"- `{f}`")

    # Main Question Answering Section
    st.subheader("💬 Ask Your Study Material")

    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How does backpropagation update weights in a neural network?",
        key="query_input",
    )

    col_btn, col_clear = st.columns([1, 6])
    with col_btn:
        ask_pressed = st.button("🚀 Ask", type="primary", use_container_width=True)

    if ask_pressed and user_query.strip():
        query = user_query.strip()
        with st.spinner("Retrieving passages and generating answer..."):
            t_total_start = time.perf_counter()
            retriever = select_retriever(retrieval_mode, chunks)

            res = answer_question(query, retriever=retriever, k=int(top_k))
            total_elapsed_ms = (time.perf_counter() - t_total_start) * 1000

        # Display Answer Section
        st.markdown("### 📝 Answer")
        if res.is_abstention:
            st.warning(
                f"**System Abstention:**\n\n> *\"{res.answer}\"*\n\n"
                "The retrieved study material did not contain sufficient evidence to answer this question. "
                "The model refused to extrapolate to prevent hallucinations.",
                icon="⚠️",
            )
        else:
            st.markdown(res.answer)

        # Cited Sources Section
        st.markdown("#### 📖 Cited Sources")
        if res.citations:
            cols = st.columns(len(res.citations) if len(res.citations) <= 4 else 4)
            for idx, cit in enumerate(res.citations):
                col_idx = idx % 4
                cols[col_idx].info(f"📄 **{cit}**")
        else:
            if not res.is_abstention:
                st.write("*No explicit source citations in answer.*")
            else:
                st.write("*No citations (abstention).*")

        # Latency Metrics Section
        st.markdown("#### ⏱️ Latency Breakdown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Retrieval Latency", f"{res.retrieval_latency_ms:.1f} ms")
        m2.metric("Generation Latency", f"{res.generation_latency_ms:.1f} ms")
        m3.metric("Total Latency", f"{total_elapsed_ms:.1f} ms")

        # Retrieved Chunks with Scores Section
        st.markdown(f"#### 🔍 Retrieved Passages (Top {len(res.chunks)})")
        if not res.chunks:
            st.write("No passages retrieved.")
        else:
            for idx, chunk in enumerate(res.chunks, start=1):
                score_val = res.scores[idx - 1] if idx - 1 < len(res.scores) else 0.0
                expander_label = f"#{idx} | Score: {score_val:.4f} | {chunk.source} (Page {chunk.page})"
                with st.expander(expander_label, expanded=(idx == 1)):
                    st.caption(f"**Document:** `{chunk.source}` | **Page:** {chunk.page} | **Chunk ID:** `{chunk.chunk_id}`")
                    st.markdown(f"```text\n{chunk.text}\n```")


if __name__ == "__main__":
    main()
