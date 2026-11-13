"""Evaluate BM25, Dense, Hybrid (RRF and Weighted), and Cross-Encoder Reranking.

Rules:
- All hyperparameter tuning (e.g. weighted score fusion w from 0.2 to 0.8) is conducted
  on the DEV split ONLY.
- The TEST split is evaluated ONCE on the chosen final configurations.
- Reports Recall@1, Recall@3, Recall@5, Recall@10, MRR, and latency_ms.

Usage:
    python scripts/run_retrieval_experiments.py
    python scripts/run_retrieval_experiments.py --docs data/docs --qa data/eval/qa.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studymate.evaluate import evaluate, filter_qa_by_split, format_table, load_qa, save_details  # noqa: E402
from studymate.ingest import build_chunks  # noqa: E402
from studymate.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    SentenceTransformerEmbedder,
)


def log_comparison(out_dir: Path, title: str, results: dict[str, dict]) -> None:
    timestamp = datetime.datetime.now().astimezone().isoformat()
    lines = [
        "=" * 80,
        f"Timestamp: {timestamp}",
        f"Experiment: {title}",
        "Results Table:",
    ]
    for method, metrics in results.items():
        m_strs = [
            f"{k}={metrics[k]:.1f}" if k == "latency_ms" else f"{k}={metrics[k]:.4f}"
            for k in metrics
        ]
        lines.append(f"  {method:<35} : {', '.join(m_strs)}")
    lines.append("=" * 80)
    lines.append("")

    entry = "\n".join(lines)
    for log_name in ("runs.log", "run_log.txt"):
        with open(out_dir / log_name, "a", encoding="utf-8") as f:
            f.write(entry)


def print_comparison_table(results: dict[str, dict], title: str) -> None:
    print("\n" + "=" * 80)
    print(f"RETRIEVAL EVALUATION RESULTS: {title}")
    print("=" * 80)
    header = f"{'Method':<32} {'Recall@1':>9} {'Recall@3':>9} {'Recall@5':>9} {'Recall@10':>10} {'MRR':>8} {'Lat(ms)':>9}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<32} {m['Recall@1']:>9.4f} {m['Recall@3']:>9.4f} {m['Recall@5']:>9.4f} "
            f"{m['Recall@10']:>10.4f} {m['MRR']:>8.4f} {m['latency_ms']:>9.1f}"
        )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--docs", default="data/docs", help="folder with source documents")
    parser.add_argument("--qa", default="data/eval/qa.csv", help="question,source,page CSV")
    parser.add_argument("--split-file", default="data/eval/split.json", help="path to split.json")
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="embedding model name",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="cross-encoder model name",
    )
    parser.add_argument("--out", default="results", help="where to save CSV results")
    args = parser.parse_args()

    docs_dir = (ROOT / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)
    qa_path = (ROOT / args.qa).resolve() if not Path(args.qa).is_absolute() else Path(args.qa)
    split_path = (ROOT / args.split_file).resolve() if not Path(args.split_file).is_absolute() else Path(args.split_file)
    out_dir = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    if out_dir.suffix.lower() == ".csv":
        out_dir = out_dir.parent

    if not qa_path.is_file():
        sys.exit(f"QA file not found: {qa_path}")
    if not split_path.is_file():
        sys.exit(f"Split file not found: {split_path}")

    print(f"Building chunks from {docs_dir}...")
    chunks, warnings = build_chunks(docs_dir, chunk_size=150, overlap=30)
    for w in warnings:
        print("  WARNING:", w)

    qa_all = load_qa(qa_path)
    qa_dev = filter_qa_by_split(qa_all, split_path, "dev")
    qa_test = filter_qa_by_split(qa_all, split_path, "test")

    print(f"\nCorpus: {len(chunks)} chunks from {len({c.source for c in chunks})} document(s)")
    print(f"QA:     {len(qa_dev)} DEV questions | {len(qa_test)} TEST questions\n")

    # 1. Base Retrievers
    print("Initializing BM25 retriever...")
    bm25 = BM25Retriever(chunks)

    print(f"Loading embedding model '{args.dense_model}'...")
    embedder = SentenceTransformerEmbedder(args.dense_model)
    dense = DenseRetriever(chunks, embedder, cache_dir=ROOT / ".cache")

    # =========================================================================
    # STEP 1: HYBRID TUNING ON DEV SPLIT ONLY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: EVALUATION AND HYBRID WEIGHT TUNING ON DEV SPLIT")
    print("=" * 80)

    # 1a. BM25 on DEV
    bm25_dev_metrics, _ = evaluate(bm25, qa_dev)
    print(f"BM25 (DEV):          MRR={bm25_dev_metrics['MRR']:.4f} | Recall@5={bm25_dev_metrics['Recall@5']:.4f}")

    # 1b. Dense on DEV
    dense_dev_metrics, _ = evaluate(dense, qa_dev)
    print(f"Dense (DEV):         MRR={dense_dev_metrics['MRR']:.4f} | Recall@5={dense_dev_metrics['Recall@5']:.4f}")

    # 1c. Hybrid-RRF (k=60) on DEV
    hybrid_rrf = HybridRetriever(bm25, dense, method="rrf", rrf_k=60)
    rrf_dev_metrics, _ = evaluate(hybrid_rrf, qa_dev)
    print(f"Hybrid-RRF (DEV):    MRR={rrf_dev_metrics['MRR']:.4f} | Recall@5={rrf_dev_metrics['Recall@5']:.4f}")

    # 1d. Tuning weighted score fusion on DEV: w in [0.2, 0.3, ..., 0.8]
    print("\nTuning Weighted Min-Max Fusion (weights 0.2 to 0.8 in steps of 0.1 on DEV)...")
    weights = [round(w * 0.1, 1) for w in range(2, 9)]
    weighted_sweep_results: list[tuple[float, dict]] = []

    for w in weights:
        h_weighted = HybridRetriever(bm25, dense, method="weighted", dense_weight=w)
        metrics, _ = evaluate(h_weighted, qa_dev)
        weighted_sweep_results.append((w, metrics))
        print(f"  w_dense={w:.1f} (w_bm25={1.0-w:.1f}): MRR={metrics['MRR']:.4f} | Recall@1={metrics['Recall@1']:.4f} | Recall@5={metrics['Recall@5']:.4f}")

    # Select best weight based on MRR, breaking ties with Recall@5
    best_weight, best_weighted_dev_metrics = max(
        weighted_sweep_results,
        key=lambda item: (item[1]["MRR"], item[1]["Recall@5"]),
    )
    print(f"\nChosen Best Weight on DEV: w_dense = {best_weight:.1f} (MRR = {best_weighted_dev_metrics['MRR']:.4f})")

    hybrid_weighted_best = HybridRetriever(bm25, dense, method="weighted", dense_weight=best_weight)

    # Determine best hybrid between RRF and Weighted
    if rrf_dev_metrics["MRR"] >= best_weighted_dev_metrics["MRR"]:
        best_hybrid_name = "hybrid-rrf"
        best_hybrid_dev = hybrid_rrf
        print(f"Best Hybrid Retriever on DEV: hybrid-rrf (MRR: {rrf_dev_metrics['MRR']:.4f})")
    else:
        best_hybrid_name = f"hybrid-weighted[w={best_weight:.1f}]"
        best_hybrid_dev = hybrid_weighted_best
        print(f"Best Hybrid Retriever on DEV: {best_hybrid_name} (MRR: {best_weighted_dev_metrics['MRR']:.4f})")

    # =========================================================================
    # STEP 2: RERANKING ON TOP 20 HYBRID CANDIDATES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: CROSS-ENCODER RERANKING OF TOP 20 HYBRID CANDIDATES")
    print("=" * 80)
    print(f"Loading Cross-Encoder '{args.reranker_model}'...")
    reranker = RerankRetriever(
        base_retriever=best_hybrid_dev,
        model_name=args.reranker_model,
        top_k_candidates=20,
    )

    rerank_dev_metrics, _ = evaluate(reranker, qa_dev)
    print(f"Reranker on DEV:     MRR={rerank_dev_metrics['MRR']:.4f} | Recall@1={rerank_dev_metrics['Recall@1']:.4f} | Recall@5={rerank_dev_metrics['Recall@5']:.4f}")

    # Summarize all DEV results
    dev_summary_results = {
        "bm25": bm25_dev_metrics,
        "dense[all-MiniLM-L6-v2]": dense_dev_metrics,
        "hybrid-rrf": rrf_dev_metrics,
        f"hybrid-weighted[w={best_weight:.1f}]": best_weighted_dev_metrics,
        f"rerank[{args.reranker_model.split('/')[-1]}]": rerank_dev_metrics,
    }
    print_comparison_table(dev_summary_results, "DEV SPLIT")
    log_comparison(out_dir, "DEV Split Evaluation", dev_summary_results)

    # =========================================================================
    # STEP 3: RUN CHOSEN CONFIGURATIONS ON TEST SPLIT (ONCE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATING FINAL CHOSEN CONFIGURATIONS ON TEST SPLIT (ONCE)")
    print("=" * 80)

    bm25_test_metrics, _ = evaluate(bm25, qa_test)
    dense_test_metrics, _ = evaluate(dense, qa_test)
    rrf_test_metrics, _ = evaluate(hybrid_rrf, qa_test)
    weighted_test_metrics, _ = evaluate(hybrid_weighted_best, qa_test)
    rerank_test_metrics, _ = evaluate(reranker, qa_test)

    test_summary_results = {
        "bm25": bm25_test_metrics,
        "dense[all-MiniLM-L6-v2]": dense_test_metrics,
        "hybrid-rrf": rrf_test_metrics,
        f"hybrid-weighted[w={best_weight:.1f}]": weighted_test_metrics,
        f"rerank[{args.reranker_model.split('/')[-1]}]": rerank_test_metrics,
    }
    print_comparison_table(test_summary_results, "TEST SPLIT")
    log_comparison(out_dir, "TEST Split Evaluation", test_summary_results)

    # Save to CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "retrieval_comparison.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "method", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "latency_ms"])
        for m_name, m in dev_summary_results.items():
            writer.writerow(["dev", m_name, m["Recall@1"], m["Recall@3"], m["Recall@5"], m["Recall@10"], m["MRR"], m["latency_ms"]])
        for m_name, m in test_summary_results.items():
            writer.writerow(["test", m_name, m["Recall@1"], m["Recall@3"], m["Recall@5"], m["Recall@10"], m["MRR"], m["latency_ms"]])

    print(f"\nSaved retrieval comparison table to {summary_csv}")
    print(f"Logged full results to {out_dir / 'runs.log'}")


if __name__ == "__main__":
    main()

