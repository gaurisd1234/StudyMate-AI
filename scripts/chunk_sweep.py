"""Run chunk size and overlap grid search on the DEV split only.

Evaluates BM25 and the dense model across:
- Chunk sizes: 80, 150, 250, 400
- Overlaps: 0%, 20%, 40%

Outputs results to results/chunk_sweep_dev.csv and logs to results/runs.log.

Usage:
    python scripts/chunk_sweep.py
    python scripts/chunk_sweep.py --docs data/docs --qa data/eval/qa.csv
    python scripts/chunk_sweep.py --docs data/sample --qa data/sample/qa.csv --split-file data/sample/split.json --dense-model hashing
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studymate.evaluate import (  # noqa: E402
    evaluate,
    filter_qa_by_split,
    format_table,
    load_qa,
)
from studymate.ingest import build_chunks  # noqa: E402
from studymate.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)

CHUNK_SIZES = [80, 150, 250, 400]
OVERLAP_PCTS = [0, 20, 40]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--docs", default="data/docs", help="folder with PDFs / txt / md")
    parser.add_argument("--qa", default="data/eval/qa.csv", help="question,source,page CSV")
    parser.add_argument(
        "--split-file",
        default=None,
        help="path to split.json (defaults to <qa_parent>/split.json or data/eval/split.json)",
    )
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="embedding model name, 'none' to skip, or 'hashing' for the test embedder",
    )
    parser.add_argument("--out", default="results", help="where to save results")
    args = parser.parse_args()

    docs_dir = (ROOT / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)
    qa_path = (ROOT / args.qa).resolve() if not Path(args.qa).is_absolute() else Path(args.qa)
    out_dir = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    if not qa_path.is_file():
        sys.exit(f"Error: QA file '{qa_path}' does not exist.")

    if args.split_file:
        split_path = Path(args.split_file)
    else:
        split_path = qa_path.parent / "split.json"
        if not split_path.is_file():
            split_path = ROOT / "data/eval/split.json"

    if not split_path.is_file():
        sys.exit(
            f"Error: Split file '{split_path}' does not exist.\n"
            f"Create the split first using scripts/make_split.py before running the chunk sweep."
        )

    all_qa = load_qa(qa_path)
    try:
        dev_qa = filter_qa_by_split(all_qa, split_path, "dev")
    except Exception as error:
        sys.exit(f"Error loading DEV split: {error}")

    if not dev_qa:
        sys.exit(f"Error: DEV split in '{split_path}' has 0 questions.")

    print("=" * 80)
    print(f"StudyMate AI - Chunk Size & Overlap Sweep (DEV Split Only)")
    print(f"Docs:       {docs_dir}")
    print(f"QA:         {qa_path} ({len(dev_qa)} DEV questions from {len(all_qa)} total)")
    print(f"Split file: {split_path}")
    print(f"Grid:       Chunk sizes {CHUNK_SIZES} words x Overlaps {OVERLAP_PCTS}%")
    print(f"Dense:      {args.dense_model}")
    print("=" * 80 + "\n")

    # Pre-initialize embedder once
    embedder = None
    if args.dense_model.lower() == "hashing":
        embedder = HashingEmbedder()
    elif args.dense_model.lower() != "none":
        print(f"Loading embedding model '{args.dense_model}'...")
        embedder = SentenceTransformerEmbedder(args.dense_model)
        print("Model loaded.\n")

    results_table: list[dict] = []
    total_configs = len(CHUNK_SIZES) * len(OVERLAP_PCTS)
    config_num = 0

    metric_keys = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "latency_ms"]

    for chunk_size in CHUNK_SIZES:
        for overlap_pct in OVERLAP_PCTS:
            config_num += 1
            overlap_words = int(round(chunk_size * (overlap_pct / 100.0)))
            print(
                f"[{config_num}/{total_configs}] Chunk size: {chunk_size} words | "
                f"Overlap: {overlap_pct}% ({overlap_words} words)..."
            )

            chunks, warnings = build_chunks(docs_dir, chunk_size, overlap_words)
            if warnings and config_num == 1:
                for w in warnings:
                    print("  WARNING:", w)
            if not chunks:
                print("  Skipping: No chunks created.")
                continue

            # 1. Evaluate BM25
            bm25 = BM25Retriever(chunks)
            bm25_metrics, _ = evaluate(bm25, dev_qa)
            row_bm25 = {
                "method": "bm25",
                "chunk_size": chunk_size,
                "overlap_pct": f"{overlap_pct}%",
                "overlap": overlap_words,
                "chunks": len(chunks),
                **{k: round(bm25_metrics[k], 4) for k in metric_keys},
            }
            results_table.append(row_bm25)

            # 2. Evaluate Dense
            if embedder is not None:
                dense = DenseRetriever(chunks, embedder, cache_dir=ROOT / ".cache")
                dense_metrics, _ = evaluate(dense, dev_qa)
                row_dense = {
                    "method": dense.name,
                    "chunk_size": chunk_size,
                    "overlap_pct": f"{overlap_pct}%",
                    "overlap": overlap_words,
                    "chunks": len(chunks),
                    **{k: round(dense_metrics[k], 4) for k in metric_keys},
                }
                results_table.append(row_dense)

            print(
                f"  BM25:  Recall@1={bm25_metrics['Recall@1']:.3f} | "
                f"Recall@5={bm25_metrics['Recall@5']:.3f} | "
                f"MRR={bm25_metrics['MRR']:.3f}"
            )
            if embedder is not None:
                print(
                    f"  Dense: Recall@1={dense_metrics['Recall@1']:.3f} | "
                    f"Recall@5={dense_metrics['Recall@5']:.3f} | "
                    f"MRR={dense_metrics['MRR']:.3f}"
                )
            print()

    # Save to results/chunk_sweep_dev.csv
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv_path = out_dir / "chunk_sweep_dev.csv"

    if results_table:
        fieldnames = list(results_table[0].keys())
        with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_table)

    # Print summary table
    print("=" * 80)
    print("SWEEP COMPLETE - RESULTS ON DEV SET")
    print("=" * 80)
    header = f"{'Method':<22} {'Size':>6} {'Ovlp%':>6} {'Ovlp':>6} {'Chunks':>8} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7} {'Lat(ms)':>8}"
    print(header)
    print("-" * len(header))
    for r in results_table:
        print(
            f"{r['method']:<22} {r['chunk_size']:>6} {r['overlap_pct']:>6} {r['overlap']:>6} "
            f"{r['chunks']:>8} {r['Recall@1']:>7.3f} {r['Recall@3']:>7.3f} {r['Recall@5']:>7.3f} "
            f"{r['MRR']:>7.3f} {r['latency_ms']:>8.1f}"
        )
    print("=" * 80)
    print(f"Saved results table to: {sweep_csv_path}")

    # Log to results/runs.log
    timestamp = datetime.datetime.now().astimezone().isoformat()
    log_lines = [
        "=" * 80,
        f"Timestamp: {timestamp}",
        "Chunk Sweep Configuration (DEV split):",
        f"  docs:         {docs_dir}",
        f"  qa:           {qa_path}",
        f"  split_file:   {split_path}",
        f"  dev_questions:{len(dev_qa)}",
        f"  dense_model:  {args.dense_model}",
        f"  chunk_sizes:  {CHUNK_SIZES}",
        f"  overlaps:     {OVERLAP_PCTS}%",
        f"  output_file:  {sweep_csv_path}",
        "Summary:",
    ]
    # Log best BM25 and Dense by MRR
    bm25_runs = [r for r in results_table if r["method"] == "bm25"]
    if bm25_runs:
        best_bm25 = max(bm25_runs, key=lambda x: x["MRR"])
        log_lines.append(
            f"  Best BM25:  size={best_bm25['chunk_size']}, overlap={best_bm25['overlap_pct']}, "
            f"MRR={best_bm25['MRR']:.4f}, Recall@5={best_bm25['Recall@5']:.4f}"
        )
    dense_runs = [r for r in results_table if r["method"] != "bm25"]
    if dense_runs:
        best_dense = max(dense_runs, key=lambda x: x["MRR"])
        log_lines.append(
            f"  Best Dense: size={best_dense['chunk_size']}, overlap={best_dense['overlap_pct']}, "
            f"MRR={best_dense['MRR']:.4f}, Recall@5={best_dense['Recall@5']:.4f}"
        )
    log_lines.append("=" * 80)
    log_lines.append("")

    entry = "\n".join(log_lines)
    for log_name in ("runs.log", "run_log.txt"):
        with open(out_dir / log_name, "a", encoding="utf-8") as f:
            f.write(entry)
    print(f"Logged run summary to: {out_dir / 'runs.log'}")


if __name__ == "__main__":
    main()

