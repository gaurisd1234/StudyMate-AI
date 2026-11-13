"""Build chunks, run the BM25 and dense baselines, and print a comparison table.

Examples
--------
Try it on the sample data (after running make_sample_data.py):
    python scripts/run_baselines.py --docs data/sample --qa data/sample/qa.csv

Your own material:
    python scripts/run_baselines.py --docs data/docs --qa data/eval/qa.csv

With dev or test split:
    python scripts/run_baselines.py --docs data/docs --qa data/eval/qa.csv --split dev

Skip the dense model (BM25 only, no downloads):
    python scripts/run_baselines.py --docs data/sample --qa data/sample/qa.csv --dense-model none
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
    save_details,
)
from studymate.ingest import build_chunks  # noqa: E402
from studymate.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)


def log_run_config(
    out_dir: Path,
    args: argparse.Namespace,
    num_chunks: int,
    num_files: int,
    num_questions: int,
    results: dict[str, dict],
    split_file_used: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().astimezone().isoformat()
    lines = [
        "=" * 80,
        f"Timestamp: {timestamp}",
        "Configuration:",
        f"  docs:         {args.docs}",
        f"  qa:           {args.qa}",
        f"  split:        {args.split}",
        f"  split_file:   {split_file_used or 'N/A'}",
        f"  chunk_size:   {args.chunk_size}",
        f"  overlap:      {args.overlap}",
        f"  dense_model:  {args.dense_model}",
        f"  chunks:       {num_chunks} from {num_files} file(s)",
        f"  questions:    {num_questions}",
        "Results:",
    ]
    for method_name, metrics in results.items():
        metric_str = ", ".join(
            f"{k}={metrics[k]:.1f}" if k == "latency_ms" else f"{k}={metrics[k]:.4f}"
            for k in metrics
        )
        lines.append(f"  {method_name}: {metric_str}")
    lines.append("=" * 80)
    lines.append("")

    entry = "\n".join(lines)
    for log_name in ("runs.log", "run_log.txt"):
        with open(out_dir / log_name, "a", encoding="utf-8") as f:
            f.write(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", default="data/docs", help="folder with PDFs / txt / md")
    parser.add_argument("--qa", default="data/eval/qa.csv", help="question,source,page CSV")
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="all",
        help="which evaluation split to use: 'dev', 'test', or 'all' (default: all)",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="path to split.json (defaults to <qa_parent>/split.json or data/eval/split.json)",
    )
    parser.add_argument("--chunk-size", type=int, default=150, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=30, help="words shared between chunks")
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="embedding model name, 'none' to skip, or 'hashing' for the NON-semantic test embedder",
    )
    parser.add_argument("--out", default="results", help="where to save CSV results")
    args = parser.parse_args()

    qa_path = (ROOT / args.qa).resolve() if not Path(args.qa).is_absolute() else Path(args.qa)
    if not qa_path.is_file():
        sys.exit(f"QA file not found: {qa_path}")

    chunks, warnings = build_chunks(args.docs, args.chunk_size, args.overlap)
    for w in warnings:
        print("WARNING:", w)
    if not chunks:
        sys.exit("No chunks were created. Put PDFs in the docs folder first.")

    qa_rows = load_qa(qa_path)
    total_raw_qa = len(qa_rows)
    split_file_used = None

    if args.split != "all":
        if args.split_file:
            cand_split = Path(args.split_file)
        else:
            cand_split = qa_path.parent / "split.json"
            if not cand_split.is_file():
                cand_split = ROOT / "data/eval/split.json"

        if not cand_split.is_file():
            sys.exit(
                f"Error: Split file '{cand_split}' not found.\n"
                f"Run scripts/make_split.py first or pass --split all."
            )
        split_file_used = str(cand_split)
        qa_rows = filter_qa_by_split(qa_rows, cand_split, args.split)

    num_files = len({c.source for c in chunks})
    print(f"{len(chunks)} chunks from {num_files} file(s); "
          f"{len(qa_rows)} evaluation questions (split='{args.split}' from {total_raw_qa} total)\n")

    if not qa_rows:
        sys.exit(f"No questions found for split '{args.split}'.")

    retrievers = [BM25Retriever(chunks)]
    if args.dense_model.lower() == "hashing":
        print("NOTE: 'hashing' is a non-semantic test embedder; do not report these numbers.\n")
        retrievers.append(DenseRetriever(chunks, HashingEmbedder(), cache_dir=ROOT / ".cache"))
    elif args.dense_model.lower() != "none":
        embedder = SentenceTransformerEmbedder(args.dense_model)
        retrievers.append(DenseRetriever(chunks, embedder, cache_dir=ROOT / ".cache"))

    results = {}
    out_dir = Path(args.out)
    for retriever in retrievers:
        metrics, details = evaluate(retriever, qa_rows)
        results[retriever.name] = metrics
        safe = retriever.name.replace("[", "_").replace("]", "").replace("/", "_")
        save_details(details, out_dir / f"per_question_{safe}.csv")

    print(format_table(results))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baselines.csv", "w", newline="", encoding="utf-8") as f:
        columns = list(next(iter(results.values())).keys())
        writer = csv.writer(f)
        writer.writerow(["method", "chunk_size", "overlap", *columns])
        for name, metrics in results.items():
            writer.writerow([name, args.chunk_size, args.overlap, *[round(metrics[c], 4) for c in columns]])

    log_run_config(
        out_dir=out_dir,
        args=args,
        num_chunks=len(chunks),
        num_files=num_files,
        num_questions=len(qa_rows),
        results=results,
        split_file_used=split_file_used,
    )

    print(f"\nSaved {out_dir / 'baselines.csv'} and per-question files.")
    print(f"Logged configuration and results to {out_dir / 'runs.log'}.")


if __name__ == "__main__":
    main()
