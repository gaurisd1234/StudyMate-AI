"""Create a fixed-seed stratified dev/test split (40% dev, 60% test by source file).

Requires at least 60 questions before splitting.
Refuses to overwrite an existing split unless --force is specified.

Usage:
    python scripts/make_split.py
    python scripts/make_split.py --qa data/eval/qa.csv --out data/eval/split.json
    python scripts/make_split.py --force
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studymate.evaluate import load_qa  # noqa: E402


def create_stratified_split(
    qa_rows: list[dict],
    seed: int = 42,
    dev_ratio: float = 0.40,
) -> dict:
    rng = random.Random(seed)

    # Group row indices by source file
    source_groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(qa_rows):
        source_name = row.get("source", "").strip()
        source_groups[source_name].append(idx)

    dev_indices: list[int] = []
    test_indices: list[int] = []
    strat_summary: dict[str, dict[str, int]] = {}

    for source in sorted(source_groups.keys()):
        indices = list(source_groups[source])
        indices.sort()  # deterministic base order before shuffle
        rng.shuffle(indices)

        n = len(indices)
        n_dev = int(round(n * dev_ratio))
        # Ensure that if n >= 2 and n_dev == 0, we still allocate at least 1 if possible
        if n >= 2 and n_dev == 0 and dev_ratio > 0:
            n_dev = 1
        # And ensure test also gets at least 1 if n >= 2
        if n >= 2 and n_dev == n and dev_ratio < 1.0:
            n_dev = n - 1

        src_dev = indices[:n_dev]
        src_test = indices[n_dev:]

        dev_indices.extend(src_dev)
        test_indices.extend(src_test)

        strat_summary[source] = {
            "total": n,
            "dev": len(src_dev),
            "test": len(src_test),
        }

    dev_indices.sort()
    test_indices.sort()

    return {
        "seed": seed,
        "dev_ratio": dev_ratio,
        "test_ratio": round(1.0 - dev_ratio, 4),
        "total_questions": len(qa_rows),
        "dev_count": len(dev_indices),
        "test_count": len(test_indices),
        "dev": dev_indices,
        "test": test_indices,
        "dev_indices": dev_indices,
        "test_indices": test_indices,
        "dev_questions": [qa_rows[i]["question"] for i in dev_indices],
        "test_questions": [qa_rows[i]["question"] for i in test_indices],
        "stratification": strat_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qa", default="data/eval/qa.csv", help="question,source,page CSV")
    parser.add_argument("--out", default="data/eval/split.json", help="path to write split.json")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible split")
    parser.add_argument("--dev-ratio", type=float, default=0.40, help="fraction for dev set (default: 0.40)")
    parser.add_argument(
        "--min-questions",
        type=int,
        default=60,
        help="minimum number of questions required to create a split (default: 60)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing split file if it exists")
    args = parser.parse_args()

    qa_path = (ROOT / args.qa).resolve() if not Path(args.qa).is_absolute() else Path(args.qa)
    out_path = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    if not qa_path.is_file():
        sys.exit(f"Error: QA file '{qa_path}' does not exist.")

    if out_path.exists() and not args.force:
        sys.exit(
            f"Error: Split file '{out_path}' already exists.\n"
            f"Pass --force if you want to overwrite it."
        )

    qa_rows = load_qa(qa_path)
    total_q = len(qa_rows)

    if total_q < args.min_questions:
        sys.exit(
            f"Error: Found only {total_q} questions in '{qa_path}'.\n"
            f"At least {args.min_questions} questions are required to create a split.\n"
            f"Write questions using scripts/browse_pages.py and data/eval/qa.csv first."
        )

    split_dict = create_stratified_split(
        qa_rows=qa_rows,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    sep = "=" * 80
    print(sep)
    print(f"Stratified Dev/Test Split Created: {out_path}")
    print(f"Seed: {args.seed} | Total Questions: {total_q}")
    print(f"Dev:  {split_dict['dev_count']} questions ({split_dict['dev_count'] / total_q * 100:.1f}%)")
    print(f"Test: {split_dict['test_count']} questions ({split_dict['test_count'] / total_q * 100:.1f}%)")
    print("-" * 80)
    print(f"{'Source File':<40} {'Total':>8} {'Dev':>8} {'Test':>8} {'Dev %':>10}")
    print("-" * 80)
    for source, counts in split_dict["stratification"].items():
        dev_pct = counts["dev"] / counts["total"] * 100 if counts["total"] else 0
        print(f"{source:<40} {counts['total']:>8} {counts['dev']:>8} {counts['test']:>8} {dev_pct:>9.1f}%")
    print(sep)


if __name__ == "__main__":
    main()

