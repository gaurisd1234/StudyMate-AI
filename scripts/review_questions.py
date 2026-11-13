"""Interactive CLI for human review of synthetic evaluation questions.

Presents random questions (default 30) along with the gold page text.
Decisions:
  y: Good question (accept)
  n: Bad, ambiguous, or unanswerable (reject)
  q: Quit and save current progress

Outputs:
  data/eval/review.csv: Record of reviewed questions with decisions.
  data/eval/qa.csv: Final evaluation set with all questions EXCEPT rejected ones.

Usage:
    python scripts/review_questions.py
    python scripts/review_questions.py --count 30 --seed 42
    python scripts/review_questions.py --accept-all
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studymate.ingest import get_page_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--synthetic-qa", default="data/eval/qa_synthetic.csv", help="synthetic QA file to review")
    parser.add_argument("--docs", default="data/docs", help="folder with source documents")
    parser.add_argument("--review-out", default="data/eval/review.csv", help="output file for review decisions")
    parser.add_argument("--qa-out", default="data/eval/qa.csv", help="output file for final accepted QA set")
    parser.add_argument("--count", type=int, default=30, help="number of questions to review (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for question selection")
    parser.add_argument(
        "--accept-all",
        action="store_true",
        help="non-interactive mode: accept all reviewed questions (useful for tests and automation)",
    )
    parser.add_argument(
        "--auto-decision",
        choices=["y", "n"],
        default=None,
        help="non-interactive mode: automatically provide decision 'y' or 'n'",
    )
    args = parser.parse_args()

    qa_synth_path = (ROOT / args.synthetic_qa).resolve() if not Path(args.synthetic_qa).is_absolute() else Path(args.synthetic_qa)
    docs_dir = (ROOT / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)
    review_out_path = (ROOT / args.review_out).resolve() if not Path(args.review_out).is_absolute() else Path(args.review_out)
    qa_out_path = (ROOT / args.qa_out).resolve() if not Path(args.qa_out).is_absolute() else Path(args.qa_out)

    if not qa_synth_path.is_file():
        sys.exit(f"Error: Synthetic QA file '{qa_synth_path}' not found.\nRun scripts/make_eval_set.py first.")

    with open(qa_synth_path, newline="", encoding="utf-8-sig") as f:
        all_questions = list(csv.DictReader(f))

    if not all_questions:
        sys.exit(f"Error: '{qa_synth_path}' contains no questions.")

    # Load any existing review progress
    existing_reviews: dict[str, str] = {}
    if review_out_path.is_file():
        with open(review_out_path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                q_text = r.get("question", "").strip()
                if q_text:
                    existing_reviews[q_text] = r.get("decision", "").strip().lower()

    # Sample random questions to review
    rng = random.Random(args.seed)
    num_to_review = min(args.count, len(all_questions))
    indices = list(range(len(all_questions)))
    rng.shuffle(indices)
    selected_indices = indices[:num_to_review]
    review_subset = [all_questions[i] for i in selected_indices]

    print("=" * 80)
    print(f"StudyMate AI - Question Review CLI")
    print(f"Reviewing {num_to_review} random questions from {len(all_questions)} total (seed={args.seed})")
    print("Commands: [y] Good (accept) | [n] Bad/ambiguous (reject) | [q] Save & Quit")
    print("=" * 80)

    # Cache document pages in memory
    doc_pages_cache: dict[str, list[tuple[int, str]]] = {}

    def fetch_page_text(source: str, page_num: int) -> str:
        if source not in doc_pages_cache:
            doc_path = docs_dir / source
            if not doc_path.is_file():
                for p in docs_dir.rglob("*"):
                    if p.is_file() and p.name.lower() == source.lower():
                        doc_path = p
                        break
            try:
                from studymate.ingest import read_document_pages
                doc_pages_cache[source] = read_document_pages(doc_path)
            except Exception as e:
                doc_pages_cache[source] = []

        pages = doc_pages_cache.get(source, [])
        if 1 <= page_num <= len(pages):
            return pages[page_num - 1][1]
        return f"[Page {page_num} not found or out of range for {source}]"

    new_reviews: list[dict] = []
    rejected_questions: set[str] = set()

    for idx, item in enumerate(review_subset, start=1):
        q = item["question"]
        source = item["source"]
        page = int(item["page"])
        q_type = item.get("type", "unknown")
        chunk_id = item.get("chunk_id", "")

        # Check if already reviewed
        prev_decision = existing_reviews.get(q)
        if prev_decision:
            new_reviews.append({**item, "decision": prev_decision})
            if prev_decision == "n":
                rejected_questions.add(q)
            continue

        page_text = fetch_page_text(source, page)
        preview_text = page_text[:400] + ("..." if len(page_text) > 400 else "")

        print(f"\n[{idx}/{num_to_review}] Source: {source} | Page: {page} | Type: {q_type.upper()}")
        print("-" * 80)
        print(f"QUESTION: {q}")
        print("-" * 80)
        print(f"GOLD PAGE TEXT PREVIEW:\n{preview_text}")
        print("-" * 80)

        decision = None
        if args.accept_all:
            decision = "y"
            print("Auto-decision: y (accept-all)")
        elif args.auto_decision:
            decision = args.auto_decision
            print(f"Auto-decision: {decision}")
        else:
            while decision not in {"y", "n", "q"}:
                try:
                    choice = input("Decision [y/n/q]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "q"
                if choice in {"y", "n", "q"}:
                    decision = choice
                else:
                    print("Please enter 'y' to accept, 'n' to reject, or 'q' to quit.")

        if decision == "q":
            print("\nQuitting review. Saving current progress...")
            break

        new_reviews.append({**item, "decision": decision})
        existing_reviews[q] = decision
        if decision == "n":
            rejected_questions.add(q)

    # Combine existing and new reviews
    review_out_path.parent.mkdir(parents=True, exist_ok=True)
    all_reviewed_map = {r["question"]: r for r in new_reviews}
    for q_text, dec in existing_reviews.items():
        if q_text not in all_reviewed_map:
            for item in all_questions:
                if item["question"] == q_text:
                    all_reviewed_map[q_text] = {**item, "decision": dec}
                    break

    review_list = list(all_reviewed_map.values())
    if review_list:
        with open(review_out_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["question", "source", "page", "chunk_id", "type", "decision"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_list)

    # Calculate acceptance rate
    total_reviewed = len(review_list)
    accepted_count = sum(1 for r in review_list if r["decision"] == "y")
    rejected_count = sum(1 for r in review_list if r["decision"] == "n")
    acceptance_rate = (accepted_count / total_reviewed * 100) if total_reviewed > 0 else 0.0

    print("\n" + "=" * 80)
    print("StudyMate AI - Review Summary")
    print("=" * 80)
    print(f"Total Reviewed Questions: {total_reviewed}")
    print(f"Accepted [y]:             {accepted_count}")
    print(f"Rejected [n]:             {rejected_count}")
    print(f"Acceptance Rate:          {acceptance_rate:.1f}%")
    print(f"Review Log Saved:         {review_out_path}")

    # Build final qa.csv containing all questions except rejected
    final_qa_rows: list[dict[str, str]] = []
    for item in all_questions:
        if existing_reviews.get(item["question"]) == "n":
            continue
        final_qa_rows.append(
            {
                "question": item["question"],
                "source": item["source"],
                "page": item["page"],
            }
        )

    qa_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qa_out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question", "source", "page"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_qa_rows)

    print(f"Final QA Dataset Saved:   {qa_out_path} ({len(final_qa_rows)} questions retained)")
    print("=" * 80)


if __name__ == "__main__":
    main()
