"""Validate evaluation questions in data/eval/qa.csv (or specified path).

Checks performed:
1. Every source file exists in data/docs.
2. Every page number is a valid 1-based page within the document's page range.
3. Flag duplicate questions.
4. Show the count of questions per file and per page.
5. Report how many questions have two gold pages (e.g., page "5|6").
6. Flag "too easy" questions where >60% of non-stopword question tokens appear
   verbatim in the gold page text (so they can be rewritten as paraphrases).

Usage:
    python scripts/validate_qa.py
    python scripts/validate_qa.py --qa data/eval/qa.csv --docs data/docs
    python scripts/validate_qa.py --qa data/sample/qa.csv --docs data/sample
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studymate.evaluate import calculate_verbatim_overlap  # noqa: E402
from studymate.ingest import SUPPORTED_SUFFIXES, read_document_pages  # noqa: E402
from studymate.retrievers import STOPWORDS  # noqa: E402


def validate_qa_file(
    qa_path: Path, docs_dir: Path, easy_threshold: float = 0.60
) -> dict:
    results = {
        "exists": False,
        "total_rows": 0,
        "valid_rows": 0,
        "errors": [],
        "duplicates": [],
        "file_counts": Counter(),
        "page_counts": defaultdict(Counter),
        "two_gold_pages": [],
        "multi_gold_pages": [],
        "too_easy": [],
    }

    if not qa_path.is_file():
        results["errors"].append(f"QA file not found: {qa_path}")
        return results

    results["exists"] = True

    try:
        with open(qa_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [c.strip() for c in (reader.fieldnames or [])]
            required_cols = {"question", "source", "page"}
            if not required_cols.issubset(set(fieldnames)):
                results["errors"].append(
                    f"CSV missing required columns. Found: {fieldnames}, Required: {sorted(required_cols)}"
                )
                return results

            rows = list(reader)
    except Exception as error:
        results["errors"].append(f"Failed to read CSV: {error}")
        return results

    results["total_rows"] = len(rows)
    if not rows:
        results["errors"].append("QA file is empty (contains no question rows).")
        return results

    # Document cache: {source_filename: list[tuple[page_num, text]]}
    doc_cache: dict[str, list[tuple[int, str]]] = {}
    missing_docs: set[str] = set()

    def get_doc_pages(source_name: str) -> list[tuple[int, str]] | None:
        if source_name in doc_cache:
            return doc_cache[source_name]
        if source_name in missing_docs:
            return None
        candidate = docs_dir / source_name
        if not candidate.is_file():
            # Try case-insensitive search
            found = None
            if docs_dir.exists():
                for p in docs_dir.rglob("*"):
                    if p.is_file() and p.name.lower() == source_name.lower():
                        found = p
                        break
            if found:
                candidate = found
            else:
                missing_docs.add(source_name)
                return None
        try:
            pages = read_document_pages(candidate)
            doc_cache[source_name] = pages
            return pages
        except Exception as e:
            results["errors"].append(f"Could not read document '{source_name}': {e}")
            missing_docs.add(source_name)
            return None

    # Track duplicates
    seen_questions: dict[str, list[int]] = defaultdict(list)

    for row_idx, row in enumerate(rows, start=2):  # 1-based index with header at row 1
        q_raw = row.get("question", "")
        q_text = q_raw.strip()
        source_raw = row.get("source", "")
        source_name = source_raw.strip()
        page_raw = row.get("page", "")
        page_str = str(page_raw).strip()

        if not q_text:
            results["errors"].append(f"Row {row_idx}: Empty question.")
            continue

        seen_questions[q_text.lower()].append(row_idx)

        if not source_name:
            results["errors"].append(f"Row {row_idx}: Empty source file for question '{q_text}'.")
            continue

        pages = get_doc_pages(source_name)
        if pages is None:
            results["errors"].append(
                f"Row {row_idx}: Source file '{source_name}' does not exist in '{docs_dir}'."
            )
            continue

        total_pages = len(pages)
        if not page_str:
            results["errors"].append(
                f"Row {row_idx}: Empty page number for question '{q_text}'."
            )
            continue

        # Parse page numbers (e.g. "5" or "5|6")
        page_parts = [p.strip() for p in page_str.split("|") if p.strip()]
        valid_page_numbers: list[int] = []
        page_error = False

        for part in page_parts:
            try:
                p_num = int(part)
                if p_num < 1 or p_num > total_pages:
                    results["errors"].append(
                        f"Row {row_idx}: Page {p_num} out of range for '{source_name}' "
                        f"(valid range: 1-{total_pages}). Question: '{q_text}'"
                    )
                    page_error = True
                else:
                    valid_page_numbers.append(p_num)
            except ValueError:
                results["errors"].append(
                    f"Row {row_idx}: Invalid page value '{part}' for '{source_name}'. Question: '{q_text}'"
                )
                page_error = True

        if page_error or not valid_page_numbers:
            continue

        results["valid_rows"] += 1
        results["file_counts"][source_name] += 1
        for p_num in valid_page_numbers:
            results["page_counts"][source_name][p_num] += 1

        if len(valid_page_numbers) == 2:
            results["two_gold_pages"].append(
                {"row": row_idx, "question": q_text, "source": source_name, "pages": valid_page_numbers}
            )
        elif len(valid_page_numbers) > 2:
            results["multi_gold_pages"].append(
                {"row": row_idx, "question": q_text, "source": source_name, "pages": valid_page_numbers}
            )

        # Check verbatim overlap (>60% non-stopword words)
        gold_page_texts = [pages[p_num - 1][1] for p_num in valid_page_numbers]
        combined_gold_text = " ".join(gold_page_texts)

        ratio, matched, tokens = calculate_verbatim_overlap(q_text, combined_gold_text)
        if ratio > easy_threshold:
            results["too_easy"].append(
                {
                    "row": row_idx,
                    "question": q_text,
                    "source": source_name,
                    "pages": page_str,
                    "overlap_ratio": ratio,
                    "matched": matched,
                    "tokens": tokens,
                }
            )

    # Flag duplicate questions
    for q_norm, row_nums in seen_questions.items():
        if len(row_nums) > 1:
            matching_q = rows[row_nums[0] - 2]["question"]
            results["duplicates"].append({"question": matching_q, "rows": row_nums})

    return results


def print_report(results: dict, qa_path: Path, docs_dir: Path, easy_threshold: float = 0.60) -> None:
    sep = "=" * 80
    subsep = "-" * 80

    print(sep)
    print(f"StudyMate AI - QA Validation Report")
    print(f"QA File:   {qa_path}")
    print(f"Docs Dir:  {docs_dir}")
    print(sep)

    if not results["exists"]:
        print(f"ERROR: QA file does not exist: {qa_path}")
        print("\nTip: When you are ready, create data/eval/qa.csv with columns: question,source,page")
        print(sep)
        return

    print(f"\n[1] Overall Statistics")
    print(f"  Total question rows:      {results['total_rows']}")
    print(f"  Valid rows:               {results['valid_rows']}")
    print(f"  Duplicate questions:      {len(results['duplicates'])}")
    print(f"  Validation errors:        {len(results['errors'])}")
    print(f"  Questions with 2 gold pages: {len(results['two_gold_pages'])}")
    if results["multi_gold_pages"]:
        print(f"  Questions with >2 gold pages: {len(results['multi_gold_pages'])}")
    print(f"  'Too easy' questions (>{easy_threshold * 100:.0f}% overlap): {len(results['too_easy'])}")

    print(f"\n[2] Distribution by Source File and Page")
    if results["file_counts"]:
        for source, count in sorted(results["file_counts"].items()):
            print(f"  File: {source} ({count} questions)")
            page_dict = results["page_counts"][source]
            page_strs = [f"p.{p} ({cnt})" for p, cnt in sorted(page_dict.items())]
            print(f"    Pages: {', '.join(page_strs)}")
    else:
        print("  (no valid questions per file to display)")

    if results["duplicates"]:
        print(f"\n[3] Duplicate Questions Flagged ({len(results['duplicates'])})")
        for dup in results["duplicates"]:
            print(f"  - Rows {dup['rows']}: \"{dup['question']}\"")

    if results["two_gold_pages"]:
        print(f"\n[4] Questions with Two Gold Pages ({len(results['two_gold_pages'])})")
        for item in results["two_gold_pages"]:
            print(f"  - Row {item['row']} [{item['source']} pages {item['pages']}]: \"{item['question']}\"")

    if results["too_easy"]:
        print(
            f"\n[5] 'Too Easy' Questions Flagged ({len(results['too_easy'])}) "
            f"[>{easy_threshold * 100:.0f}% non-stopword verbatim overlap]"
        )
        print("  Consider rewriting these questions into paraphrases that do not copy exact page wording:")
        for item in results["too_easy"]:
            pct = item["overlap_ratio"] * 100
            matched_str = ", ".join(item["matched"])
            print(f"  - Row {item['row']} ({pct:.1f}% verbatim) [{item['source']} p.{item['pages']}]:")
            print(f"    Question: \"{item['question']}\"")
            print(f"    Matched non-stopwords: {matched_str}")

    if results["errors"]:
        print(f"\n[6] Validation Errors ({len(results['errors'])})")
        for err in results["errors"]:
            print(f"  - ERROR: {err}")

    print(sep)
    has_fatal_errors = bool(results["errors"] or results["duplicates"])
    if has_fatal_errors:
        print("STATUS: FAILED (resolve errors/duplicates before building split)")
    else:
        print("STATUS: PASSED")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qa", default="data/eval/qa.csv", help="question,source,page CSV")
    parser.add_argument("--docs", default="data/docs", help="folder with source documents")
    parser.add_argument(
        "--easy-threshold",
        type=float,
        default=0.60,
        help="Verbatim non-stopword ratio threshold to flag as too easy (default: 0.60)",
    )
    args = parser.parse_args()

    qa_path = (ROOT / args.qa).resolve() if not Path(args.qa).is_absolute() else Path(args.qa)
    docs_dir = (ROOT / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)

    results = validate_qa_file(qa_path, docs_dir, easy_threshold=args.easy_threshold)
    print_report(results, qa_path, docs_dir, easy_threshold=args.easy_threshold)

    has_fatal_errors = bool(results["errors"] or results["duplicates"] or not results["exists"])
    if has_fatal_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

