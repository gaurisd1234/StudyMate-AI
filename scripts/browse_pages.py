"""CLI tool to inspect the text of any document and page.

Usage examples:
    python scripts/browse_pages.py data/docs/pdf1.pdf 2
    python scripts/browse_pages.py pdf1.pdf 2
    python scripts/browse_pages.py --file mit15_773_s24_lec02.pdf --page 10
    python scripts/browse_pages.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studymate.ingest import SUPPORTED_SUFFIXES, read_document_pages  # noqa: E402


def resolve_file(file_str: str, docs_dir: Path) -> Path | None:
    candidate = Path(file_str)
    if candidate.is_file():
        return candidate.resolve()
    candidate_in_docs = docs_dir / file_str
    if candidate_in_docs.is_file():
        return candidate_in_docs.resolve()
    # Check case-insensitive match or stem match in docs_dir
    if docs_dir.exists():
        for p in docs_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                if p.name.lower() == file_str.lower() or p.stem.lower() == file_str.lower():
                    return p.resolve()
    return None


def list_documents(docs_dir: Path) -> None:
    if not docs_dir.exists():
        print(f"Docs folder '{docs_dir}' does not exist.")
        return
    files = sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        print(f"No supported documents ({', '.join(SUPPORTED_SUFFIXES)}) found in '{docs_dir}'.")
        return
    print(f"Available documents in '{docs_dir}':")
    for f in files:
        try:
            pages = read_document_pages(f)
            non_empty = sum(1 for _, t in pages if t.strip())
            print(f"  - {f.name} ({len(pages)} pages, {non_empty} with text)")
        except Exception as e:
            print(f"  - {f.name} (error reading: {e})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file_pos", nargs="?", default=None, help="File name or path")
    parser.add_argument("page_pos", nargs="?", type=int, default=None, help="1-based page number")
    parser.add_argument("--file", "-f", default=None, help="File name or path")
    parser.add_argument("--page", "-p", type=int, default=None, help="1-based page number")
    parser.add_argument("--docs", "-d", default="data/docs", help="Docs directory (default: data/docs)")
    parser.add_argument("--list", "-l", action="store_true", help="List available files and page counts")
    args = parser.parse_args()

    docs_dir = (ROOT / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)

    if args.list:
        list_documents(docs_dir)
        return

    target_file = args.file or args.file_pos
    target_page = args.page if args.page is not None else args.page_pos

    if not target_file:
        list_documents(docs_dir)
        print("\nSpecify a file and page number. Example:")
        print("    python scripts/browse_pages.py <filename> <page_number>")
        sys.exit(1)

    file_path = resolve_file(target_file, docs_dir)
    if not file_path:
        print(f"Error: Could not find file '{target_file}'.")
        list_documents(docs_dir)
        sys.exit(1)

    try:
        pages = read_document_pages(file_path)
    except Exception as error:
        sys.exit(f"Error reading '{file_path.name}': {error}")

    total_pages = len(pages)
    if total_pages == 0:
        sys.exit(f"'{file_path.name}' has 0 pages.")

    if target_page is None:
        print(f"'{file_path.name}' has {total_pages} page(s). Valid pages: 1 to {total_pages}.")
        print(f"Example: python scripts/browse_pages.py {file_path.name} 1")
        sys.exit(1)

    if target_page < 1 or target_page > total_pages:
        sys.exit(
            f"Error: Page {target_page} is out of range for '{file_path.name}'. "
            f"Valid pages are 1 to {total_pages}."
        )

    _, page_text = pages[target_page - 1]
    border = "=" * 80
    print(border)
    print(f"File: {file_path.name} | Page: {target_page} of {total_pages}")
    print(border)
    if page_text.strip():
        print(page_text)
    else:
        print("[No extractable text found on this page (scanned image or empty page)]")
    print(border)


if __name__ == "__main__":
    main()

