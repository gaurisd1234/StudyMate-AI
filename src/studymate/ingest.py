"""Step 1 of the pipeline: turn PDFs / text files into small, searchable chunks.

Why chunks? A search system cannot return "page 5 of a 300-page PDF" and expect
the language model to read all of it. We cut each page into pieces of roughly
150 words. Small pieces are easier to match to a question and cheaper to feed
to an LLM later.

Every chunk remembers WHERE it came from (file name + page number). That is what
lets us (a) cite sources in answers and (b) measure retrieval quality honestly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str  # file name, e.g. "ml_notes.pdf"
    page: int  # 1-based page number (always 1 for .txt / .md files)
    text: str


def clean_text(text: str) -> str:
    """Light cleanup. PDFs often contain line-break hyphens and messy spacing."""
    text = text.replace("­", "")  # invisible soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "classi-\nfication" -> "classification"
    text = re.sub(r"[\ud800-\udfff]", "", text)  # remove unmatched surrogate code points from PDF math fonts
    text = re.sub(r"\s+", " ", text)  # collapse all whitespace/newlines
    return text.strip()


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Scanned PDFs (images only) give empty text."""
    from pypdf import PdfReader  # imported here so other modules work without it

    reader = PdfReader(str(path))
    return [
        (number, clean_text(page.extract_text() or ""))
        for number, page in enumerate(reader.pages, start=1)
    ]


def read_text_file(path: Path) -> list[tuple[int, str]]:
    return [(1, clean_text(path.read_text(encoding="utf-8", errors="ignore")))]


def read_document_pages(path: str | Path) -> list[tuple[int, str]]:
    """Read any supported document (.pdf, .txt, .md) and return [(page_number, text), ...]."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_pages(p)
    elif suffix in {".txt", ".md"}:
        return read_text_file(p)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {SUPPORTED_SUFFIXES}")


def get_page_text(path: str | Path, page_number: int) -> str:
    """Return the cleaned text of a single 1-based page from a document."""
    pages = read_document_pages(path)
    if not (1 <= page_number <= len(pages)):
        raise IndexError(
            f"Page {page_number} out of range for {Path(path).name} (total pages: {len(pages)})"
        )
    return pages[page_number - 1][1]


def chunk_words(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    """Split text into windows of `chunk_size` words that share `overlap` words.

    The overlap matters: if an answer sits on the boundary between two chunks,
    the overlapping words make sure at least one chunk contains the whole idea.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):  # this window already reached the end
            break
    return chunks


def build_chunks(
    docs_dir: str | Path, chunk_size: int = 150, overlap: int = 30
) -> tuple[list[Chunk], list[str]]:
    """Read every supported file under `docs_dir` and return (chunks, warnings).

    Chunks never cross page boundaries, so each chunk has one exact page number.
    """
    docs_dir = Path(docs_dir)
    files = sorted(
        p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    chunks: list[Chunk] = []
    warnings: list[str] = []
    if not files:
        warnings.append(f"No .pdf/.txt/.md files found in {docs_dir}")

    for path in files:
        pages = (
            read_pdf_pages(path) if path.suffix.lower() == ".pdf" else read_text_file(path)
        )
        empty_pages = [number for number, text in pages if not text]
        if empty_pages:
            warnings.append(
                f"{path.name}: {len(empty_pages)} page(s) have no extractable text "
                f"(scanned images?), e.g. page {empty_pages[0]}. They were skipped."
            )
        for number, text in pages:
            for index, piece in enumerate(chunk_words(text, chunk_size, overlap)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{path.stem}_p{number}_c{index}",
                        source=path.name,
                        page=number,
                        text=piece,
                    )
                )
    return chunks, warnings
