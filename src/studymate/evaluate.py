"""Step 3: measure retrieval quality honestly.

How we decide whether a search result is "correct":
Each evaluation question is labelled with the file and page that contain its
answer (the "gold" page). A retrieved chunk counts as relevant if it comes from
that page. Labelling pages instead of chunk IDs means your labels stay valid even
when you later change chunk size or overlap.

Metrics
-------
Recall@k : fraction of questions where a relevant chunk appears in the top k
           results. (With one gold page per question this is the "hit rate".)
MRR      : Mean Reciprocal Rank. For each question take 1/rank of the first
           relevant result (1, 1/2, 1/3, ... or 0 if none in the top results),
           then average. It rewards putting the right answer FIRST.
Note: k counts chunks, not pages, so two chunks from the same page use two slots.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path


def load_qa(path: str | Path) -> list[dict]:
    """CSV columns: question, source, page   (page may be "5" or "5|6")."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("question", "").strip()]
    for row in rows:
        row["gold"] = {
            (row["source"].strip(), int(p)) for p in str(row["page"]).split("|") if p.strip()
        }
    return rows


def load_split(path: str | Path) -> dict:
    """Load split.json containing dev and test indices/questions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def filter_qa_by_split(
    qa_rows: list[dict], split_path: str | Path, split_name: str
) -> list[dict]:
    """Filter qa_rows by split name ('dev', 'test', or 'all')."""
    split_name = split_name.lower()
    if split_name == "all":
        return qa_rows
    split_data = load_split(split_path)
    if split_name not in split_data:
        raise ValueError(
            f"Split '{split_name}' not found in {split_path}. Available keys: {list(split_data.keys())}"
        )
    split_items = split_data[split_name]
    if not split_items:
        return []
    first_item = split_items[0]
    if isinstance(first_item, int):
        target_indices = set(split_items)
        return [row for i, row in enumerate(qa_rows) if i in target_indices]
    elif isinstance(first_item, str):
        target_questions = set(split_items)
        return [row for row in qa_rows if row["question"] in target_questions]
    elif isinstance(first_item, dict) and "question" in first_item:
        target_questions = {item["question"] for item in split_items}
        return [row for row in qa_rows if row["question"] in target_questions]
    else:
        raise ValueError(f"Unrecognized split format in {split_path} under '{split_name}'")


def calculate_verbatim_overlap(
    question: str, page_text: str
) -> tuple[float, list[str], list[str]]:
    """Compute what fraction of non-stopword tokens in the question appear verbatim in the page.

    Returns (overlap_ratio, matched_words, non_stopwords).
    """
    from .retrievers import STOPWORDS

    q_tokens = [w for w in re.findall(r"\w+", question.lower()) if w not in STOPWORDS]
    if not q_tokens:
        return 0.0, [], []
    page_words = set(re.findall(r"\w+", page_text.lower()))
    matches = [w for w in q_tokens if w in page_words]
    ratio = len(matches) / len(q_tokens)
    return ratio, matches, q_tokens


def evaluate(retriever, qa_rows: list[dict], ks=(1, 3, 5, 10)):
    """Return (metrics dict, per-question details list)."""
    max_k = max(ks)
    ranks: list[int | None] = []
    latencies: list[float] = []
    details: list[dict] = []

    for row in qa_rows:
        start = time.perf_counter()
        hits = retriever.search(row["question"], max_k)
        latencies.append((time.perf_counter() - start) * 1000)

        rank = None
        for position, (chunk, _score) in enumerate(hits, start=1):
            if (chunk.source, chunk.page) in row["gold"]:
                rank = position
                break
        ranks.append(rank)

        top_chunk = hits[0][0] if hits else None
        details.append(
            {
                "question": row["question"],
                "gold": "|".join(f"{s}:p{p}" for s, p in sorted(row["gold"])),
                "first_relevant_rank": rank if rank is not None else "",
                "top1": f"{top_chunk.source}:p{top_chunk.page}" if top_chunk else "",
                "top1_text": top_chunk.text[:160] if top_chunk else "",
            }
        )

    n = len(qa_rows)
    metrics = {
        f"Recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks
    }
    metrics["MRR"] = sum(1 / r for r in ranks if r is not None) / n
    metrics["latency_ms"] = sum(latencies) / n
    return metrics, details


def format_table(results: dict[str, dict]) -> str:
    """Pretty-print {method name: metrics} as a plain-text table."""
    if not results:
        return "(no results)"
    columns = list(next(iter(results.values())).keys())
    width = max(len(name) for name in results) + 2
    header = "Method".ljust(width) + "".join(c.rjust(11) for c in columns)
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        cells = "".join(
            (f"{metrics[c]:.1f}" if c == "latency_ms" else f"{metrics[c]:.3f}").rjust(11)
            for c in columns
        )
        lines.append(name.ljust(width) + cells)
    return "\n".join(lines)


def save_details(details: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)
