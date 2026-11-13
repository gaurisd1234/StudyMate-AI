"""Generate synthetic evaluation questions using LLM with quality filtering.

1. Samples 150 chunks (seed 42, chunks >= 60 words), stratified across documents.
2. Asks the LLM for 1 student-style question per chunk, cycling through 5 types:
   definition, why, how, compare, application.
3. Enforces paraphrasing:
   Rejects questions with >=4 consecutive shared words with the chunk or >50% non-stopword overlap.
   Retries up to 3 times per chunk.
4. Saves data/eval/qa_synthetic.csv (question,source,page,chunk_id,type).
5. Prints distribution by document and type, plus rejection statistics.

Usage:
    python scripts/make_eval_set.py
    python scripts/make_eval_set.py --sample-size 150 --seed 42
    python scripts/make_eval_set.py --fake-llm
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studymate.ingest import Chunk, build_chunks  # noqa: E402
from studymate.llm import FakeLLM, generate, set_active_llm  # noqa: E402
from studymate.retrievers import STOPWORDS  # noqa: E402

QUESTION_TYPES = ["definition", "why", "how", "compare", "application"]

TYPE_GUIDANCE = {
    "definition": "Ask for the definition, meaning, or core identity of a key concept or term in the passage.",
    "why": "Ask for the reason, motivation, cause, or theoretical rationale behind a concept or method in the passage.",
    "how": "Ask how a specific process, mechanism, method, or algorithm works or is carried out in the passage.",
    "compare": "Ask how two concepts, approaches, components, or options contrast or differ, or what trade-offs exist.",
    "application": "Ask about a practical use case, real-world scenario, or problem where the technique or concept applies.",
}


def find_shared_4grams(question: str, chunk_text: str) -> list[str]:
    """Find any sequence of 4 consecutive words shared verbatim between question and chunk."""
    q_words = re.findall(r"\w+", question.lower())
    c_words = re.findall(r"\w+", chunk_text.lower())
    if len(q_words) < 4 or len(c_words) < 4:
        return []
    c_4grams = {tuple(c_words[i : i + 4]): " ".join(c_words[i : i + 4]) for i in range(len(c_words) - 3)}
    shared = []
    for i in range(len(q_words) - 3):
        gram = tuple(q_words[i : i + 4])
        if gram in c_4grams:
            shared.append(c_4grams[gram])
    return shared


def get_non_stopword_overlap(question: str, chunk_text: str) -> tuple[float, list[str], list[str]]:
    """Compute verbatim non-stopword overlap ratio between question and chunk text."""
    q_tokens = [w for w in re.findall(r"\w+", question.lower()) if w not in STOPWORDS]
    if not q_tokens:
        return 0.0, [], []
    c_words = set(re.findall(r"\w+", chunk_text.lower()))
    matches = [w for w in q_tokens if w in c_words]
    ratio = len(matches) / len(q_tokens)
    return ratio, matches, q_tokens


def check_question_quality(question: str, chunk_text: str) -> tuple[bool, str]:
    """Check question against paraphrasing constraints.

    Returns (is_valid, rejection_reason).
    """
    if not question or len(question.split()) < 3:
        return False, "Question is too short or empty."

    # Rule 1: No 4+ consecutive words shared
    shared_4grams = find_shared_4grams(question, chunk_text)
    if shared_4grams:
        return False, f"Shares 4+ consecutive words with chunk: '{shared_4grams[0]}'"

    # Rule 2: Non-stopword overlap <= 50%
    overlap_ratio, matches, tokens = get_non_stopword_overlap(question, chunk_text)
    if overlap_ratio > 0.50:
        return False, f"Non-stopword overlap is {overlap_ratio * 100:.1f}% (> 50% threshold; matches: {', '.join(matches[:5])})"

    return True, ""


def sample_stratified_chunks(
    chunks: list[Chunk],
    sample_size: int = 150,
    seed: int = 42,
    min_words: int = 60,
) -> list[Chunk]:
    """Filter chunks by length and sample proportionally stratified across documents."""
    valid_chunks = [c for c in chunks if len(c.text.split()) >= min_words]
    if not valid_chunks:
        raise ValueError(f"No chunks have at least {min_words} words.")

    rng = random.Random(seed)
    doc_groups: dict[str, list[Chunk]] = defaultdict(list)
    for c in valid_chunks:
        doc_groups[c.source].append(c)

    total_valid = len(valid_chunks)
    target_total = min(sample_size, total_valid)

    # Proportional allocation
    sampled: list[Chunk] = []
    allocated_counts: dict[str, int] = {}
    remaining = target_total

    sorted_sources = sorted(doc_groups.keys())
    for idx, source in enumerate(sorted_sources):
        src_chunks = doc_groups[source]
        # Sort deterministically
        src_chunks.sort(key=lambda c: (c.page, c.chunk_id))
        if idx == len(sorted_sources) - 1:
            count = remaining
        else:
            count = int(round(target_total * (len(src_chunks) / total_valid)))
            count = max(1, min(count, len(src_chunks), remaining))
            remaining -= count
        allocated_counts[source] = count

    for source in sorted_sources:
        src_chunks = list(doc_groups[source])
        rng.shuffle(src_chunks)
        sampled.extend(src_chunks[: allocated_counts[source]])

    # Final deterministic order
    sampled.sort(key=lambda c: (c.source, c.page, c.chunk_id))
    return sampled


def generate_question_for_chunk(
    chunk: Chunk,
    q_type: str,
    max_tries: int = 3,
) -> tuple[str | None, int, list[str]]:
    """Generate and validate a question for a chunk with up to max_tries attempts.

    Returns (accepted_question, attempts_used, rejection_reasons).
    """
    system_prompt = (
        "You are an expert study assistant creating realistic exam preparation questions for university students. "
        "Your questions must be natural, conceptually rich, and rigorously paraphrased."
    )

    rejection_reasons: list[str] = []
    last_reason = ""

    for attempt in range(1, max_tries + 1):
        prompt_lines = [
            f"Read the following study material passage and generate ONE question of type '{q_type.upper()}'.",
            f"Question Type: {q_type.upper()}",
            f"Guidance: {TYPE_GUIDANCE[q_type]}",
            "",
            "Rules you must strictly follow:",
            "1. Realistic Student Question: Write naturally as a student asking about this topic.",
            "2. Answerable: The question MUST be fully answerable using ONLY information in this passage.",
            "3. Strict Paraphrasing: Express all ideas using YOUR OWN words. Do NOT copy phrases from the passage.",
            "4. NO 4-WORD SEQUENCES: Under no circumstances copy 4 or more consecutive words from the passage.",
            "5. Low Overlap: More than half of your non-stopword words must NOT appear in the passage.",
            "6. Standalone: Do NOT say 'according to the passage' or 'in the text'. Make it a direct conceptual question.",
            "7. Output format: Output ONLY the question itself on a single line. No quotes, numbering, or prefixes.",
            "",
            "Passage:",
            f'"""{chunk.text}"""',
        ]
        if last_reason:
            prompt_lines.append("")
            prompt_lines.append(f"PREVIOUS ATTEMPT REJECTED: {last_reason}")
            prompt_lines.append("Please rewrite the question with completely new phrasing to avoid verbatim word overlap.")

        prompt_lines.append("\nQuestion:")
        full_prompt = "\n".join(prompt_lines)

        try:
            raw_response = generate(full_prompt, system=system_prompt)
        except Exception as error:
            rejection_reasons.append(f"LLM generation failed: {error}")
            break

        # Clean response
        clean_q = raw_response.strip().strip('"').strip("'")
        clean_q = re.sub(r"^(question\s*\d*[:.-]?\s*)", "", clean_q, flags=re.IGNORECASE).strip()
        # Take first non-empty line
        lines = [line.strip() for line in clean_q.splitlines() if line.strip()]
        candidate_q = lines[0] if lines else ""

        is_valid, reason = check_question_quality(candidate_q, chunk.text)
        if is_valid:
            return candidate_q, attempt, rejection_reasons

        last_reason = reason
        rejection_reasons.append(f"Attempt {attempt}: {reason}")

    return None, max_tries, rejection_reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--docs", default="data/docs", help="directory with documents")
    parser.add_argument("--out", default="data/eval/qa_synthetic.csv", help="output CSV path")
    parser.add_argument("--sample-size", type=int, default=150, help="number of chunks to sample (default: 150)")
    parser.add_argument("--seed", type=int, default=42, help="fixed seed for sampling and shuffling")
    parser.add_argument("--min-words", type=int, default=60, help="minimum words for a chunk (default: 60)")
    parser.add_argument("--max-retries", type=int, default=3, help="max generation tries per chunk (default: 3)")
    parser.add_argument("--fake-llm", action="store_true", help="use FakeLLM for testing or offline generation")
    args = parser.parse_args()

    docs_dir = (ROOT / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)
    out_path = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    if args.fake_llm:
        print("Using FakeLLM (mock mode).")
        set_active_llm(FakeLLM())
    else:
        # Check API key if not in mock mode
        if not os.environ.get("GEMINI_API_KEY"):
            print("Notice: GEMINI_API_KEY is not set in .env or environment.")
            print("Falling back to FakeLLM to generate reproducible offline evaluation questions.")
            print("(To use Google Gemini, add GEMINI_API_KEY=your_key in .env)\n")
            set_active_llm(FakeLLM())

    print(f"Building chunks from {docs_dir}...")
    chunks, warnings = build_chunks(docs_dir, chunk_size=150, overlap=30)
    for w in warnings:
        print("  WARNING:", w)

    if not chunks:
        sys.exit(f"Error: No chunks extracted from {docs_dir}.")

    sampled_chunks = sample_stratified_chunks(
        chunks=chunks,
        sample_size=args.sample_size,
        seed=args.seed,
        min_words=args.min_words,
    )

    print(
        f"Sampled {len(sampled_chunks)} chunks (seed={args.seed}, min_words={args.min_words}) "
        f"stratified across {len({c.source for c in sampled_chunks})} document(s).\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    attempts_distribution: Counter = Counter()
    rejection_counts: Counter = Counter()
    total_rejections = 0

    print("Generating and validating questions...")
    for idx, chunk in enumerate(sampled_chunks, start=1):
        q_type = QUESTION_TYPES[(idx - 1) % len(QUESTION_TYPES)]
        question, attempts_used, rejections = generate_question_for_chunk(
            chunk=chunk,
            q_type=q_type,
            max_tries=args.max_retries,
        )

        attempts_distribution[attempts_used] += 1
        total_rejections += len(rejections)
        for r in rejections:
            if "4+ consecutive" in r:
                rejection_counts["4+ shared consecutive words"] += 1
            elif "overlap is" in r or "threshold" in r:
                rejection_counts[">50% non-stopword overlap"] += 1
            else:
                rejection_counts["other rejection"] += 1

        if question:
            rows.append(
                {
                    "question": question,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "type": q_type,
                }
            )

        if idx % 10 == 0 or idx == len(sampled_chunks):
            print(f"  Processed {idx}/{len(sampled_chunks)} chunks ({len(rows)} valid questions saved)...")

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question", "source", "page", "chunk_id", "type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sep = "=" * 80
    print("\n" + sep)
    print("StudyMate AI - Synthetic Question Generation Summary")
    print(sep)
    print(f"Output File:           {out_path}")
    print(f"Total Chunks Sampled:  {len(sampled_chunks)}")
    print(f"Questions Generated:   {len(rows)} (success rate: {len(rows) / len(sampled_chunks) * 100:.1f}%)")

    print("\n[1] Distribution by Source Document:")
    doc_counter = Counter(r["source"] for r in rows)
    for doc, count in sorted(doc_counter.items()):
        print(f"  - {doc:<40} : {count:>4} questions ({count / len(rows) * 100:.1f}%)")

    print("\n[2] Distribution by Question Type:")
    type_counter = Counter(r["type"] for r in rows)
    for q_t in QUESTION_TYPES:
        count = type_counter[q_t]
        print(f"  - {q_t:<15} : {count:>4} questions ({count / len(rows) * 100:.1f}%)")

    print("\n[3] Rejection & Regeneration Statistics:")
    print(f"  - Total initial rejections:  {total_rejections}")
    print(f"  - Accepted on Attempt 1:     {attempts_distribution[1]} ({attempts_distribution[1] / len(sampled_chunks) * 100:.1f}%)")
    print(f"  - Accepted on Attempt 2:     {attempts_distribution[2]} ({attempts_distribution[2] / len(sampled_chunks) * 100:.1f}%)")
    print(f"  - Accepted on Attempt 3:     {attempts_distribution[3]} ({attempts_distribution[3] / len(sampled_chunks) * 100:.1f}%)")
    if rejection_counts:
        print("  - Rejection Reasons Encountered:")
        for reason, count in rejection_counts.items():
            print(f"      * {reason}: {count}")

    print(sep)


if __name__ == "__main__":
    main()

