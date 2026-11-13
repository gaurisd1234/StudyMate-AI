"""End-to-end evaluation of RAG answer generation.

Evaluates 40 answerable questions from the TEST split plus 15 out-of-scope questions.

Metrics reported:
1. Citation validity: percentage of cited pages that genuinely exist in the retrieved context.
2. Abstention rate: percentage of out-of-scope questions where the model correctly replied
   "I could not find this in the provided material."
3. LLM-judged answer match score: does the generated answer match the gold reference passage?
4. LLM-judged faithfulness score: is every claim in the answer strictly supported by the retrieved context?
5. Latency per stage: retrieval latency (ms), generation latency (ms), and end-to-end latency (ms).

Saves 20 judged examples to results/judged_answers_sample.csv for manual inspection.

Usage:
    python scripts/eval_answers.py
    python scripts/eval_answers.py --docs data/docs --qa data/eval/qa.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studymate.answer import ABSTENTION_TEXT, answer_question  # noqa: E402
from studymate.evaluate import filter_qa_by_split, load_qa  # noqa: E402
from studymate.ingest import build_chunks, get_page_text  # noqa: E402
from studymate.llm import FakeLLM, generate, set_active_llm  # noqa: E402
from studymate.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    SentenceTransformerEmbedder,
)

CURATED_OUT_OF_SCOPE = [
    "What specific historical events led up to the signing of the Magna Carta in 1215?",
    "How does the aldol condensation mechanism proceed under basic conditions in organic chemistry?",
    "What physiological adaptations enable emperor penguins to withstand extreme Antarctic winters?",
    "Why did Filippo Brunelleschi design a double-shell dome for the Florence Cathedral?",
    "How do oceanic-continental subduction zones generate volcanic mountain ranges in plate tectonics?",
    "What distinguishing architectural proportions differentiate Doric and Corinthian orders in ancient Greece?",
    "How does Johann Sebastian Bach construct a multi-voice invertible counterpoint in a fugue?",
    "What prevailing hypotheses account for the Late Bronze Age collapse around 1200 BCE?",
    "How does the spatial separation of CO2 fixation in C4 plants reduce photorespiration compared to C3 plants?",
    "What vector dynamics explain why the Black Death pandemic spread so rapidly through fourteenth-century Europe?",
    "How do astronomers use absorption spectroscopy to identify elements in stellar atmospheres?",
    "What phonological vowel shifts characterized the Great Vowel Shift in Early Modern English?",
    "How does the renin-angiotensin-aldosterone hormonal pathway regulate arterial blood pressure?",
    "What economic factors precipitated the collapse of the 1637 Dutch tulip speculation bubble?",
    "How do directional stress and heat recrystallize minerals to form foliation in metamorphic gneiss?",
]


def judge_answer_match(question: str, gold_text: str, answer: str) -> float:
    """LLM judge: does the answer correctly reflect the gold chunk?"""
    if answer.strip().lower() == ABSTENTION_TEXT.lower():
        return 0.0

    prompt = (
        f"You are an impartial evaluator assessing whether an AI-generated answer matches the ground truth information.\n\n"
        f"Question: {question}\n\n"
        f"Gold Reference Passage:\n{gold_text[:1200]}\n\n"
        f"AI Answer:\n{answer}\n\n"
        f"Criteria:\n"
        f"- Output 1 if the AI answer accurately conveys the facts and concepts from the gold reference passage without factual distortion.\n"
        f"- Output 0 if the AI answer contradicts the gold reference passage, gives incorrect information, or is completely off-topic.\n\n"
        f"Answer with ONLY the number 1 or 0:"
    )
    res = generate(prompt).strip()
    return 1.0 if "1" in res else 0.0


def judge_faithfulness(context_text: str, answer: str) -> float:
    """LLM judge: is every claim in the answer supported by the retrieved context?"""
    if answer.strip().lower() == ABSTENTION_TEXT.lower():
        # Correctly abstaining when unsure is 100% faithful
        return 1.0

    prompt = (
        f"You are an impartial evaluator assessing the faithfulness of an answer to its provided context passages.\n\n"
        f"Retrieved Context Passages:\n{context_text[:1500]}\n\n"
        f"AI Answer:\n{answer}\n\n"
        f"Criteria:\n"
        f"- Output 1 if EVERY factual claim in the AI answer is directly supported by the context passages above (no hallucinated external facts).\n"
        f"- Output 0 if the answer introduces unsupported claims, speculative facts, or outside information not found in the context.\n\n"
        f"Answer with ONLY the number 1 or 0:"
    )
    res = generate(prompt).strip()
    return 1.0 if "1" in res else 0.0


def get_out_of_scope_questions(n: int = 15, seed: int = 42) -> list[str]:
    """Return 15 realistic student questions on topics outside the course scope."""
    rng = random.Random(seed)
    if n <= len(CURATED_OUT_OF_SCOPE):
        return rng.sample(CURATED_OUT_OF_SCOPE, n)

    # If more are needed, ask the LLM
    prompt = (
        f"Generate {n} realistic, distinct study questions that university students might ask in biology, "
        f"ancient history, geology, or art history. They must be completely unrelated to machine learning, "
        f"computer science, mathematics, or neural networks.\n"
        f"Output one question per line without numbers or prefixes."
    )
    raw = generate(prompt)
    lines = [line.strip().lstrip("-*0123456789. ") for line in raw.splitlines() if line.strip()]
    return lines[:n] if len(lines) >= n else CURATED_OUT_OF_SCOPE[:n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--docs", default="data/docs", help="folder with source documents")
    parser.add_argument("--qa", default="data/eval/qa.csv", help="question,source,page CSV")
    parser.add_argument("--split-file", default="data/eval/split.json", help="path to split.json")
    parser.add_argument("--k", type=int, default=5, help="number of chunks to retrieve for context")
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling test questions")
    parser.add_argument("--sample-save", type=int, default=20, help="how many judged examples to save to CSV")
    parser.add_argument("--out-csv", default="results/judged_answers_sample.csv", help="path for saved judged examples")
    parser.add_argument("--fake-llm", action="store_true", help="use FakeLLM for offline testing")
    args = parser.parse_args()

    docs_dir = (ROOT / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)
    qa_path = (ROOT / args.qa).resolve() if not Path(args.qa).is_absolute() else Path(args.qa)
    split_path = (ROOT / args.split_file).resolve() if not Path(args.split_file).is_absolute() else Path(args.split_file)
    out_csv_path = (ROOT / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)

    if args.fake_llm or not os.environ.get("GEMINI_API_KEY"):
        if not os.environ.get("GEMINI_API_KEY"):
            print("Notice: GEMINI_API_KEY not set in .env; using FakeLLM for simulation.")
        set_active_llm(FakeLLM())

    print("Building chunks and setting up retrieval pipeline...")
    chunks, _ = build_chunks(docs_dir, chunk_size=150, overlap=30)
    bm25 = BM25Retriever(chunks)
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    dense = DenseRetriever(chunks, embedder, cache_dir=ROOT / ".cache")
    hybrid = HybridRetriever(bm25, dense, method="rrf", rrf_k=60)
    reranker = RerankRetriever(base_retriever=hybrid, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k_candidates=20)

    # Load 40 answerable test questions
    qa_all = load_qa(qa_path)
    qa_test = filter_qa_by_split(qa_all, split_path, "test")

    rng = random.Random(args.seed)
    num_answerable = min(40, len(qa_test))
    answerable_questions = rng.sample(qa_test, num_answerable)

    out_of_scope_questions = get_out_of_scope_questions(n=15, seed=args.seed)

    print(f"\nEvaluating RAG pipeline on:")
    print(f"  - {len(answerable_questions)} answerable questions from TEST split")
    print(f"  - {len(out_of_scope_questions)} out-of-scope questions\n")

    # Document cache for gold text
    doc_pages_cache: dict[str, list[tuple[int, str]]] = {}

    def get_gold_text(source: str, page_num: int) -> str:
        if source not in doc_pages_cache:
            p = docs_dir / source
            if not p.is_file():
                for cand in docs_dir.rglob("*"):
                    if cand.is_file() and cand.name.lower() == source.lower():
                        p = cand
                        break
            try:
                from studymate.ingest import read_document_pages
                doc_pages_cache[source] = read_document_pages(p)
            except Exception:
                doc_pages_cache[source] = []
        pages = doc_pages_cache.get(source, [])
        if 1 <= page_num <= len(pages):
            return pages[page_num - 1][1]
        return ""

    eval_records: list[dict] = []

    # Latency tracking
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []

    # Metric accumulators
    valid_citation_answers = 0
    total_cited_answers = 0
    answerable_matches = 0
    answerable_faithfulness = 0
    out_of_scope_abstentions = 0

    print("Running evaluation on answerable questions...")
    for idx, row in enumerate(answerable_questions, start=1):
        q = row["question"]
        source = row["source"]
        gold_page = min(p for _, p in row["gold"]) if row.get("gold") else int(str(row["page"]).split("|")[0])
        gold_text = get_gold_text(source, gold_page)

        ans_res = answer_question(q, reranker, k=args.k)
        retrieval_latencies.append(ans_res.retrieval_latency_ms)
        generation_latencies.append(ans_res.generation_latency_ms)

        retrieved_set = {(c.source, c.page) for c in ans_res.chunks}
        retrieved_summary = "; ".join(f"{c.source}:p{c.page}" for c in ans_res.chunks)

        # Check citation validity
        is_citation_valid = True
        if ans_res.cited_pages:
            total_cited_answers += 1
            all_in_context = all((s, p) in retrieved_set for s, p in ans_res.cited_pages)
            if all_in_context:
                valid_citation_answers += 1
            else:
                is_citation_valid = False

        # LLM Judges
        context_concat = "\n\n".join(f"[{c.source} p.{c.page}]\n{c.text}" for c in ans_res.chunks)
        match_score = judge_answer_match(q, gold_text, ans_res.answer)
        faith_score = judge_faithfulness(context_concat, ans_res.answer)

        if match_score == 1.0:
            answerable_matches += 1
        if faith_score == 1.0:
            answerable_faithfulness += 1

        eval_records.append(
            {
                "question": q,
                "category": "answerable",
                "answer": ans_res.answer,
                "citations": ", ".join(ans_res.citations),
                "retrieved_sources": retrieved_summary,
                "citation_valid": 1 if is_citation_valid else 0,
                "match_score": match_score,
                "faithfulness_score": faith_score,
                "retrieval_ms": round(ans_res.retrieval_latency_ms, 2),
                "generation_ms": round(ans_res.generation_latency_ms, 2),
                "gold_source_page": f"{source} p.{gold_page}",
            }
        )
        if idx % 10 == 0 or idx == len(answerable_questions):
            print(f"  Processed {idx}/{len(answerable_questions)} answerable questions...")

    print("\nRunning evaluation on out-of-scope questions...")
    for idx, q in enumerate(out_of_scope_questions, start=1):
        ans_res = answer_question(q, reranker, k=args.k)
        retrieval_latencies.append(ans_res.retrieval_latency_ms)
        generation_latencies.append(ans_res.generation_latency_ms)

        retrieved_set = {(c.source, c.page) for c in ans_res.chunks}
        retrieved_summary = "; ".join(f"{c.source}:p{c.page}" for c in ans_res.chunks)

        if ans_res.is_abstention:
            out_of_scope_abstentions += 1
            citation_valid = 1
            faith_score = 1.0
        else:
            if ans_res.cited_pages:
                citation_valid = 1 if all((s, p) in retrieved_set for s, p in ans_res.cited_pages) else 0
            else:
                citation_valid = 1
            context_concat = "\n\n".join(f"[{c.source} p.{c.page}]\n{c.text}" for c in ans_res.chunks)
            faith_score = judge_faithfulness(context_concat, ans_res.answer)

        eval_records.append(
            {
                "question": q,
                "category": "out_of_scope",
                "answer": ans_res.answer,
                "citations": ", ".join(ans_res.citations),
                "retrieved_sources": retrieved_summary,
                "citation_valid": citation_valid,
                "match_score": 0.0,
                "faithfulness_score": faith_score,
                "retrieval_ms": round(ans_res.retrieval_latency_ms, 2),
                "generation_ms": round(ans_res.generation_latency_ms, 2),
                "gold_source_page": "N/A (out-of-scope)",
            }
        )

    # Save 20 judged examples
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_records = eval_records[:args.sample_save]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(sample_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_records)

    # Summary calculations
    n_ans = len(answerable_questions)
    n_oos = len(out_of_scope_questions)
    total_q = n_ans + n_oos

    citation_validity_pct = (valid_citation_answers / total_cited_answers * 100) if total_cited_answers > 0 else 100.0
    abstention_rate_pct = (out_of_scope_abstentions / n_oos * 100) if n_oos > 0 else 0.0
    match_score_pct = (answerable_matches / n_ans * 100) if n_ans > 0 else 0.0
    faithfulness_pct = (answerable_faithfulness / n_ans * 100) if n_ans > 0 else 0.0

    avg_retrieval_ms = sum(retrieval_latencies) / total_q
    avg_gen_ms = sum(generation_latencies) / total_q
    avg_total_ms = avg_retrieval_ms + avg_gen_ms

    sep = "=" * 80
    print("\n" + sep)
    print("StudyMate AI - RAG Answer Generation Evaluation Report")
    print(sep)
    print(f"Total Questions Evaluated:     {total_q} ({n_ans} answerable test + {n_oos} out-of-scope)")
    print(f"Retriever Pipeline:            Hybrid (RRF k=60) + Cross-Encoder Reranker (top 20)")
    print(f"Context Passages per Query:    k = {args.k}")
    print("-" * 80)
    print("KEY METRICS:")
    print(f"  1. Citation Validity:        {citation_validity_pct:5.1f}% ({valid_citation_answers}/{total_cited_answers} cited answers with 100% valid citations)")
    print(f"  2. Out-of-Scope Abstention:  {abstention_rate_pct:5.1f}% ({out_of_scope_abstentions}/{n_oos} correctly abstained with exact text)")
    print(f"  3. Answer Match Score (LLM): {match_score_pct:5.1f}% ({answerable_matches}/{n_ans} answers match gold source facts)")
    print(f"  4. Faithfulness Score (LLM): {faithfulness_pct:5.1f}% ({answerable_faithfulness}/{n_ans} answers fully grounded in context)")
    print("-" * 80)
    print("LATENCY PER STAGE:")
    print(f"  - Retrieval Latency:         {avg_retrieval_ms:6.1f} ms")
    print(f"  - Generation Latency:        {avg_gen_ms:6.1f} ms")
    print(f"  - Total End-to-End Latency:  {avg_total_ms:6.1f} ms")
    print("-" * 80)
    print(f"Saved {len(sample_records)} judged examples to: {out_csv_path}")
    print()
    print("Note: LLM-as-a-judge evaluations are imperfect heuristics subject to judge bias and inconsistency; manual verification on the saved CSV sample is recommended.")
    print(sep)


if __name__ == "__main__":
    main()

