"""Basic correctness tests. Run:  python -m unittest discover -s tests -v

They check the plumbing (chunking, search, metric maths). They do NOT prove your
retrieval is good; that is what the evaluation on your own questions is for.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_sample_data  # noqa: E402
from studymate.evaluate import (  # noqa: E402
    calculate_verbatim_overlap,
    evaluate,
    filter_qa_by_split,
    load_qa,
)
from studymate.ingest import (  # noqa: E402
    Chunk,
    build_chunks,
    chunk_words,
    get_page_text,
    read_document_pages,
)
from studymate.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    RerankRetriever,
)
from studymate.answer import (  # noqa: E402
    ABSTENTION_TEXT,
    answer_question,
    build_answer_prompt,
    extract_citations,
)
import make_split  # noqa: E402
import run_baselines  # noqa: E402
import validate_qa  # noqa: E402
import make_eval_set  # noqa: E402
import review_questions  # noqa: E402
import run_all_experiments  # noqa: E402
from studymate.llm import FakeLLM, generate, set_active_llm  # noqa: E402


class ChunkingTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(len(chunk_words("one two three", 150, 30)), 1)

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(chunk_words("", 150, 30), [])

    def test_overlap_and_full_coverage(self):
        words = [f"w{i}" for i in range(100)]
        chunks = chunk_words(" ".join(words), chunk_size=40, overlap=10)
        split = [c.split() for c in chunks]
        self.assertTrue(all(len(s) <= 40 for s in split))
        for a, b in zip(split, split[1:]):  # consecutive chunks share exactly 10 words
            self.assertEqual(a[-10:], b[:10])
        self.assertEqual(split[0][0], "w0")
        self.assertEqual(split[-1][-1], "w99")  # nothing lost at the end

    def test_bad_overlap_rejected(self):
        with self.assertRaises(ValueError):
            chunk_words("a b c", chunk_size=10, overlap=10)


class FakeRetriever:
    """Returns a scripted page for each question so we can verify the metric maths."""

    name = "fake"

    def __init__(self, scripted: dict[str, list[int]]):
        self.scripted = scripted

    def search(self, query, k):
        pages = self.scripted[query][:k]
        return [(Chunk(f"c{i}", "doc.pdf", p, "x"), 1.0) for i, p in enumerate(pages)]


class MetricTests(unittest.TestCase):
    def test_recall_and_mrr(self):
        qa = [
            {"question": "q1", "gold": {("doc.pdf", 1)}},  # hit at rank 1
            {"question": "q2", "gold": {("doc.pdf", 2)}},  # hit at rank 3
            {"question": "q3", "gold": {("doc.pdf", 9)}},  # never found
        ]
        retriever = FakeRetriever({"q1": [1, 5, 6], "q2": [4, 5, 2], "q3": [1, 2, 3]})
        metrics, details = evaluate(retriever, qa, ks=(1, 3))
        self.assertAlmostEqual(metrics["Recall@1"], 1 / 3)
        self.assertAlmostEqual(metrics["Recall@3"], 2 / 3)
        self.assertAlmostEqual(metrics["MRR"], (1 + 1 / 3 + 0) / 3)
        self.assertEqual([d["first_relevant_rank"] for d in details], [1, 3, ""])


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        make_sample_data.OUT = tmp  # write the sample files to a temp folder
        make_sample_data.main()
        cls.chunks, cls.warnings = build_chunks(tmp)
        cls.qa = load_qa(tmp / "qa.csv")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_pdf_ingestion(self):
        self.assertEqual(self.warnings, [])
        self.assertEqual({c.page for c in self.chunks}, set(range(1, 9)))  # 8 pages
        self.assertTrue(all(c.source == "ml_notes.pdf" for c in self.chunks))
        self.assertGreater(len(self.chunks), 8)  # pages were split into several chunks

    def test_bm25_finds_obvious_keyword_match(self):
        hits = BM25Retriever(self.chunks).search("What does dropout do during training?", 3)
        self.assertEqual(hits[0][0].page, 7)

    def test_dense_plumbing_returns_chunk_itself(self):
        retriever = DenseRetriever(self.chunks, HashingEmbedder())
        target = self.chunks[5]
        top = retriever.search(target.text, 1)[0][0]
        self.assertEqual(top.chunk_id, target.chunk_id)

    def test_embedding_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as cache:
            DenseRetriever(self.chunks, HashingEmbedder(), cache_dir=cache)
            DenseRetriever(self.chunks, HashingEmbedder(), cache_dir=cache)
            self.assertEqual(len(list(Path(cache).glob("*.npy"))), 1)

    def test_bm25_pipeline_runs_and_is_reasonable(self):
        metrics, _ = evaluate(BM25Retriever(self.chunks), self.qa)
        # Loose floor: BM25 should find the right page for most sample questions.
        self.assertGreaterEqual(metrics["Recall@5"], 0.6)


class EvaluationToolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        # Create dummy documents
        cls.doc1 = cls.tmp / "test_doc1.txt"
        cls.doc1.write_text("Page one content with words for retrieval.", encoding="utf-8")
        cls.doc2 = cls.tmp / "test_doc2.txt"
        cls.doc2.write_text("Second document content with different vocabulary.", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_get_page_text(self):
        txt = get_page_text(self.doc1, 1)
        self.assertIn("Page one content", txt)
        with self.assertRaises(IndexError):
            get_page_text(self.doc1, 2)
        with self.assertRaises(IndexError):
            get_page_text(self.doc1, 0)

    def test_calculate_verbatim_overlap(self):
        q = "What is the retrieval content?"  # non-stopwords: retrieval, content (2)
        page = "Here is retrieval content and other information."
        ratio, matches, tokens = calculate_verbatim_overlap(q, page)
        self.assertEqual(tokens, ["retrieval", "content"])
        self.assertEqual(matches, ["retrieval", "content"])
        self.assertEqual(ratio, 1.0)

        q2 = "What is something completely unrelated?"
        ratio2, matches2, tokens2 = calculate_verbatim_overlap(q2, page)
        self.assertEqual(ratio2, 0.0)

    def test_qa_validation_checks(self):
        import csv

        qa_file = self.tmp / "qa_test.csv"
        rows = [
            {"question": "What is retrieval content?", "source": "test_doc1.txt", "page": "1"},
            {"question": "How does second document work?", "source": "test_doc2.txt", "page": "1"},
            {"question": "What spans two pages?", "source": "test_doc1.txt", "page": "1|1"},
            {"question": "What is retrieval content?", "source": "test_doc1.txt", "page": "1"},
            {"question": "Invalid page question?", "source": "test_doc1.txt", "page": "99"},
            {"question": "Missing file question?", "source": "missing.txt", "page": "1"},
        ]
        with open(qa_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["question", "source", "page"])
            w.writeheader()
            w.writerows(rows)

        results = validate_qa.validate_qa_file(qa_file, self.tmp)
        self.assertEqual(len(results["duplicates"]), 1)
        self.assertGreaterEqual(len(results["errors"]), 2)
        self.assertEqual(len(results["two_gold_pages"]), 1)
        self.assertGreaterEqual(len(results["too_easy"]), 1)
        self.assertIn("test_doc1.txt", results["file_counts"])

    def test_make_split_stratification_and_threshold(self):
        import csv
        import json

        qa_file = self.tmp / "qa_split.csv"
        rows = []
        for i in range(30):
            rows.append({"question": f"Question A{i} about doc1?", "source": "test_doc1.txt", "page": "1"})
        for i in range(30):
            rows.append({"question": f"Question B{i} about doc2?", "source": "test_doc2.txt", "page": "1"})

        with open(qa_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["question", "source", "page"])
            w.writeheader()
            w.writerows(rows)

        split_dict = make_split.create_stratified_split(rows, seed=42, dev_ratio=0.40)
        self.assertEqual(split_dict["total_questions"], 60)
        self.assertEqual(split_dict["dev_count"], 24)
        self.assertEqual(split_dict["test_count"], 36)
        self.assertEqual(split_dict["stratification"]["test_doc1.txt"]["dev"], 12)
        self.assertEqual(split_dict["stratification"]["test_doc2.txt"]["dev"], 12)

        split_path = self.tmp / "split.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_dict, f)

        dev_filtered = filter_qa_by_split(rows, split_path, "dev")
        self.assertEqual(len(dev_filtered), 24)
        test_filtered = filter_qa_by_split(rows, split_path, "test")
        self.assertEqual(len(test_filtered), 36)
        all_filtered = filter_qa_by_split(rows, split_path, "all")
        self.assertEqual(len(all_filtered), 60)

    def test_run_baselines_with_split_and_logging(self):
        import unittest.mock as mock

        log_out = self.tmp / "results"
        split_path = self.tmp / "split.json"
        qa_file = self.tmp / "qa_split.csv"

        test_args = [
            "run_baselines.py",
            "--docs", str(self.tmp),
            "--qa", str(qa_file),
            "--split", "dev",
            "--split-file", str(split_path),
            "--dense-model", "hashing",
            "--out", str(log_out),
        ]
        with mock.patch("sys.argv", test_args):
            run_baselines.main()

        self.assertTrue((log_out / "baselines.csv").exists())
        self.assertTrue((log_out / "runs.log").exists())
        log_content = (log_out / "runs.log").read_text(encoding="utf-8")
        self.assertIn("split:        dev", log_content)
        self.assertIn("dense[hashing-512]", log_content)


class SyntheticEvaluationTests(unittest.TestCase):
    def test_llm_fake_and_caching(self):
        fake = FakeLLM(default_response="What is gradient descent?")
        set_active_llm(fake)
        try:
            res1 = generate("Prompt 1")
            self.assertEqual(res1, "What is gradient descent?")
            self.assertEqual(len(fake.calls), 1)
        finally:
            set_active_llm(None)

    def test_rejection_rules(self):
        chunk_text = "Support vector machines find the optimal separating hyperplane that maximizes the geometric margin between data points."
        # Shares 4 consecutive words: "the optimal separating hyperplane"
        bad_q1 = "What is the optimal separating hyperplane in support vectors?"
        self.assertTrue(make_eval_set.find_shared_4grams(bad_q1, chunk_text))
        valid1, reason1 = make_eval_set.check_question_quality(bad_q1, chunk_text)
        self.assertFalse(valid1)
        self.assertIn("4+ consecutive", reason1)

        # High verbatim overlap (> 50%)
        bad_q2 = "How do machines find optimal geometric margin?"
        valid2, reason2 = make_eval_set.check_question_quality(bad_q2, chunk_text)
        self.assertFalse(valid2)

        # Well-paraphrased question
        good_q = "How does this linear classification method establish decision boundaries to best separate distinct categories?"
        valid3, reason3 = make_eval_set.check_question_quality(good_q, chunk_text)
        self.assertTrue(valid3)
        self.assertEqual(reason3, "")

    def test_sample_stratified_chunks(self):
        long_text = "word " * 70  # 70 words (> 60)
        short_text = "word " * 40  # 40 words (< 60)
        chunks = [
            Chunk("c1", "docA.pdf", 1, long_text),
            Chunk("c2", "docA.pdf", 2, short_text),  # skipped
            Chunk("c3", "docB.pdf", 1, long_text),
            Chunk("c4", "docB.pdf", 2, long_text),
        ]
        sampled = make_eval_set.sample_stratified_chunks(chunks, sample_size=2, seed=42, min_words=60)
        self.assertEqual(len(sampled), 2)
        # c2 should not be included
        self.assertTrue(all(c.chunk_id != "c2" for c in sampled))

    def test_review_and_qa_generation(self):
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            synth_csv = tmp / "qa_synth.csv"
            review_csv = tmp / "review.csv"
            qa_csv = tmp / "qa.csv"
            doc_file = tmp / "doc.txt"
            doc_file.write_text("Detailed study notes about machine learning algorithms.", encoding="utf-8")

            rows = [
                {"question": "Q1: What is ML?", "source": "doc.txt", "page": "1", "chunk_id": "c1", "type": "definition"},
                {"question": "Q2: Bad ambiguous question?", "source": "doc.txt", "page": "1", "chunk_id": "c2", "type": "why"},
            ]
            import csv
            with open(synth_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["question", "source", "page", "chunk_id", "type"])
                w.writeheader()
                w.writerows(rows)

            test_args = [
                "review_questions.py",
                "--synthetic-qa", str(synth_csv),
                "--docs", str(tmp),
                "--review-out", str(review_csv),
                "--qa-out", str(qa_csv),
                "--accept-all",
            ]
            with mock.patch("sys.argv", test_args):
                review_questions.main()

            self.assertTrue(review_csv.exists())
            self.assertTrue(qa_csv.exists())
            qa_rows = load_qa(qa_csv)
            self.assertEqual(len(qa_rows), 2)


class HybridAndAnswerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("c1", "doc1.pdf", 1, "Decision trees split data using information gain."),
            Chunk("c2", "doc1.pdf", 2, "Neural networks learn representations through backpropagation."),
            Chunk("c3", "doc2.pdf", 1, "Support vector machines maximize the separation margin."),
        ]
        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(self.chunks, HashingEmbedder())

    def test_hybrid_rrf_and_weighted(self):
        # Test RRF fusion
        hybrid_rrf = HybridRetriever(self.bm25, self.dense, method="rrf", rrf_k=60)
        hits_rrf = hybrid_rrf.search("decision trees", k=2)
        self.assertEqual(len(hits_rrf), 2)
        self.assertEqual(hits_rrf[0][0].source, "doc1.pdf")

        # Test weighted min-max score fusion
        hybrid_weighted = HybridRetriever(self.bm25, self.dense, method="weighted", dense_weight=0.6)
        hits_weighted = hybrid_weighted.search("support vector margin", k=2)
        self.assertEqual(len(hits_weighted), 2)
        self.assertEqual(hits_weighted[0][0].source, "doc2.pdf")

    def test_citations_and_answer_prompt(self):
        text = "This algorithm optimizes margins [doc2.pdf p.1] and backprop [doc1.pdf p.2]."
        citations = extract_citations(text)
        self.assertEqual(citations, [("doc2.pdf", 1), ("doc1.pdf", 2)])

        prompt = build_answer_prompt("What is backprop?", self.chunks[:2])
        self.assertIn("Context Passages:", prompt)
        self.assertIn("Question: What is backprop?", prompt)
        self.assertIn(ABSTENTION_TEXT, prompt)

    def test_answer_generation_and_abstention(self):
        fake_ans = FakeLLM(default_response="Decision trees split using gain [doc1.pdf p.1].")
        res = answer_question("How do trees split?", self.bm25, k=2, client=fake_ans)
        self.assertFalse(res.is_abstention)
        self.assertIn("doc1.pdf p.1", res.citations)
        self.assertEqual(len(res.scores), 2)
        self.assertGreaterEqual(res.retrieval_latency_ms, 0.0)

        fake_abstain = FakeLLM(default_response=ABSTENTION_TEXT)
        res_abstain = answer_question("What is the history of Rome?", self.bm25, k=2, client=fake_abstain)
        self.assertTrue(res_abstain.is_abstention)
        self.assertEqual(res_abstain.answer, ABSTENTION_TEXT)
        self.assertEqual(res_abstain.citations, [])

    def test_fusion_math_and_validation(self):
        # Verify RRF formula 1 / (rrf_k + rank)
        hybrid_rrf = HybridRetriever(self.bm25, self.dense, method="rrf", rrf_k=60)
        hits = hybrid_rrf.search("Decision trees", k=3)
        self.assertEqual(len(hits), 3)
        # Scores should be strictly positive and sorted descending
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], 0.0)

        # Invalid method validation
        with self.assertRaises(ValueError):
            HybridRetriever(self.bm25, self.dense, method="invalid_mode")

    def test_citation_verification_valid_and_invalid(self):
        # 1. Complex extraction
        text = "Fact A [notes.pdf p.4] and Fact B [lecture.pdf p.12]. Ignore [invalid] and [doc.pdf p.abc]."
        citations = extract_citations(text)
        self.assertEqual(citations, [("notes.pdf", 4), ("lecture.pdf", 12)])

        # 2. Check validity against retrieved context
        retrieved_pairs = {("notes.pdf", 4), ("notes.pdf", 5), ("lecture.pdf", 12)}
        valid = [1 if (src, pg) in retrieved_pairs else 0 for src, pg in citations]
        self.assertEqual(valid, [1, 1])

        # Hallucinated citation
        fake_citations = [("hallucinated.pdf", 99)]
        hallucinated_valid = [1 if (src, pg) in retrieved_pairs else 0 for src, pg in fake_citations]
        self.assertEqual(hallucinated_valid, [0])

    def test_abstention_variations_and_empty_retrieval(self):
        # 1. Exact phrase
        fake_llm = FakeLLM(default_response=ABSTENTION_TEXT)
        res = answer_question("Any query", self.bm25, k=2, client=fake_llm)
        self.assertTrue(res.is_abstention)
        self.assertEqual(res.answer, ABSTENTION_TEXT)

        # 2. Quoted and lowercase variations
        fake_quoted = FakeLLM(default_response=f'"{ABSTENTION_TEXT}"')
        res_quoted = answer_question("Any query", self.bm25, k=2, client=fake_quoted)
        self.assertTrue(res_quoted.is_abstention)

        # 3. Empty chunks should abstain immediately without calling LLM
        empty_retriever = BM25Retriever([])
        res_empty = answer_question("Any query", empty_retriever, k=2, client=fake_llm)
        self.assertTrue(res_empty.is_abstention)
        self.assertEqual(res_empty.chunks, [])
        self.assertEqual(res_empty.scores, [])

    def test_summary_markdown_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            summary_path = run_all_experiments.generate_summary_markdown(tmp, "data/sample", "data/sample/qa.csv")
            self.assertTrue(summary_path.exists())
            content = summary_path.read_text(encoding="utf-8")
            self.assertIn("# StudyMate AI — Comprehensive Evaluation Summary", content)
            self.assertIn("## 1. Overview and Evaluation Setup", content)


if __name__ == "__main__":
    unittest.main()
