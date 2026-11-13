# Week 1 guide: what you are building and why

Read this once before running anything, then keep it open while you read the code.
Each section ends with a question you should be able to answer out loud.

## 0. The big picture

RAG (Retrieval-Augmented Generation) has two halves:

1. **Retrieve**: find the few passages in your material that are relevant to a question.
2. **Generate**: give those passages plus the question to an LLM so it answers from them.

Week 1 builds and measures half 1 only. Everything else depends on it: if retrieval hands the LLM the wrong passages, the answer will be wrong however good the LLM is.

## 1. Chunking (`ingest.py`)

PDFs are too big to search or feed to an LLM whole, so we split each page into windows of about 150 words that overlap by 30 words.

- **Too big a chunk**: it mixes several topics, so it matches many questions weakly, and you waste LLM context.
- **Too small a chunk**: an idea is cut in half, so no single chunk holds the answer.
- **Overlap** protects ideas that fall on a boundary.
- Chunks never cross pages, so each one keeps an exact page number for citations and for evaluation.
- **Scanned PDFs** are pictures with no text. The code warns you; they need OCR, which we are not doing yet.

*Check yourself: why does overlap help? What would happen to your results if chunks were 20 words long?*

## 2. BM25: keyword search (`BM25Retriever`)

BM25 scores a chunk against a question using three intuitions:

1. A chunk that contains the question's words is more relevant.
2. A **rare** word ("vanishing") is stronger evidence than a common one ("model"). This is the IDF idea.
3. Repeating a word helps only up to a point, and long chunks are slightly penalized, so a long chunk does not win just by being long.

It needs no training and no GPU, and it is very good when the question reuses the document's words. It fails when the question is a paraphrase ("forget things from many steps ago" vs. "vanishing gradient").

*Check yourself: why is "the" ignored, and why does a rare word count more?*

## 3. Dense retrieval: search by meaning (`DenseRetriever`)

An **embedding model** reads text and outputs a vector (a list of 384 numbers for MiniLM). Texts with similar meaning get vectors pointing in similar directions.

- **Cosine similarity** measures the angle between two vectors. If vectors are scaled to length 1 (we do this with `normalize_embeddings=True`), cosine similarity equals the plain dot product.
- **FAISS** is a library that finds the closest vectors quickly. We use the exact version (`IndexFlatIP`), which is fine for thousands of chunks.
- Embedding all chunks is the slow step, so we cache it, and the cache key changes whenever the model or any text changes.

Dense search can match paraphrases, but it can miss exact terms such as names, formulas, and codes, which BM25 handles well. That is exactly why Week 3 builds a **hybrid** of both.

*Check yourself: when would BM25 beat dense search, and when the reverse?*

## 4. Evaluation (`evaluate.py`)

A system you have not measured is a demo, not a project.

- You write questions and label the **page** that answers each one.
- A result counts as correct if a retrieved chunk comes from that page.
- **Recall@k**: for what fraction of questions does a correct chunk appear in the top k?
- **MRR**: average of 1/rank of the first correct result. Rank 1 gives 1.0, rank 2 gives 0.5, rank 3 gives 0.33, a miss gives 0. It rewards putting the answer first.

Worked example, 3 questions with the right page at rank 1, rank 3, and not found:
Recall@1 = 1/3, Recall@3 = 2/3, MRR = (1 + 0.33 + 0) / 3 = 0.44. The unit test `test_recall_and_mrr` checks exactly this.

*Check yourself: why does MRR 0.44 look worse than Recall@3 of 0.67 even though they describe the same results?*

## Important: what the sample results mean

On the bundled sample, BM25 scores 1.000 on every metric. **That is not a result.** The sample has only 12 chunks on 8 very different topics, so any method finds the right page. A real evaluation needs:

- a larger corpus (hundreds of pages) where topics overlap,
- **questions you write yourself** (at least 60 to 80), including paraphrases that avoid the notes' exact wording,
- a few questions whose answer spans two pages.

Only then will BM25 and dense search separate, and that gap is the interesting finding.

## 5. Your Week 1 checklist

1. [ ] Create the venv, install the requirements, run the unit tests (all should pass).
2. [ ] Run the sample with `--dense-model none`, then with the default model. Open `results/per_question_*.csv` and read it.
3. [ ] Read `ingest.py`, `retrievers.py` and `evaluate.py` once, top to bottom. Every function has a comment explaining the idea.
4. [ ] Put 3 to 5 of your own PDFs (about 100+ pages total) in `data/docs/`. Run `run_baselines.py` with no `--qa` file yet, only to confirm chunking works with no warnings.
5. [ ] Try different `--chunk-size` and `--overlap` on the chunk counts and write down what you notice.
6. [ ] Answer the five "check yourself" questions in your own words in a notes file.

Week 2 starts when you have your material in place: we build the real evaluation set together (write questions without copying phrases, label pages) and record honest baseline numbers.

## Interview questions this week prepares you for

- What is the difference between BM25 and embedding-based retrieval?
- Why chunk with overlap? How did you choose the chunk size?
- What is cosine similarity and why normalize vectors?
- What do Recall@k and MRR measure, and how did you build your test set?
- Why might a small sample score 100%, and why is that not a result?
