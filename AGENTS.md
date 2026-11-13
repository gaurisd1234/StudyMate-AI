# Repository Guidelines

## Project Structure & Module Organization

The application is a small Python package under `src/studymate/`: `ingest.py` reads PDF, text, and Markdown files into page-aware chunks; `retrievers.py` implements BM25 and dense search; and `evaluate.py` measures retrieval quality. CLI entry points live in `scripts/` (`make_sample_data.py` and `run_baselines.py`), and `tests/test_pipeline.py` covers ingestion, retrieval plumbing, and metrics. Put source documents in `data/docs/`, evaluation CSVs in `data/eval/`, and generated sample inputs in `data/sample/`. Runtime reports go to `results/`; embedding cache files go to `.cache/`.

## Build, Test, and Development Commands

On Windows PowerShell, create and activate an environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Generate the sample PDF and run both baselines with `python scripts/make_sample_data.py` followed by `python scripts/run_baselines.py --docs data/sample --qa data/sample/qa.csv`. Use `--dense-model none` to run BM25 without downloading an embedding model. Run correctness checks with `python -m unittest discover -s tests -v`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, type hints for public functions, descriptive `snake_case` names, and short module or function docstrings that explain behavior. Keep package code in `src/studymate/`; keep runnable scripts in `scripts/`. Prefer standard-library tools unless a dependency is needed by the retrieval pipeline.

## Testing Guidelines

Use `unittest`; name test files `test_*.py` and test methods `test_*`. Add focused coverage for behavior changes, including edge cases such as empty input and invalid chunk settings. The hashing embedder is for plumbing checks only; it is not a semantic retrieval benchmark.

## Commits and Pull Requests

This checkout has no Git history available to establish a commit-message convention. Use a concise imperative summary (for example, `Add empty-document handling`). PRs should describe the behavior change, list validation commands and outcomes, link related issues when applicable, and include sample output or screenshots only when they clarify a user-visible change.

## Data and Configuration

Do not commit virtual environments, `.cache/` contents, or generated reports. Keep evaluation questions in the documented `question,source,page` CSV format, and do not present tiny sample-corpus scores as meaningful retrieval results.
