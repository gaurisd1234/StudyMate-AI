"""Step 2: the two baseline search methods.

1. BM25  - classic keyword search. Scores a chunk by how many of the question's
           words it contains, weighting rare words more and long chunks less.
2. Dense - "meaning" search. An embedding model turns text into a vector of
           numbers; chunks whose vectors point in a similar direction to the
           question's vector are considered relevant, even with different words.

Both expose the same method:  search(query, k) -> [(Chunk, score), ...]
so the evaluation code can treat them identically.
"""
from __future__ import annotations

import hashlib
import re
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from .ingest import Chunk

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on",
    "at", "to", "for", "and", "or", "it", "its", "this", "that", "with", "as",
    "by", "from", "do", "does", "did", "how", "what", "why", "which", "can",
    "i", "we", "you", "my", "me", "when", "than", "then", "so",
}


def tokenize(text: str) -> list[str]:
    """Lowercase words with stopwords removed. \\w keeps non-English letters too."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        if chunks:
            self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        else:
            self._bm25 = None

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


# ----------------------------------------------------------------------------
# Embedders: anything with `.name` and `.encode(list[str]) -> unit-length vectors`
# ----------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """Real semantic embeddings. Downloads the model on first use (about 90 MB)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SystemExit(
                "sentence-transformers is not installed. Run:\n"
                "    pip install -r requirements.txt\n"
                "or run with --dense-model none to use BM25 only."
            ) from error

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # unit length -> dot product == cosine similarity
            show_progress_bar=len(texts) > 200,
        )
        return np.asarray(vectors, dtype="float32")


class HashingEmbedder:
    """A toy embedder for TESTING THE PLUMBING ONLY. It is NOT semantic.

    It counts words into a fixed number of buckets, so it behaves like a crude
    keyword matcher. Never report results from it as "dense retrieval".
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder, cache_dir: str | Path | None = None):
        import faiss

        self.chunks = chunks
        self.embedder = embedder
        self.name = f"dense[{embedder.name.split('/')[-1]}]"
        vectors = self._embed_chunks(cache_dir)
        # IndexFlatIP = exact inner-product search. With unit vectors that equals
        # cosine similarity. Fine for thousands of chunks; no approximation needed.
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def _embed_chunks(self, cache_dir) -> np.ndarray:
        texts = [c.text for c in self.chunks]
        if cache_dir is None:
            return self.embedder.encode(texts)
        # Embedding is the slow step, so cache it. The key changes if the model
        # or ANY chunk text changes, so you never get stale vectors.
        key = hashlib.sha1(
            (self.embedder.name + "\n" + "\n".join(texts)).encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        cache_file = Path(cache_dir) / f"emb_{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        vectors = self.embedder.encode(texts)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, vectors)
        return vectors

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_vec = self.embedder.encode([query])
        scores, ids = self._index.search(query_vec, min(k, len(self.chunks)))
        return [
            (self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1
        ]


# ----------------------------------------------------------------------------
# Hybrid & Reranking Retrievers
# ----------------------------------------------------------------------------
class HybridRetriever:
    """Combines BM25 and Dense search using Reciprocal Rank Fusion (RRF) or Weighted Score Fusion.

    Concepts:
    - RRF (Reciprocal Rank Fusion): score(c) = sum(1 / (k + rank_m(c))). Rank-based; doesn't care
      about score calibrations across different models.
    - Min-Max Score Fusion: normalizes raw scores of both retrievers to [0, 1] then computes
      weighted sum: w * dense_norm + (1 - w) * bm25_norm.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        method: str = "rrf",  # "rrf" or "weighted"
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        candidate_pool: int = 100,
    ):
        from collections import defaultdict
        self._defaultdict = defaultdict

        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.chunks = bm25_retriever.chunks
        self.method = method.lower()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.candidate_pool = candidate_pool
        if self.method == "rrf":
            self.name = "hybrid-rrf"
        elif self.method == "weighted":
            self.name = f"hybrid-weighted[w={dense_weight:.1f}]"
        else:
            raise ValueError(f"Unknown fusion method '{method}'. Choose 'rrf' or 'weighted'.")

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        pool_size = min(len(self.chunks), max(k, self.candidate_pool))
        bm25_hits = self.bm25.search(query, k=pool_size)
        dense_hits = self.dense.search(query, k=pool_size)

        if self.method == "rrf":
            return self._fuse_rrf(bm25_hits, dense_hits, k)
        else:
            return self._fuse_weighted(bm25_hits, dense_hits, k)

    def _fuse_rrf(
        self,
        bm25_hits: list[tuple[Chunk, float]],
        dense_hits: list[tuple[Chunk, float]],
        k: int,
    ) -> list[tuple[Chunk, float]]:
        scores: dict[str, float] = self._defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(bm25_hits, start=1):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        for rank, (chunk, _) in enumerate(dense_hits, start=1):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(chunk_map[cid], score) for cid, score in ranked]

    def _fuse_weighted(
        self,
        bm25_hits: list[tuple[Chunk, float]],
        dense_hits: list[tuple[Chunk, float]],
        k: int,
    ) -> list[tuple[Chunk, float]]:
        chunk_map: dict[str, Chunk] = {}

        def min_max_normalize(hits: list[tuple[Chunk, float]]) -> dict[str, float]:
            if not hits:
                return {}
            vals = [score for _, score in hits]
            min_v, max_v = min(vals), max(vals)
            norm_map: dict[str, float] = {}
            for chunk, score in hits:
                chunk_map[chunk.chunk_id] = chunk
                if max_v > min_v:
                    norm_map[chunk.chunk_id] = (score - min_v) / (max_v - min_v)
                else:
                    norm_map[chunk.chunk_id] = 1.0
            return norm_map

        bm25_norm = min_max_normalize(bm25_hits)
        dense_norm = min_max_normalize(dense_hits)

        all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
        w_dense = self.dense_weight
        w_bm25 = 1.0 - w_dense

        fused_scores: dict[str, float] = {}
        for cid in all_ids:
            s_dense = dense_norm.get(cid, 0.0)
            s_bm25 = bm25_norm.get(cid, 0.0)
            fused_scores[cid] = (w_dense * s_dense) + (w_bm25 * s_bm25)

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(chunk_map[cid], score) for cid, score in ranked]


class RerankRetriever:
    """Second-stage reranker using a CrossEncoder on top candidates from a first-stage retriever.

    Concepts:
    - Bi-encoders encode query and passage independently into vectors (fast for large scale).
    - Cross-encoders pass query and passage together through joint transformer attention layers.
      They capture cross-attention interactions that bi-encoders miss, making them significantly
      more accurate for reranking a smaller pool of top candidates (e.g. 20).
    """

    def __init__(
        self,
        base_retriever: Retriever,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_candidates: int = 20,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise SystemExit("sentence-transformers is required for CrossEncoder reranking.") from e

        self.base_retriever = base_retriever
        self.top_k_candidates = top_k_candidates
        self.model_name = model_name
        self._model = CrossEncoder(model_name)
        self.name = f"rerank[{model_name.split('/')[-1]}]"

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        candidates = self.base_retriever.search(query, k=max(k, self.top_k_candidates))
        if not candidates:
            return []

        pairs = [(query, c.text) for c, _ in candidates]
        scores = self._model.predict(pairs)

        scored = list(zip([c for c, _ in candidates], [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
