"""Question answering module with grounded context, strict citations, and abstention.

Directs the LLM to answer strictly from numbered context passages, cite sources
as [file p.X], and return exactly "I could not find this in the provided material."
if the context is insufficient.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .ingest import Chunk
from .llm import LLMClient, generate
from .retrievers import Retriever

ABSTENTION_TEXT = "I could not find this in the provided material."

SYSTEM_PROMPT = (
    "You are an accurate, honest AI study assistant. "
    "Your objective is to answer student questions strictly and solely using the provided numbered study passages. "
    "Do not extrapolate, assume, or incorporate any external knowledge beyond what is explicitly stated in the context."
)


def build_context_block(chunks: list[Chunk]) -> str:
    """Format chunks as numbered context blocks with document and page citations."""
    blocks = []
    for idx, c in enumerate(chunks, start=1):
        blocks.append(f"[{idx}] Source: [{c.source} p.{c.page}]\n{c.text}")
    return "\n\n".join(blocks)


def build_answer_prompt(query: str, chunks: list[Chunk]) -> str:
    """Construct prompt enforcing grounding, exact citations, and strict abstention."""
    context_str = build_context_block(chunks)
    return (
        f"Context Passages:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"1. Answer the question ONLY from the numbered context passages above. Do NOT use outside knowledge.\n"
        f"2. Every claim or fact in your answer MUST cite its source as [filename p.X] (e.g. [lecture1.pdf p.3]).\n"
        f"3. If the provided context passages do not contain sufficient information to answer the question, reply with EXACTLY:\n"
        f'"{ABSTENTION_TEXT}"\n'
        f"4. Be concise, direct, and factual. Do not explain what is missing if you abstain.\n\n"
        f"Answer:"
    )


def extract_citations(text: str) -> list[tuple[str, int]]:
    """Extract all [file p.X] citations from text as [(source, page_int), ...]."""
    # Matches patterns like [notes.pdf p.12] or [doc.txt p.1]
    matches = re.findall(r"\[([\w\.-]+)\s+p\.(\d+)\]", text)
    citations: list[tuple[str, int]] = []
    for source, page_str in matches:
        try:
            citations.append((source.strip(), int(page_str)))
        except ValueError:
            continue
    return citations


@dataclass
class AnswerResult:
    query: str
    answer: str
    citations: list[str]
    cited_pages: list[tuple[str, int]]
    chunks: list[Chunk]
    is_abstention: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "cited_pages": self.cited_pages,
            "chunks": self.chunks,
            "scores": self.scores,
            "is_abstention": self.is_abstention,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
        }


def answer_question(
    query: str,
    retriever: Retriever,
    k: int = 5,
    client: LLMClient | None = None,
) -> AnswerResult:
    """Retrieve top-k chunks, format prompt, generate answer, and parse citations."""
    # 1. Retrieval stage
    t0 = time.perf_counter()
    ranked_chunks = retriever.search(query, k=k)
    chunks = [c for c, _ in ranked_chunks]
    scores = [float(s) for _, s in ranked_chunks]
    retrieval_latency_ms = (time.perf_counter() - t0) * 1000

    if not chunks:
        return AnswerResult(
            query=query,
            answer=ABSTENTION_TEXT,
            citations=[],
            cited_pages=[],
            chunks=[],
            scores=[],
            is_abstention=True,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=0.0,
        )

    # 2. Generation stage
    prompt = build_answer_prompt(query, chunks)
    t1 = time.perf_counter()
    raw_answer = generate(prompt, system=SYSTEM_PROMPT, client=client)
    generation_latency_ms = (time.perf_counter() - t1) * 1000

    clean_answer = raw_answer.strip()
    is_abstention = (
        clean_answer.lower() == ABSTENTION_TEXT.lower()
        or clean_answer.strip('"').strip("'").lower() == ABSTENTION_TEXT.lower()
        or ABSTENTION_TEXT.lower() in clean_answer.lower()
    )

    if is_abstention:
        clean_answer = ABSTENTION_TEXT
        cited_pages = []
        citations = []
    else:
        cited_pages = extract_citations(clean_answer)
        citations = [f"{s} p.{p}" for s, p in cited_pages]

    return AnswerResult(
        query=query,
        answer=clean_answer,
        citations=citations,
        cited_pages=cited_pages,
        chunks=chunks,
        scores=scores,
        is_abstention=is_abstention,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
    )

