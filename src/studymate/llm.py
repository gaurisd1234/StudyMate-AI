"""LLM wrapper for Google Gemini with rate-limit retries and disk caching.

Uses the official google-genai SDK, loads GEMINI_API_KEY from .env (python-dotenv),
and defaults to the fast gemini-2.5-flash model.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"

DEFAULT_MODEL = "gemini-2.5-flash"


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


class FakeLLM:
    """Mock LLM for unit tests and offline testing."""

    def __init__(
        self,
        responses: list[str] | dict[str, str] | None = None,
        default_response: str | None = None,
    ):
        self.responses = responses or []
        self.response_dict = responses if isinstance(responses, dict) else {}
        self.default_response = default_response
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        self._call_count += 1

        if self.response_dict:
            for key, val in self.response_dict.items():
                if key in prompt:
                    return val

        if isinstance(self.responses, list) and self.responses:
            idx = (self._call_count - 1) % len(self.responses)
            return self.responses[idx]

        if self.default_response is not None:
            return self.default_response

        # Impartial evaluation judge prompt
        if "Answer with ONLY the number 1 or 0:" in prompt:
            return "1"

        # Answer generation prompt from context passages
        if "Context Passages:" in prompt and "Question:" in prompt:
            import re
            citations = re.findall(r"Source:\s+\[([\w\.-]+\s+p\.\d+)\]", prompt)
            passage_match = re.search(r"\[1\]\s+Source:\s+\[.*?\]\s*\n(.*?)(?=\n\n\[\d+\]|\n\nQuestion:)", prompt, re.DOTALL)
            if citations and passage_match:
                lines = [l.strip() for l in passage_match.group(1).splitlines() if l.strip() and not l.strip().startswith("[")]
                body = " ".join(lines[:2]) if lines else "the mechanism is detailed in the lecture materials"
                if len(body) > 180:
                    body = body[:180] + "..."
                return f"Based on the provided materials, {body} [{citations[0]}]."
            elif citations:
                return f"Based on the provided context, the concept is addressed in [{citations[0]}]."
            else:
                return "I could not find this in the provided material."

        # Dynamic mock generation aware of question types and passage topics
        prompt_lower = prompt.lower()
        import re
        from collections import Counter
        from .retrievers import STOPWORDS

        if not hasattr(self, "_generated_questions"):
            self._generated_questions: set[str] = set()

        passage_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
        candidate_words: list[str] = []
        if passage_match:
            raw_passage = passage_match.group(1)
            candidate_words = [
                w.lower()
                for w in re.findall(r"\b[A-Za-z]{4,}\b", raw_passage)
                if w.lower() not in STOPWORDS
            ]

        # Filter out very generic verbs
        generic_words = {"make", "take", "give", "said", "also", "come", "will", "well", "done"}
        informative_words = [w for w in candidate_words if w not in generic_words] or candidate_words or ["methodology"]
        common = [w for w, _ in Counter(informative_words).most_common(15)]

        templates = {
            "definition": [
                "How is {t} conceptually defined and interpreted in this context?",
                "What is the precise definition of {t} as outlined in this material?",
                "Could you explain what {t} represents within this theoretical framework?",
            ],
            "why": [
                "Why is {t} considered essential for achieving the stated objectives?",
                "What rationale motivates the adoption of {t} over conventional strategies?",
                "Why does {t} play a critical role in the behavior of this system?",
            ],
            "how": [
                "In what manner does the procedure regarding {t} operate in practice?",
                "How is the mechanism for {t} carried out step by step?",
                "What process governs how {t} functions according to the text?",
            ],
            "compare": [
                "How does the methodology for {t} contrast with alternative strategies?",
                "What trade-offs distinguish {t} from other approaches discussed?",
                "In what ways does {t} diverge from standard techniques in this field?",
            ],
            "application": [
                "Under what specific circumstances would a practitioner deploy {t}?",
                "In which real-world scenario is {t} most effectively applied?",
                "When would an engineer choose to implement {t} to address practical constraints?",
            ],
        }

        q_cat = "definition"
        for cat in ["why", "how", "compare", "application"]:
            if cat in prompt_lower:
                q_cat = cat
                break

        chosen_q = ""
        # Try words and template variations until a completely unique question is found
        for w_offset in range(len(common)):
            topic = common[(self._call_count + w_offset) % len(common)]
            for tmpl in templates[q_cat]:
                cand = tmpl.format(t=topic)
                if cand not in self._generated_questions:
                    chosen_q = cand
                    break
            if chosen_q:
                break

        if not chosen_q:
            chosen_q = f"How does {topic} function in context item {self._call_count}?"

        self._generated_questions.add(chosen_q)
        return chosen_q


# Global override used during tests
_ACTIVE_LLM: LLMClient | None = None


def set_active_llm(llm: LLMClient | None) -> None:
    """Override the active LLM client (e.g. with a FakeLLM for testing)."""
    global _ACTIVE_LLM
    _ACTIVE_LLM = llm


def _get_cache() -> dict[str, str]:
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp_path.replace(CACHE_FILE)


def _compute_cache_key(prompt: str, system: str | None, model: str) -> str:
    payload = f"model:{model}\nsystem:{system or ''}\nprompt:{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate(
    prompt: str,
    system: str | None = None,
    client: LLMClient | None = None,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> str:
    """Generate text from a prompt using Gemini (or FakeLLM), with caching and backoff.

    Args:
        prompt: User prompt content.
        system: Optional system instruction.
        client: Optional custom client (e.g. FakeLLM instance).
        max_retries: Max attempts for transient / rate-limit errors.
        initial_delay: Base backoff sleep time in seconds.
        backoff_factor: Multiplier for backoff on subsequent retries.

    Returns:
        Generated text string.
    """
    active_client = client or _ACTIVE_LLM
    if active_client is not None:
        return active_client.generate(prompt, system=system)

    model_name = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    cache_key = _compute_cache_key(prompt, system, model_name)

    # 1. Check disk cache
    cache = _get_cache()
    if cache_key in cache:
        return cache[cache_key]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set in environment or .env file.\n"
            "Please copy .env.example to .env and configure your GEMINI_API_KEY."
        )

    from google import genai
    from google.genai import errors, types

    genai_client = genai.Client(api_key=api_key)

    config = (
        types.GenerateContentConfig(system_instruction=system)
        if system
        else None
    )

    last_error: Exception | None = None
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()

            # Store in disk cache
            cache[cache_key] = text
            _save_cache(cache)
            return text

        except (errors.APIError, errors.ServerError, errors.ClientError) as error:
            last_error = error
            error_str = str(error).lower()
            status_code = getattr(error, "code", None)

            # Check if retryable (429 rate limit, 503 service unavailable, quota)
            is_rate_limit = (
                status_code in {429, 500, 503, 504}
                or "resource_exhausted" in error_str
                or "rate limit" in error_str
                or "quota" in error_str
            )
            if not is_rate_limit or attempt == max_retries:
                raise error

            jitter = random.uniform(0.1, 0.5)
            sleep_time = delay + jitter
            time.sleep(sleep_time)
            delay *= backoff_factor

        except Exception as error:
            raise error

    if last_error:
        raise last_error
    return ""
