"""Query understanding via the Groq API, with a guaranteed fallback.

Contract: `understand()` always returns an Intent. Network failure, missing
key, rate limit, malformed JSON, or a model that answers in prose all degrade
to the deterministic parser in `fallback.py`. Nothing in the call path is
allowed to raise into the request.
"""

from __future__ import annotations

import json
import os

from .. import config
from . import fallback
from .cache import default_cache
from .schema import SOURCE_MODEL, Intent, coerce

# Kept short on purpose. Every token here is spent on every query, and the free
# tier's binding limit is tokens per minute rather than requests per day.
SYSTEM_PROMPT = "\n".join(
    [
        "You classify a student's description of a coding problem they are stuck on.",
        "Reply with JSON only, matching exactly this shape:",
        '{"technique": string|null, "difficulty": "Easy"|"Medium"|"Hard"|null,'
        ' "mode": "ramp"|"single"}',
        "technique: the algorithmic technique the student is describing, in"
        ' lowercase (for example "sliding window", "binary search", "monotonic'
        ' stack"). Infer it from the mechanic they describe even when they never'
        " name it. Use null only if nothing in the text identifies a technique.",
        "difficulty: only if the student explicitly asked for a level. Otherwise"
        " null.",
        'mode: "single" ONLY if they explicitly ask for one problem. Default to'
        ' "ramp" -- a student describing repeated failure wants a progression,'
        " not one problem.",
    ]
)

# Both the technique and mode wordings above are the result of measuring the
# live model, not of drafting. The first version said only 'mode: "ramp" if they
# want a progression, "single" if they want one', and the model answered
# "single" to almost everything -- 2 of 5 correct on a hand-checked set,
# including "single" for "I keep failing problems where you shrink a window from
# the left", which is a description of repeated failure and the single clearest
# case for a progression. Naming the default explicitly took it to 5 of 5, and
# incidentally improved technique extraction ("stack" became "monotonic stack"),
# for about 12% more tokens.

_RESPONSE_FORMAT = {"type": "json_object"}


class GroqUnavailable(RuntimeError):
    """Raised internally when no usable client can be constructed."""


def _api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return key or None


def available() -> bool:
    """Whether a live call could even be attempted. Never makes one."""
    return _api_key() is not None


def _client():
    key = _api_key()
    if not key:
        raise GroqUnavailable("GROQ_API_KEY is not set")
    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise GroqUnavailable("groq package is not installed") from exc
    return Groq(api_key=key, timeout=config.GROQ_TIMEOUT_SECONDS)


def call_model(query: str, *, model: str = config.GROQ_MODEL) -> tuple[str, dict]:
    """Make the API call and return (raw_text, usage). Raises on any failure."""
    client = _client()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        # Greedy decoding, which makes the reading stable *almost* always --
        # and the qualifier is measured, not hedging. Repeating all twenty
        # evaluation queries three times each at this setting, nineteen were
        # bit-identical every time and one was not: "detect whether a linked
        # list has a cycle in it" came back as both "two pointers" and
        # "tortoise and hare".
        #
        # Temperature 0 is greedy sampling, not a reproducibility guarantee;
        # served models batch requests and route tokens in ways that can break
        # ties differently between calls. Worth stating because the system
        # shows its reading back to the student as though it were a fixed
        # property of their question, and once in a while it is not.
        temperature=0.0,
        max_tokens=config.GROQ_MAX_OUTPUT_TOKENS,
        response_format=_RESPONSE_FORMAT,
        # gpt-oss is a reasoning model: it emits reasoning tokens before the
        # answer, and they are billed and rate-limited like any other. Measured
        # over three queries, "low" costs 295 tokens against 488 at the default
        # and 558 at "high", runs faster (0.64s against 0.95s), *and* extracted
        # a technique the default missed. It is better on every axis measured,
        # which is unusual enough to be worth stating.
        reasoning_effort="low",
    )
    text = completion.choices[0].message.content or ""
    usage = {}
    if getattr(completion, "usage", None) is not None:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
    return text, usage


def _extract_json(text: str) -> object | None:
    """Parse JSON from a model response, tolerating surrounding prose.

    Even with a JSON response format enforced, a model can emit a fenced block
    or a leading sentence. Slicing between the first '{' and the last '}' is
    ugly but recovers the common failure without a dependency on the model
    behaving.
    """
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except ValueError:
        return None


def understand(
    query: str,
    *,
    model: str = config.GROQ_MODEL,
    allow_network: bool = True,
    cache=None,
) -> Intent:
    """Parse a query into an Intent, degrading rather than failing.

    `allow_network=False` forces the offline path, which is how the test suite
    exercises this function without a key or a connection.

    A successful model reading is cached, so the same question asked twice
    gives the same answer. Only model readings are cached: the rule-based
    parser is already deterministic and costs nothing, so storing its output
    would add a staleness risk for no benefit.
    """
    if not allow_network or not available():
        parsed = fallback.parse(query)
        reason = "no API key configured" if allow_network else "offline mode"
        return Intent(
            technique=parsed.technique,
            difficulty=parsed.difficulty,
            mode=parsed.mode,
            source=parsed.source,
            notes=[*parsed.notes, f"rule-based parse ({reason})"],
        )

    store = default_cache() if cache is None else cache
    cached = store.get(query, model=model, prompt=SYSTEM_PROMPT)
    if cached is not None:
        return cached

    try:
        text, _usage = call_model(query, model=model)
    except Exception as exc:  # noqa: BLE001 - any failure must degrade, not raise
        parsed = fallback.parse(query)
        return Intent(
            technique=parsed.technique,
            difficulty=parsed.difficulty,
            mode=parsed.mode,
            source=parsed.source,
            notes=[*parsed.notes, f"rule-based parse (model call failed: {type(exc).__name__})"],
        )

    payload = _extract_json(text)
    if payload is None:
        parsed = fallback.parse(query)
        return Intent(
            technique=parsed.technique,
            difficulty=parsed.difficulty,
            mode=parsed.mode,
            source=parsed.source,
            notes=[*parsed.notes, "rule-based parse (model returned unparseable output)"],
        )

    intent = coerce(payload, source=SOURCE_MODEL)

    # A model parse that recovered no technique is worse than the rule parse,
    # which at least matches known phrasings. Prefer the rule technique in that
    # case, keeping the model's other fields.
    if intent.technique is None:
        parsed = fallback.parse(query)
        if parsed.technique:
            intent = Intent(
                technique=parsed.technique,
                difficulty=intent.difficulty,
                mode=intent.mode,
                source=SOURCE_MODEL,
                notes=[*intent.notes, "technique recovered by rule-based parse"],
            )

    # Cached after the fallback blend above, so what is stored is the reading
    # actually used rather than the raw model response.
    store.put(query, intent, model=model, prompt=SYSTEM_PROMPT)
    return intent
