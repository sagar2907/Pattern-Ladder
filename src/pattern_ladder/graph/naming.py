"""Optional model-written descriptions for discovered families.

A family's name is the headline of every answer, and the tag-derived name is
honest but taxonomic: "Subarray / Sliding Window / Prefix Sum" tells a student
which shelf the problems sit on, not what the shared idea is. A one-line
description -- "a window that grows and shrinks to keep a running condition
true" -- is what actually teaches.

Three properties make this safe to add:

* It is **offline**. Descriptions are written once during the index build and
  cached in families.json, so no query ever waits on a model.
* It is **additive**. The deterministic tag-derived `name` is always present
  and unchanged; the description is a separate optional field. Without an API
  key the system behaves exactly as before, and every measurement that scores
  against family names still scores against the same strings.
* It is **grounded**. The model is shown the family's actual member titles and
  tags and asked to describe what they share. It is not asked to recall
  anything about LeetCode, and a description that fails validation is dropped
  rather than repaired.
"""

from __future__ import annotations

import time

from .. import config
from ..data import Problem
from .families import Family

# How many member titles the model sees. Enough to show the shared idea across
# difficulty tiers, few enough to keep the prompt cheap -- this runs once per
# family and the free tier's binding limit is tokens per minute.
TITLE_SAMPLE = 12

# A description longer than this is a paragraph, not a name, and will not fit
# where the name is displayed.
MAX_DESCRIPTION_CHARS = 90

SYSTEM_PROMPT = "\n".join(
    [
        "You name recurring patterns in programming problems.",
        "You are given the titles and topic tags of problems that were found to"
        " cluster together. Describe the technique they share, in one short"
        " phrase a student would recognise.",
        "Rules:",
        "- Under 12 words. No trailing period.",
        "- Describe the *mechanic*, not the topic. Prefer 'a window that shrinks"
        " while a condition holds' over 'sliding window problems'.",
        "- Do not invent problems, and do not mention any problem by name.",
        "- If the problems have no single shared technique, reply exactly: MIXED",
        "Reply with the phrase alone and nothing else.",
    ]
)


def _prompt_for(family: Family, by_slug: dict[str, Problem]) -> str:
    members = [by_slug[s] for s in family.members if s in by_slug]
    # Spread the sample across the ladder rather than taking the first N, which
    # would show only the easiest members and describe the family by its
    # gentlest cases.
    if len(members) > TITLE_SAMPLE:
        step = len(members) / TITLE_SAMPLE
        members = [members[int(i * step)] for i in range(TITLE_SAMPLE)]

    titles = "\n".join(f"- {m.title}" for m in members)
    return f"Topic tags: {', '.join(family.tags) or 'none'}\nProblems:\n{titles}"


# Typographic characters models emit freely, mapped to the ASCII a reader and a
# PDF renderer can both handle. A non-breaking hyphen looks identical to a
# hyphen on screen and is a different codepoint everywhere else -- it survived
# into a description as "two<U+2011>pointer", which would search, sort and
# render subtly differently from "two-pointer" forever after.
_PUNCTUATION_FIXES = {
    "‑": "-", "‐": "-", "‒": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "…": "...", " ": " ", " ": " ", " ": " ",
}


def normalise_description(text: str) -> str | None:
    """Validate a model response into a usable description, or reject it.

    Model output is the only text in this system not written by hand or taken
    verbatim from the dataset, so it is normalised on the way in rather than
    trusted. Idempotent, so it can be reapplied to already-stored descriptions.
    """
    if not text:
        return None

    # Structural checks run on the *raw* response, before whitespace is
    # collapsed and before punctuation is normalised. Order matters in both
    # directions and both were wrong at first: collapsing whitespace first made
    # the newline check unreachable, and normalising first turned every em-dash
    # into "- ", which the list-marker check then rejected -- so any
    # description containing an em-dash was silently thrown away.
    raw = text.strip()
    if "\n" in raw or raw.startswith(("-", "*", "1.", "{")) or "}" in raw:
        return None

    for unusual, plain in _PUNCTUATION_FIXES.items():
        text = text.replace(unusual, plain)
    description = " ".join(text.strip().split())
    # Models wrap short answers in quotes surprisingly often.
    description = description.strip('"').strip("'").rstrip(".").strip()

    if not description or description.upper() == "MIXED":
        return None
    if len(description) > MAX_DESCRIPTION_CHARS:
        return None
    # Anything still outside ASCII is not something this pipeline can promise
    # to render, and a family name is not the place to discover that.
    if not description.isascii():
        return None
    return description


def describe_family(family: Family, by_slug: dict[str, Problem], *, client=None) -> str | None:
    """Return a one-line description for a family, or None if unavailable.

    Never raises. Any failure -- no key, rate limit, network, an unusable
    response -- leaves the family with its deterministic name and nothing else.
    """
    from ..understand import groq_client

    if client is None and not groq_client.available():
        return None

    try:
        active = client or groq_client._client()
        completion = active.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _prompt_for(family, by_slug)},
            ],
            temperature=0.0,
            max_tokens=config.GROQ_MAX_OUTPUT_TOKENS,
            reasoning_effort="low",
        )
        return normalise_description(completion.choices[0].message.content or "")
    except Exception:  # noqa: BLE001 - a naming failure must never fail a build
        return None


def describe_all(
    families: list[Family],
    problems: list[Problem],
    *,
    throttle_seconds: float = config.NAMING_THROTTLE_SECONDS,
    progress=None,
) -> dict[int, str]:
    """Describe every family. Returns {family_id: description} for successes.

    Sequential and throttled rather than concurrent: the free tier allows 30
    requests a minute, and a build step that trips the rate limit and falls
    back for half the corpus is worse than one that takes a few minutes.
    """
    from ..understand import groq_client

    if not groq_client.available():
        return {}

    try:
        client = groq_client._client()
    except Exception:  # noqa: BLE001
        return {}

    by_slug = {p.slug: p for p in problems}
    descriptions: dict[int, str] = {}
    for index, family in enumerate(families):
        description = describe_family(family, by_slug, client=client)
        if description:
            descriptions[family.family_id] = description
        if progress is not None:
            progress(index + 1, len(families), family, description)
        if throttle_seconds and index + 1 < len(families):
            time.sleep(throttle_seconds)
    return descriptions
