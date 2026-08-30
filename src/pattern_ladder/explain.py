"""Grounded explanations for why a result was returned.

Every sentence produced here is assembled from fields that were actually
retrieved: the family a problem belongs to, the difficulty and acceptance rate
in the corpus, and which retrieval arm surfaced it.

No sentence is *written* by a language model at query time. One phrase may have
been: a family's description, if the index was built with `--describe`, is
model-written -- offline, once, from that family's own member titles, and
cached. The claim being made about it here ("in family X") is a fact about
graph membership either way; only the family's label varies. The distinction
matters enough to state, because the whole point of this module is that it does
not manufacture reasons.

That is a deliberate reversal of the original design, which called an LLM a
second time to write a one-line justification per result. Two reasons it was
wrong. First, on the free tier the binding constraint is tokens per minute, not
requests per day, and a second call per query roughly doubles token spend for
the least load-bearing part of the answer. Second, and more seriously, a model
asked to justify a ranking it did not produce will produce a plausible reason
whether or not a real one exists -- and a confident false explanation of a
correct result is worse than no explanation, because it teaches the student a
relationship that is not there.
"""

from __future__ import annotations

from .data import Problem
from .graph.families import Family

# Which retrieval arms found a document is genuinely informative: a lexical-only
# hit means the query's literal words appear in the problem, a dense-only hit
# means the phrasing matched without shared vocabulary.
MATCH_BOTH = "both"
MATCH_LEXICAL = "lexical"
MATCH_DENSE = "dense"


def match_kind(in_lexical: bool, in_dense: bool) -> str:
    if in_lexical and in_dense:
        return MATCH_BOTH
    if in_lexical:
        return MATCH_LEXICAL
    return MATCH_DENSE


def _why_matched(kind: str) -> str:
    if kind == MATCH_BOTH:
        return "matched on both wording and meaning"
    if kind == MATCH_LEXICAL:
        return "uses your exact terms"
    return "matches the idea, not the words"


def explain(
    problem: Problem,
    *,
    kind: str,
    family: Family | None,
    anchor: Problem | None = None,
    shared_family_with_anchor: bool = False,
    technique: str | None = None,
) -> str:
    """One grounded sentence about this result.

    `anchor` is the top-ranked result, so later results can be described in
    relation to it -- that comparison is what turns a list into a path. The
    relation is only asserted when the two genuinely share a family.

    `technique` is what the student asked about, and it decides which of the
    family's two labels to name. A family holding more than one technique has a
    description that committed to one of them, so naming it unconditionally
    tells a student asking about sliding windows that they are looking at prefix
    sums -- once per result, in the sentence that is supposed to justify the
    recommendation. See Family.headline_for.
    """
    parts = [_why_matched(kind)]

    related = anchor is not None and anchor.slug != problem.slug

    label = family.headline_for(technique) if family is not None else None

    # A curated link is the strongest and most specific thing that can be said
    # about two problems, and it costs nothing to check -- the relationship is
    # already on the record, put there by a person. Preferring it over the
    # difficulty comparison below is the difference between "a step up from
    # Two Sum" (true of hundreds of problems) and "LeetCode lists this as
    # similar to Two Sum" (true of nineteen).
    if related and _directly_linked(problem, anchor):
        parts.append(f"listed as similar to {anchor.title}")
    elif family is not None:
        parts.append(f"in family '{label}' ({family.size} problems)")

    if related and _directly_linked(problem, anchor) and family is not None:
        parts.append(f"family '{label}'")

    if related and shared_family_with_anchor:
        # Only a difficulty or acceptance *gap* is worth stating; saying two
        # problems are equally hard is not a reason to solve one after the
        # other.
        if problem.difficulty_rank > anchor.difficulty_rank:
            parts.append(f"a step up from {anchor.title}")
        elif problem.difficulty_rank < anchor.difficulty_rank:
            parts.append(f"gentler entry than {anchor.title}")
        elif problem.acceptance_rate + 5.0 < anchor.acceptance_rate:
            parts.append(f"same tier as {anchor.title} but less often solved")

    parts.append(f"{problem.difficulty}, {problem.acceptance_rate:.0f}% accepted")
    return "; ".join(parts)


def _directly_linked(problem: Problem, other: Problem) -> bool:
    """Whether a curated similar-questions edge joins these two problems.

    Checked in both directions because the upstream lists are not reliably
    symmetric: A often names B without B naming A.
    """
    return other.slug in problem.similar_slugs or problem.slug in other.similar_slugs
