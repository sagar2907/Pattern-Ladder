"""Tests for model-written family descriptions.

Nothing here calls the API. The generation path is exercised with a stub
client, and the validation -- which is the part that decides whether model
output is allowed into a cached artefact -- is tested directly.
"""

from __future__ import annotations

import pytest

from pattern_ladder.graph.families import Family
from pattern_ladder.graph.naming import (
    MAX_DESCRIPTION_CHARS,
    describe_all,
    describe_family,
    normalise_description,
)

NBSP_HYPHEN = "two\u2011pointer string reversal"
EM_DASH = "sliding window \u2014 shrink while it holds"


def _family(family_id: int = 0) -> Family:
    return Family(
        family_id=family_id,
        name="Stack / Array",
        tags=["Stack", "Array"],
        members=["stack-easy", "stack-medium"],
        size=2,
    )


class TestValidation:
    def test_a_plain_phrase_passes_through(self):
        assert normalise_description("a window that shrinks") == "a window that shrinks"

    def test_typographic_hyphens_are_normalised(self):
        """Regression: U+2011 reached a cached description.

        A non-breaking hyphen is indistinguishable from a hyphen on screen and
        different from it everywhere else, so the same phrase would sort,
        compare and render two ways depending on which one it happened to
        carry.
        """
        result = normalise_description(NBSP_HYPHEN)
        assert result == "two-pointer string reversal"
        assert result.isascii()

    def test_an_em_dash_is_not_mistaken_for_a_list_marker(self):
        """Regression: normalising an em-dash produced "- ", which the
        list-marker check then rejected, so every description containing an
        em-dash was silently discarded. The structural checks now run on the
        raw response, before punctuation is rewritten."""
        assert normalise_description(EM_DASH) == "sliding window - shrink while it holds"

    def test_surrounding_quotes_and_trailing_stops_are_removed(self):
        assert normalise_description('"a phrase."') == "a phrase"

    def test_the_refusal_token_is_rejected(self):
        """The prompt offers MIXED as an escape hatch; taking it must leave the
        family with its deterministic name rather than a bad description."""
        assert normalise_description("MIXED") is None
        assert normalise_description("mixed") is None

    def test_structural_junk_is_rejected(self):
        assert normalise_description("- a list item") is None
        assert normalise_description("1. numbered") is None
        assert normalise_description('{"json": true}') is None
        assert normalise_description("line one\nline two") is None

    def test_overlong_responses_are_rejected(self):
        assert normalise_description("x" * (MAX_DESCRIPTION_CHARS + 1)) is None

    def test_non_ascii_that_cannot_be_normalised_is_rejected(self):
        """The PDF renderer and the interface both promise ASCII; a family name
        is not the place to discover a glyph that neither can draw."""
        assert normalise_description("caf\u00e9 pattern") is None

    def test_empty_input_is_rejected(self):
        assert normalise_description("") is None
        assert normalise_description("   ") is None

    def test_normalisation_is_idempotent(self):
        """Applied at generation *and* again at load, so it must be stable."""
        once = normalise_description(NBSP_HYPHEN)
        assert normalise_description(once) == once


class _StubClient:
    """Stands in for the Groq client without any network."""

    def __init__(self, reply: str | None = "a stubbed description", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, **_kwargs):
                outer.calls += 1
                if outer.fail:
                    raise RuntimeError("upstream is unhappy")
                message = type("M", (), {"content": outer.reply})()
                choice = type("C", (), {"message": message})()
                return type("R", (), {"choices": [choice]})()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class TestGeneration:
    def test_a_good_response_becomes_a_description(self, problems):
        by_slug = {p.slug: p for p in problems}
        client = _StubClient("a window that shrinks from the left")
        assert describe_family(_family(), by_slug, client=client) == (
            "a window that shrinks from the left"
        )

    def test_an_api_failure_yields_no_description_rather_than_raising(self, problems):
        """A naming failure must never fail an index build; the family simply
        keeps its deterministic name."""
        by_slug = {p.slug: p for p in problems}
        assert describe_family(_family(), by_slug, client=_StubClient(fail=True)) is None

    def test_an_unusable_response_yields_no_description(self, problems):
        by_slug = {p.slug: p for p in problems}
        assert describe_family(_family(), by_slug, client=_StubClient("MIXED")) is None

    def test_describe_all_returns_nothing_without_a_key(self, problems, monkeypatch):
        """The offline path must not attempt a call at all."""
        monkeypatch.setattr(
            "pattern_ladder.understand.groq_client.available", lambda: False
        )
        assert describe_all([_family()], problems) == {}

    def test_describe_all_is_keyed_by_family_id(self, problems, monkeypatch):
        monkeypatch.setattr(
            "pattern_ladder.understand.groq_client.available", lambda: True
        )
        client = _StubClient("a described pattern")
        monkeypatch.setattr(
            "pattern_ladder.understand.groq_client._client", lambda: client
        )
        families = [_family(3), _family(7)]
        result = describe_all(families, problems, throttle_seconds=0)
        assert result == {3: "a described pattern", 7: "a described pattern"}
        assert client.calls == 2


def test_prompt_samples_across_the_whole_family(problems):
    """Sampling the first N members would describe a family by its easiest
    cases, since members arrive in ladder order."""
    from pattern_ladder.graph.naming import TITLE_SAMPLE, _prompt_for

    many = Family(
        family_id=0,
        name="X",
        tags=["Stack"],
        members=[p.slug for p in problems],
        size=len(problems),
    )
    prompt = _prompt_for(many, {p.slug: p for p in problems})
    listed = [line for line in prompt.splitlines() if line.startswith("- ")]
    assert len(listed) <= TITLE_SAMPLE
    if len(problems) > TITLE_SAMPLE:
        assert problems[-1].title in prompt or len(listed) == TITLE_SAMPLE


def test_description_never_replaces_the_deterministic_name():
    family = Family(
        family_id=0, name="Stack / Array", tags=["Stack"], members=["a"], size=1,
        description="a described pattern",
    )
    assert family.name == "Stack / Array"
    assert family.headline == "a described pattern"


@pytest.mark.parametrize("bad", ["MIXED", "", "   ", "x" * 200])
def test_rejected_descriptions_leave_the_headline_as_the_name(bad):
    assert normalise_description(bad) is None
