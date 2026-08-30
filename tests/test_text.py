"""Tests for HTML extraction from problem statements."""

from __future__ import annotations

import pytest

from pattern_ladder.text import build_index_text, strip_html


def test_block_tags_become_word_boundaries():
    """Regression: stripping tags without separators welded words together.

    A naive `re.sub(r'<[^>]+>', '', html)` turns "<p>a</p><p>b</p>" into "ab",
    producing a token that matches no query. The failure is silent -- the
    document is still indexed, just unreachable.
    """
    assert strip_html("<p>alpha</p><p>beta</p>") == "alpha\nbeta"
    assert "alphabeta" not in strip_html("<p>alpha</p><p>beta</p>")


def test_non_breaking_space_becomes_a_real_space():
    """Regression: &nbsp; survived unescaping and welded adjacent words.

    LeetCode statements are full of "nums&nbsp;and". Left as U+00A0 the
    tokeniser can emit "numsand", which again is silently unmatchable.
    """
    out = strip_html("<p>integers <code>nums</code>&nbsp;and a target</p>")
    # Written as an escape rather than a literal: a bare U+00A0 in source is
    # invisible, and a reader would not be able to tell this assertion from
    # one about ordinary spaces.
    assert "\u00a0" not in out
    assert "nums and a target" in out


def test_script_and_style_content_is_dropped():
    out = strip_html("<p>keep</p><script>var x = 1;</script><style>.a{}</style>")
    assert "keep" in out
    assert "var x" not in out
    assert ".a{}" not in out


def test_entities_are_decoded():
    assert strip_html("<p>a &amp; b &lt; c</p>") == "a & b < c"


def test_empty_and_none_are_safe():
    assert strip_html(None) == ""
    assert strip_html("") == ""
    assert strip_html("   ") == ""


def test_plain_text_is_unchanged_and_idempotent():
    """Safe to apply to fields that may or may not contain markup."""
    text = "already plain text"
    assert strip_html(text) == text
    assert strip_html(strip_html(text)) == text


def test_malformed_html_degrades_rather_than_raising():
    """A single bad row must not fail an entire index build."""
    out = strip_html("<p>unclosed <b>bold <i>nested</p>")
    assert "unclosed" in out
    assert "bold" in out


def test_whitespace_runs_collapse_but_paragraphs_survive():
    out = strip_html("<p>a     b</p>\n\n\n<p>c</p>")
    assert "a b" in out
    assert out.count("\n") <= 2


def test_build_index_text_repeats_title():
    """Title repetition is how a title match outweighs a body match in BM25,
    which has no concept of fields."""
    text = build_index_text(title="Two Sum", topics=["Array"], description="body")
    assert text.count("Two Sum") == 3
    assert "Array" in text
    assert "body" in text


def test_build_index_text_rejects_zero_repeat():
    """Zero would drop the title from the index entirely."""
    with pytest.raises(ValueError):
        build_index_text(title="T", topics=[], description="d", title_repeat=0)


def test_build_index_text_omits_empty_topics():
    text = build_index_text(title="T", topics=[], description="d", title_repeat=1)
    assert text == "T\nd"
