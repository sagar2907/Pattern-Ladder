"""HTML -> plain text for LeetCode problem statements.

LeetCode descriptions are HTML fragments with a specific shape: prose in <p>,
worked examples in <pre>, identifiers in <code>, and a great many &nbsp;
entities. A generic tag-stripper mangles them in two ways that matter for
retrieval:

  1. Removing tags without inserting separators welds words together
     ("<p>a</p><p>b</p>" -> "ab"), creating tokens that match nothing.
  2. The &nbsp; entity (U+00A0) survives naive unescaping and is not treated
     as whitespace by every tokeniser, so "nums<U+00A0>and" can stay a single
     token that no query will ever match.

Both produce silent recall loss rather than an error, which is why this is a
deliberate parser and not a regex.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Tags after which a word boundary must exist. Anything block-level: welding
# across these is the bug described above.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "pre", "br", "li", "ul", "ol", "table", "tr", "td", "th",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "section", "hr",
    }
)

# Content of these is not prose and must not reach the index.
_DROP_CONTENT_TAGS = frozenset({"script", "style"})

_WHITESPACE_RUN = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._suppress_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:  # noqa: ARG002
        if tag in _DROP_CONTENT_TAGS:
            self._suppress_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_CONTENT_TAGS:
            self._suppress_depth = max(0, self._suppress_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._suppress_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def strip_html(fragment: str | None) -> str:
    """Convert a LeetCode HTML description to normalised plain text.

    Idempotent on text that contains no markup, so it is safe to apply to
    fields that may or may not be HTML.
    """
    if not fragment:
        return ""

    parser = _Extractor()
    # HTMLParser raises on some malformed input in strict mode; convert_charrefs
    # plus the default non-strict behaviour handles the real corpus, but a
    # single bad row must not fail an entire index build.
    try:
        parser.feed(fragment)
        parser.close()
        raw = parser.text()
    except Exception:  # noqa: BLE001 - degraded output beats a failed build
        raw = re.sub(r"<[^>]+>", " ", fragment)

    # U+00A0 and friends: normalise every Unicode space to a plain space before
    # collapsing runs, or "a\xa0b" survives as one token.
    raw = raw.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    raw = raw.replace("\u2019", "'").replace("\u2018", "'")
    raw = raw.replace("\u201c", '"').replace("\u201d", '"')

    lines = [_WHITESPACE_RUN.sub(" ", line).strip() for line in raw.split("\n")]
    text = "\n".join(line for line in lines if line)
    return _BLANK_LINES.sub("\n\n", text).strip()


def build_index_text(
    *,
    title: str,
    topics: list[str],
    description: str,
    title_repeat: int = 3,
) -> str:
    """Compose the single string that both retrievers index.

    `title_repeat` is a term-frequency thumb on the scale: BM25 has no notion
    of fields, so the only way to say "a title match is worth more than a body
    match" is to repeat the title. It also changes the dense embedding, since
    both retrievers index this same string.

    Three is the smallest value that puts the expected problem in the candidate
    pool for every query on the smoke set. Measured in
    scripts/sweep_retrieval.py: pool recall runs 0.950 at one repeat, 0.988 at
    two, and 1.000 at three and at five. Five buys nothing over three and
    dilutes the statement text further, so three it is.
    """
    if title_repeat < 1:
        raise ValueError("title_repeat must be >= 1")

    parts = [title] * title_repeat
    if topics:
        parts.append(", ".join(topics))
    parts.append(description)
    return "\n".join(p for p in parts if p)
