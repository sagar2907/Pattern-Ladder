"""Corpus acquisition and normalisation.

Turns the upstream LeetCode dump into a clean, ordered, deterministic list of
`Problem` records. Everything downstream (indexes, graph, ladders) is keyed by
`slug`, so this module owns the decision about what counts as a problem.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from . import config
from .text import build_index_text, strip_html

_DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

# Strength of the prior used when smoothing the like/dislike ratio, in
# notional votes, and the value it pulls toward. The corpus median approval
# is 0.92 and the median problem carries about 1,200 votes, so a prior worth
# 50 votes barely moves a well-established problem while keeping a
# thinly-voted one from reaching either extreme on almost no evidence.
PRIOR_VOTES = 50
PRIOR_APPROVAL = 0.92


@dataclass(frozen=True)
class Problem:
    """One retrievable problem.

    `index_text` is precomputed and stored rather than derived at query time:
    it is the exact string both retrievers saw, so a debugging session can
    compare a query against what was actually indexed instead of against a
    reconstruction that might differ.
    """

    slug: str
    problem_id: int
    title: str
    difficulty: str
    acceptance_rate: float
    topics: list[str]
    description: str
    index_text: str
    similar_slugs: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    likes: int = 0
    dislikes: int = 0
    url: str = ""

    @property
    def difficulty_rank(self) -> int:
        """0/1/2 for Easy/Medium/Hard. Unknown difficulty sorts last."""
        return _DIFFICULTY_ORDER.get(self.difficulty, 3)

    @property
    def approval(self) -> float:
        """How well regarded the problem is, from 0 to 1, smoothed toward the
        corpus median.

        Acceptance rate says how often a problem is *solved*; approval says
        whether people think it was worth solving. They are close to
        independent -- the correlation across the corpus is +0.14 -- so this
        carries information the acceptance rate does not. "Design an Ordered
        Stream" is accepted 82% of the time and approved by 13% of 4,115
        voters, and an ordering that knows only about acceptance puts it first.

        The smoothing matters as much as the ratio. A problem with three likes
        and one dislike is not better regarded than one with 40,000 likes and
        4,000 dislikes; it simply has less evidence. Blending toward the corpus
        median with a prior worth PRIOR_VOTES votes makes a sparse record say
        "unremarkable" rather than shouting.
        """
        total = self.likes + self.dislikes
        return (self.likes + PRIOR_VOTES * PRIOR_APPROVAL) / (total + PRIOR_VOTES)


def download_raw(dest: Path, *, url: str = config.DATASET_URL, force: bool = False) -> Path:
    """Fetch the upstream JSON dump, skipping the download if already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest

    response = requests.get(url, timeout=180, stream=True)
    response.raise_for_status()

    # Write to a temporary sibling and rename, so an interrupted download can
    # never leave a truncated file that a later run treats as complete.
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 16):
            handle.write(chunk)
    tmp.replace(dest)
    return dest


def _parse_similar(raw: str | None) -> list[str]:
    """similar_questions ships as a JSON *string*, not a nested array."""
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(entries, list):
        return []
    slugs = [e.get("titleSlug") for e in entries if isinstance(e, dict)]
    return [s for s in slugs if isinstance(s, str) and s]


def normalise(records: list[dict], *, title_repeat: int = 3) -> list[Problem]:
    """Filter and clean raw rows into the retrievable corpus.

    Two exclusions, both load-bearing:

    * `paidOnly` problems ship with an empty `description`. Keeping them would
      put ~700 documents with no text into the index, where they can never be
      retrieved but still shift BM25's average document length and therefore
      every other document's score.
    * Rows whose description is empty after HTML stripping, for the same
      reason.

    Ordering is by numeric problem id so that a rebuild produces byte-identical
    output; several downstream artefacts (embeddings matrix rows, BM25 doc ids)
    are positional and would silently permute otherwise.
    """
    problems: list[Problem] = []
    for row in records:
        if row.get("paidOnly"):
            continue

        description = strip_html(row.get("description"))
        if not description:
            continue

        slug = row.get("titleSlug") or ""
        if not slug:
            continue

        try:
            problem_id = int(row.get("frontendQuestionId") or 0)
        except (TypeError, ValueError):
            problem_id = 0

        topics = [t for t in (row.get("topics") or []) if isinstance(t, str)]
        title = (row.get("title") or slug).strip()

        problems.append(
            Problem(
                slug=slug,
                problem_id=problem_id,
                title=title,
                difficulty=row.get("difficulty") or "Unknown",
                acceptance_rate=float(row.get("acceptance_rate") or 0.0),
                topics=topics,
                description=description,
                index_text=build_index_text(
                    title=title,
                    topics=topics,
                    description=description,
                    title_repeat=title_repeat,
                ),
                similar_slugs=_parse_similar(row.get("similar_questions")),
                hints=[strip_html(h) for h in (row.get("hints") or []) if isinstance(h, str)],
                likes=int(row.get("likes") or 0),
                dislikes=int(row.get("dislikes") or 0),
                url=row.get("url") or f"https://leetcode.com/problems/{slug}/",
            )
        )

    problems.sort(key=lambda p: (p.problem_id, p.slug))

    # Slugs must be unique, because they are the key every other structure is
    # built on: slug -> problem, slug -> family, and slug -> embedding row. A
    # duplicate does not raise; the dictionaries silently keep the last entry,
    # so one of the twins becomes unreachable while the other is served under a
    # row index that belongs to its sibling. The upstream file happens to have
    # no duplicates today, which is exactly why this needs enforcing rather
    # than assuming -- nothing would announce it if that changed.
    deduplicated: list[Problem] = []
    seen_slugs: set[str] = set()
    for problem in problems:
        if problem.slug in seen_slugs:
            continue
        seen_slugs.add(problem.slug)
        deduplicated.append(problem)
    problems = deduplicated

    # Drop similar-question links whose target is not in the corpus (paid or
    # otherwise excluded), and any link a problem makes to itself. Dangling
    # links would create graph nodes with no text, which surface in ladders as
    # un-openable entries; self-links are simply meaningless and would let a
    # problem be presented as preparation for itself.
    known = set(seen_slugs)
    return [
        Problem(
            **{
                **asdict(p),
                "similar_slugs": [
                    s for s in p.similar_slugs if s in known and s != p.slug
                ],
            }
        )
        for p in problems
    ]


def save_corpus(problems: list[Problem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(p) for p in problems]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=None), encoding="utf-8")


def load_corpus(path: Path) -> list[Problem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Problem(**row) for row in payload]


def build_corpus(paths: config.Paths | None = None, *, title_repeat: int = 3) -> list[Problem]:
    """Download (if needed), normalise, cache, and return the corpus."""
    paths = paths or config.default_paths()
    raw_path = paths.raw / "leetcode_problems.json"
    download_raw(raw_path)
    records = json.loads(raw_path.read_text(encoding="utf-8"))
    problems = normalise(records, title_repeat=title_repeat)
    save_corpus(problems, paths.corpus)
    return problems
