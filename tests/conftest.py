"""Shared fixtures.

Everything here is synthetic and offline. No test downloads a model, reads a
key, or touches the network: the suite has to pass on a fresh clone in CI with
no secrets configured, which is also the only way it can be trusted as a
regression gate rather than an integration smoke test.

The dense index is built from a hand-written matrix rather than a real encoder
for the same reason -- and because a hand-written matrix lets a test state
exactly which documents are near which query, instead of hoping a 22M-parameter
model agrees.
"""

from __future__ import annotations

import numpy as np
import pytest

from pattern_ladder.data import Problem
from pattern_ladder.index.dense import DenseIndex
from pattern_ladder.index.lexical import LexicalIndex
from pattern_ladder.text import build_index_text


def make_problem(
    slug: str,
    problem_id: int,
    title: str,
    difficulty: str = "Medium",
    acceptance: float = 50.0,
    topics: tuple[str, ...] = (),
    description: str = "",
    similar: tuple[str, ...] = (),
) -> Problem:
    description = description or f"Statement for {title}."
    return Problem(
        slug=slug,
        problem_id=problem_id,
        title=title,
        difficulty=difficulty,
        acceptance_rate=acceptance,
        topics=list(topics),
        description=description,
        index_text=build_index_text(
            title=title, topics=list(topics), description=description
        ),
        similar_slugs=list(similar),
        hints=[],
        url=f"https://leetcode.com/problems/{slug}/",
    )


@pytest.fixture
def problem_factory():
    """The `make_problem` helper, exposed as a fixture.

    Test modules cannot import from conftest directly (tests/ is not a
    package), and making it one only to share a helper would be a worse
    trade than handing it over as a fixture.
    """
    return make_problem


@pytest.fixture
def problems() -> list[Problem]:
    """A twelve-problem corpus with two clean clusters plus two strays.

    Cluster A (window/*) is chained by similar_slugs; cluster B (stack/*) is a
    separate chain. `lonely-*` have no links at all, standing in for the 38% of
    the real corpus that is isolated.
    """
    return [
        make_problem("window-easy", 1, "Maximum Average Subarray", "Easy", 70.0,
                     ("Array", "Sliding Window"),
                     "Find a contiguous window of fixed size with the largest average.",
                     ("window-medium",)),
        make_problem("window-medium", 2, "Longest Substring Without Repeating", "Medium", 55.0,
                     ("Hash Table", "String", "Sliding Window"),
                     "Shrink the window from the left when a character repeats.",
                     ("window-easy", "window-hard")),
        make_problem("window-hard", 3, "Minimum Window Substring", "Hard", 40.0,
                     ("Hash Table", "String", "Sliding Window"),
                     "Smallest window of the string covering every required character.",
                     ("window-medium", "window-extra")),
        make_problem("window-extra", 4, "Permutation in String", "Medium", 47.0,
                     ("Hash Table", "Sliding Window", "Two Pointers"),
                     "Fixed size window comparing character counts.",
                     ("window-hard",)),
        make_problem("window-fifth", 5, "Longest Repeating Character Replacement", "Medium", 54.0,
                     ("Hash Table", "String", "Sliding Window"),
                     "Grow and shrink a window under a replacement budget.",
                     ("window-medium",)),
        make_problem("stack-easy", 10, "Valid Parentheses", "Easy", 42.0,
                     ("Stack", "String"),
                     "Push opening brackets and pop on the matching close.",
                     ("stack-medium",)),
        make_problem("stack-medium", 11, "Next Greater Element", "Medium", 66.0,
                     ("Stack", "Monotonic Stack", "Array"),
                     "Maintain a stack that stays sorted to find the next larger value.",
                     ("stack-easy", "stack-hard")),
        make_problem("stack-hard", 12, "Largest Rectangle in Histogram", "Hard", 45.0,
                     ("Stack", "Monotonic Stack", "Array"),
                     "Monotonic stack over bar heights.",
                     ("stack-medium", "stack-extra")),
        make_problem("stack-extra", 13, "Daily Temperatures", "Medium", 67.0,
                     ("Stack", "Monotonic Stack", "Array"),
                     "For each day find the next warmer day using a stack.",
                     ("stack-hard",)),
        make_problem("stack-fifth", 14, "Min Stack", "Medium", 56.0,
                     ("Stack", "Design"),
                     "Design a stack that reports its minimum in constant time.",
                     ("stack-easy",)),
        make_problem("lonely-array", 20, "Running Sum of Array", "Easy", 87.0,
                     ("Array", "Prefix Sum"),
                     "Cumulative totals of an array."),
        make_problem("lonely-math", 21, "Convert the Temperature", "Easy", 89.0,
                     ("Math",),
                     "Arithmetic conversion between units."),
    ]


@pytest.fixture
def embeddings(problems) -> np.ndarray:
    """Hand-built unit vectors: window problems on one axis, stack on another.

    `lonely-array` is placed near the window cluster and shares the Array tag
    with it, so the shared-tag backfill has something legitimate to attach it
    to; `lonely-math` is placed away from both and shares no tag, so it must
    stay isolated.
    """
    directions = {
        "window": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "stack": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "lonely-array": np.array([0.94, 0.0, 0.34], dtype=np.float32),
        "lonely-math": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    rows = []
    for problem in problems:
        if problem.slug in directions:
            vector = directions[problem.slug]
        elif problem.slug.startswith("window"):
            vector = directions["window"]
        else:
            vector = directions["stack"]
        # Nudge each vector so no two rows are exactly identical, which would
        # make nearest-neighbour ties order-dependent and the tests flaky.
        jitter = np.array([0.0, 0.0, 0.01 * problem.problem_id], dtype=np.float32)
        vector = vector + jitter
        rows.append(vector / np.linalg.norm(vector))
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


def fake_encoder(texts: list[str]) -> np.ndarray:
    """A three-axis stand-in for the real sentence encoder.

    Maps text onto the same axes the fixture matrix uses, by keyword. Crude by
    design: the point of these tests is the pipeline's composition, and a real
    encoder would make them slow, network-dependent, and dependent on a model's
    opinion rather than on stated relationships.
    """
    window_words = ("window", "shrink", "substring", "contiguous", "subarray")
    stack_words = ("stack", "parenthes", "bracket", "greater", "monotonic", "histogram")
    rows = []
    for text in texts:
        lowered = text.lower()
        window = sum(word in lowered for word in window_words)
        stack = sum(word in lowered for word in stack_words)
        vector = np.array([float(window), float(stack), 0.05], dtype=np.float32)
        if not vector[:2].any():
            vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rows.append(vector / np.linalg.norm(vector))
    return np.vstack(rows).astype(np.float32)


@pytest.fixture
def dense_index(embeddings) -> DenseIndex:
    return DenseIndex(embeddings, encoder=fake_encoder)


@pytest.fixture
def lexical_index(problems) -> LexicalIndex:
    return LexicalIndex.build([p.index_text for p in problems])


@pytest.fixture(autouse=True)
def isolated_parse_cache(tmp_path, monkeypatch):
    """Point the query-reading cache at a throwaway file for every test.

    Autouse and unconditional. The cache is a module-level singleton backed by
    a real file under artifacts/, so without this a test that exercises query
    understanding would read whatever a previous *live* run happened to leave
    there -- which is exactly how a passing test can depend on a developer's
    machine. It showed up immediately: adding the cache broke a test that
    monkeypatches the model call, because the reading was already on disk and
    the patched call never ran.
    """
    from pattern_ladder.understand import cache as cache_module

    monkeypatch.setattr(
        cache_module,
        "_DEFAULT",
        cache_module.ParseCache(tmp_path / "parse_cache.json"),
    )
