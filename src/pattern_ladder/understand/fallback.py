"""Rule-based query parsing, used when the model is unavailable or unusable.

This module matters more than the prompt does. The model is a free-tier
dependency with a rate limit, a network between us, and a non-zero chance of
returning something unparseable; the fallback is what decides whether that is a
degraded answer or no answer. It is deliberately deterministic and offline, so
the entire system remains testable and demonstrable with no key present.

The technique vocabulary is drawn from the phrasings a student uses when they
cannot name the technique -- which is the situation this project exists for.
"Shrink a window from the left" is the query; "sliding window" is the answer,
and no keyword index gets there on its own.
"""

from __future__ import annotations

import re

from .schema import SOURCE_FALLBACK, Intent

# Matching is by longest cue, so a specific phrase beats a generic one:
# "binary search tree" wins over "binary search", and "monotonic stack" over
# a bare "stack".
#
# Cues are deliberately phrases rather than single words. An earlier version
# listed "traversal" under tree, which read "matrix spiral traversal" as a
# tree problem -- a confidently wrong answer, which is worse than none, since
# the technique gets appended to the query and steers retrieval. Bare words
# appear only where they are unambiguous in this domain ("trie", "heap").
_TECHNIQUE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "sliding window",
        ("sliding window", "shrink a window", "shrink the window", "window from the left",
         "grow the window", "expand the window", "contiguous substring", "window of size",
         "longest substring", "subarray of length", "fixed size window"),
    ),
    (
        "two pointers",
        ("two pointer", "two pointers", "left and right pointer", "pointer from each end",
         "meet in the middle", "opposite ends", "fast and slow pointer", "tortoise and hare",
         "in place without extra space"),
    ),
    (
        "binary search",
        ("binary search", "search a sorted", "search in a sorted", "halve the search",
         "log n search", "guess and narrow", "monotonic predicate",
         "binary search on the answer", "rotated sorted"),
    ),
    (
        "dynamic programming",
        ("dynamic programming", "memoi", "memoize", "subproblem", "overlapping subproblem",
         "bottom up", "top down", "optimal substructure", "dp table",
         "longest common subsequence", "edit distance", "knapsack", "coin change",
         "how many ways", "minimum cost to", "maximum profit", "longest increasing"),
    ),
    (
        "backtracking",
        ("backtrack", "try all combinations", "undo the choice", "permutations of",
         "generate all subsets", "prune the search", "n queens", "sudoku"),
    ),
    (
        "depth-first search",
        ("depth first", " dfs ", "recurse into", "explore all paths", "flood fill",
         "connected region"),
    ),
    (
        "breadth-first search",
        ("breadth first", " bfs ", "level by level", "level order",
         "shortest number of steps", "shortest path in an unweighted", "word ladder",
         "fewest moves"),
    ),
    (
        "monotonic stack",
        ("monotonic stack", "next greater", "previous smaller", "stack that stays sorted",
         "next larger element", "next warmer", "largest rectangle"),
    ),
    (
        "stack",
        ("using a stack", "with a stack", "balanced parenthes", "balanced bracket",
         "matching bracket", "valid parenthes", "push and pop", "last in first out",
         "lifo"),
    ),
    (
        "queue",
        ("queue", "first in first out", "fifo", "deque", "circular buffer"),
    ),
    (
        "heap",
        ("heap", "priority queue", "k largest", "k smallest", "top k", "kth largest",
         "kth smallest", "median of a stream", "median from a stream",
         "median of a data stream", "data stream",
         "running median", "merge k sorted"),
    ),
    (
        "union find",
        ("union find", "disjoint set", "connected components", "merge groups",
         "number of provinces", "redundant connection"),
    ),
    (
        "prefix sum",
        ("prefix sum", "running total", "cumulative sum", "range sum",
         "sum of all subarrays", "subarray sum equals"),
    ),
    (
        "greedy",
        ("greedy", "locally optimal", "take the best each step", "as early as possible"),
    ),
    (
        "bit manipulation",
        ("bitmask", "bit manipulation", "xor", "set bits", "bitwise", "power of two",
         "single number"),
    ),
    (
        "graph",
        ("graph", "adjacency", "cycle detection", "shortest path", "dijkstra",
         "bellman ford", "minimum spanning tree", "kruskal", "prim"),
    ),
    (
        "topological sort",
        ("topological", "course schedule", "prerequisite", "dependency order",
         "build order", "indegree"),
    ),
    (
        "linked list",
        ("linked list", "reverse a list", "cycle in a list", "nodes in pairs",
         "middle of the list", "merge two sorted lists"),
    ),
    (
        "tree",
        ("binary tree", "binary search tree", "tree traversal", "inorder", "preorder",
         "postorder", "subtree", "leaf node", "lowest common ancestor",
         "serialize and deserialize", "depth of a tree", "root to leaf"),
    ),
    (
        "trie",
        ("trie", "prefix tree", "autocomplete", "word dictionary", "starts with"),
    ),
    (
        "interval",
        ("interval", "merge ranges", "overlapping ranges", "meeting room",
         "non overlapping", "sweep line"),
    ),
    (
        "sorting",
        ("sort the", "sorting", "merge sort", "quick sort", "quicksort", "quickselect",
         "counting sort", "bucket sort", "count inversions", "custom comparator"),
    ),
    (
        "matrix",
        ("matrix", "2d grid", "spiral order", "rotate an image", "rotate the image",
         "row and column", "transpose"),
    ),
    (
        "hash table",
        ("hash map", "hash table", "hash set", "count occurrences", "frequency map",
         "seen before", "anagram", "group by key", "duplicate"),
    ),
    (
        "string",
        ("string compression", "reverse the words", "palindrom", "substring search",
         "pattern matching", "kmp", "rolling hash"),
    ),
    (
        "math",
        ("modulo", "gcd", "greatest common divisor", "prime", "factorial",
         "number theory", "digits of", "base conversion"),
    ),
    (
        "randomised",
        ("reservoir sampling", "random pick", "shuffle an array", "randomly select"),
    ),
    (
        "simulation",
        ("simulate", "step by step until", "game of life", "robot moves"),
    ),
    (
        "design",
        ("design a", "implement a class", "lru cache", "lfu cache", "data structure that"),
    ),
)

_DIFFICULTY_CUES = {
    "Easy": ("easy", "beginner", "simple", "starter", "warm up", "warmup", "basic"),
    "Medium": ("medium", "intermediate", "moderate"),
    "Hard": ("hard", "difficult", "advanced", "challenging", "tough"),
}

# Phrases indicating the student wants one problem rather than a progression.
_SINGLE_CUES = (
    "just one", "a single", "one problem", "single problem", "only one",
    "give me a problem", "find the problem", "which problem",
)

_RAMP_CUES = (
    "ladder", "ramp", "progression", "path", "roadmap", "step by step",
    "work up", "build up", "series of", "practice plan", "from easy",
)


def _normalise(text: str) -> str:
    # Collapse punctuation to spaces so "window-from-the-left" matches the cue
    # "window from the left"; keep alphanumerics only.
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse(query: str) -> Intent:
    """Best-effort structured reading of a query, with no network access."""
    text = _normalise(query)

    # Score techniques by the length of the longest matching cue. A longer cue
    # is more specific, so "binary search tree" beats "binary search" and
    # "tree" wins the tie it should win.
    best_technique: str | None = None
    best_len = 0
    for technique, cues in _TECHNIQUE_CUES:
        for cue in cues:
            normalised_cue = _normalise(cue)
            if normalised_cue and normalised_cue in text and len(normalised_cue) > best_len:
                best_technique, best_len = technique, len(normalised_cue)

    difficulty: str | None = None
    for level, cues in _DIFFICULTY_CUES.items():
        if any(_normalise(cue) in text for cue in cues):
            difficulty = level
            break

    mode = "ramp"
    if any(_normalise(cue) in text for cue in _SINGLE_CUES):
        mode = "single"
    elif any(_normalise(cue) in text for cue in _RAMP_CUES):
        mode = "ramp"

    notes = [] if best_technique else ["no technique recognised from wording"]
    return Intent(
        technique=best_technique,
        difficulty=difficulty,
        mode=mode,
        source=SOURCE_FALLBACK,
        notes=notes,
    )
