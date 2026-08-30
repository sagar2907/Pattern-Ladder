"""Tests for corpus normalisation and its exclusion rules."""

from __future__ import annotations

from pattern_ladder.data import Problem, normalise


def _row(**overrides) -> dict:
    row = {
        "titleSlug": "two-sum",
        "frontendQuestionId": "1",
        "title": "Two Sum",
        "difficulty": "Easy",
        "paidOnly": False,
        "description": "<p>Given an array of integers.</p>",
        "topics": ["Array", "Hash Table"],
        "hints": ["<p>Use a map.</p>"],
        "acceptance_rate": 55.5,
        "likes": 10,
        "dislikes": 1,
        "similar_questions": "[]",
        "url": "https://leetcode.com/problems/two-sum",
    }
    row.update(overrides)
    return row


def test_paid_problems_are_excluded():
    """Paid problems ship with an empty description.

    Keeping them would put ~700 textless documents in the index. They could
    never be retrieved, but they would still shift BM25's average document
    length, which changes the score of every other document in the corpus.
    """
    out = normalise([_row(), _row(titleSlug="paid", paidOnly=True, description="")])
    assert [p.slug for p in out] == ["two-sum"]


def test_problems_without_description_are_excluded():
    out = normalise([_row(), _row(titleSlug="empty", frontendQuestionId="2", description="   ")])
    assert [p.slug for p in out] == ["two-sum"]


def test_similar_questions_json_string_is_parsed():
    """The field is a JSON *string*, not a nested array."""
    rows = [
        _row(similar_questions='[{"title": "3Sum", "titleSlug": "3sum"}]'),
        _row(titleSlug="3sum", frontendQuestionId="15", title="3Sum"),
    ]
    out = normalise(rows)
    two_sum = next(p for p in out if p.slug == "two-sum")
    assert two_sum.similar_slugs == ["3sum"]


def test_links_to_excluded_problems_are_dropped():
    """A link to a paid problem would create a graph node with no text, which
    surfaces in a ladder as an entry the student cannot open."""
    rows = [
        _row(similar_questions='[{"titleSlug": "paid-one"}, {"titleSlug": "3sum"}]'),
        _row(titleSlug="3sum", frontendQuestionId="15", title="3Sum"),
        _row(titleSlug="paid-one", frontendQuestionId="99", paidOnly=True, description=""),
    ]
    out = normalise(rows)
    two_sum = next(p for p in out if p.slug == "two-sum")
    assert two_sum.similar_slugs == ["3sum"]


def test_malformed_similar_questions_does_not_raise():
    out = normalise([_row(similar_questions="not json at all")])
    assert out[0].similar_slugs == []


def test_missing_similar_questions_is_empty():
    out = normalise([_row(similar_questions=None)])
    assert out[0].similar_slugs == []


def test_ordering_is_by_problem_id_and_deterministic():
    """Embedding rows and BM25 doc ids are positional. If normalise() returned
    a different order between runs, every cached artefact would silently
    permute against the corpus."""
    rows = [
        _row(titleSlug="c", frontendQuestionId="30", title="C"),
        _row(titleSlug="a", frontendQuestionId="10", title="A"),
        _row(titleSlug="b", frontendQuestionId="20", title="B"),
    ]
    assert [p.problem_id for p in normalise(rows)] == [10, 20, 30]
    assert normalise(rows) == normalise(list(reversed(rows)))


def test_non_numeric_id_does_not_raise():
    out = normalise([_row(frontendQuestionId="not-a-number")])
    assert out[0].problem_id == 0


def test_html_is_stripped_from_description_and_hints():
    out = normalise([_row()])
    assert "<p>" not in out[0].description
    assert all("<p>" not in h for h in out[0].hints)


def test_difficulty_rank_orders_easy_medium_hard():
    ranks = [
        Problem(
            slug=d.lower(), problem_id=1, title=d, difficulty=d, acceptance_rate=0.0,
            topics=[], description="d", index_text="d",
        ).difficulty_rank
        for d in ("Easy", "Medium", "Hard")
    ]
    assert ranks == [0, 1, 2]


def test_unknown_difficulty_sorts_last():
    problem = Problem(
        slug="x", problem_id=1, title="X", difficulty="Unknown", acceptance_rate=0.0,
        topics=[], description="d", index_text="d",
    )
    assert problem.difficulty_rank == 3


def test_duplicate_slugs_are_collapsed():
    """A repeated slug silently misaligns every slug-keyed structure.

    The slug is the key for problem lookup, family membership, and the
    embedding row index. Two rows sharing one slug do not raise: the
    dictionaries keep whichever came last, so one twin becomes unreachable and
    the other is served under a row index belonging to its sibling. Every
    result stays plausible and one of them is wrong.
    """
    rows = [_row(), _row(title="Two Sum (again)", frontendQuestionId="2")]
    out = normalise(rows)
    assert len(out) == 1
    slugs = [p.slug for p in out]
    assert len(slugs) == len(set(slugs))


def test_a_problem_is_never_listed_as_similar_to_itself():
    """Self-links are meaningless and would let a problem be presented as
    preparation for itself."""
    out = normalise([_row(similar_questions='[{"titleSlug": "two-sum"}]')])
    assert out[0].similar_slugs == []


def test_missing_optional_fields_do_not_raise():
    """The loader must survive an upstream row that drops fields entirely."""
    out = normalise([{"titleSlug": "x", "description": "<p>body</p>"}])
    assert len(out) == 1
    problem = out[0]
    assert problem.difficulty == "Unknown"
    assert problem.acceptance_rate == 0.0
    assert problem.topics == []
    assert problem.similar_slugs == []
