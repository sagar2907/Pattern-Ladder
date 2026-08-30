"""Regression tests for defects found during development.

Each test's docstring states the failure it documents, not just the assertion.
These are kept together rather than spread through the suite because they share
a property: every one of them was a bug that produced *plausible* output. None
raised an exception, and none would have been caught by a test that only
checked the pipeline ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pattern_ladder import config
from pattern_ladder.graph.build import add_knn_backfill, build_link_graph
from pattern_ladder.graph.families import build_families, detect_families, order_ladder
from pattern_ladder.retrieval.fusion import reciprocal_rank_fusion
from pattern_ladder.retrieval.search import SearchEngine
from pattern_ladder.understand.schema import Intent


@pytest.fixture
def engine(problems, lexical_index, dense_index, embeddings) -> SearchEngine:
    graph = build_link_graph(problems)
    add_knn_backfill(graph, problems, embeddings)
    families = build_families(detect_families(graph, resolution=1.0), problems, min_size=5)
    return SearchEngine(problems, lexical_index, dense_index, families)


def test_ladder_filters_family_members_by_relevance_to_the_query(engine):
    """Ordering a whole family by difficulty put off-topic members on rung one.

    A family can hold 40+ problems and is not perfectly pure. Since the ladder
    sorts by difficulty and then by acceptance rate, a family's easiest and
    most-accepted problems lead -- and those are frequently the ones least
    related to the query. On the real corpus, "detect whether a linked list has
    a cycle" returned a family that did contain the three correct linked-list
    problems, and served a ladder consisting entirely of digit-arithmetic
    problems instead.

    Selecting by relevance before ordering by difficulty fixed it, raising the
    share of queries whose ladder contains the canonical problem from 0.40 to
    0.85 on the twenty-query smoke set.
    """
    response = engine.search(
        "monotonic stack next greater element", Intent(technique="monotonic stack"),
        use_reranker=False,
    )
    assert response.ladder is not None
    slugs = {p.slug for p in response.ladder.rungs}
    # The unrelated strays share a family only by construction, never by topic.
    assert "lonely-math" not in slugs


def test_ladder_never_collapses_to_a_single_rung(engine):
    """The relevance filter, applied strictly, could leave one problem.

    A one-rung ladder is not a progression -- it is a single result wearing a
    ladder's label. When the filter is that aggressive the honest response is
    the most relevant few members, not nothing.
    """
    for query in ("monotonic stack", "sliding window shrink", "stack"):
        response = engine.search(query, Intent(), use_reranker=False)
        if response.ladder is not None:
            assert len(response.ladder.rungs) >= min(
                config.LADDER_MIN_RUNGS, response.ladder.truncated_from
            )


def test_reranker_is_fused_with_retrieval_not_substituted_for_it(engine):
    """Letting the cross-encoder overwrite the retrieval order lost recall.

    Measured on the smoke set, rerank-as-final-word scored hit@5 = 0.85 against
    0.90 for no reranking at all, and on obliquely-phrased queries -- the ones
    this project exists to serve -- 0.83 against 1.00. It did sharpen precision,
    pulling the mean rank of found problems from 1.56 to 1.06, so it is a trade
    rather than an upgrade. Fusing the two orderings scored 0.95, beating both.

    This test pins the structural property that makes that possible: the fused
    result must be able to differ from either input ordering alone.
    """
    assert config.RERANK_FUSION is True
    retrieval = [(1, 0.0), (2, 0.0), (3, 0.0)]
    reranked = [(3, 5.0), (2, 4.0), (1, 3.0)]
    fused = [doc for doc, _ in reciprocal_rank_fusion([retrieval, reranked])]
    assert fused != [doc for doc, _ in retrieval]
    assert fused != [doc for doc, _ in reranked]


def test_backfill_does_not_overwhelm_the_curated_graph(problems, embeddings):
    """The first backfill added 5,494 inferred edges to 1,932 curated ones.

    At a 3:1 ratio the curated links stopped deciding family boundaries and the
    clustering became embedding clusters wearing a similar-questions costume:
    family count fell from 54 to 27 and the largest family reached 260
    problems, which is a topic rather than a pattern.
    """
    graph = build_link_graph(problems)
    curated = graph.number_of_edges()
    under_connected = {
        p.slug for p in problems if graph.degree(p.slug) <= config.KNN_MAX_DEGREE
    }
    well_connected = {
        p.slug: graph.degree(p.slug)
        for p in problems
        if graph.degree(p.slug) > config.KNN_MAX_DEGREE
    }

    add_knn_backfill(graph, problems, embeddings)
    inferred = graph.number_of_edges() - curated

    # The structural bound is what actually prevents the runaway: only
    # under-connected nodes are eligible, and each takes at most `neighbours`
    # edges. Asserting the real corpus ratio directly would not work here --
    # this twelve-problem fixture is mostly isolated by design, so inferred
    # edges legitimately outnumber curated ones at this scale. The bound holds
    # at every scale; on the real corpus it yields 2,463 inferred against 1,932
    # curated.
    assert inferred <= len(under_connected) * config.KNN_NEIGHBOURS

    # And no inferred edge joins two already-well-connected nodes, which is
    # what stops the backfill bridging and merging established families. Note
    # the asymmetry: a well-connected node may still *receive* an edge from a
    # stray attaching itself, it just never initiates one.
    for source, target, data in graph.edges(data=True):
        if data.get("source") == "knn":
            assert source in under_connected or target in under_connected
            assert not (source in well_connected and target in well_connected)


def test_louvain_resolution_is_high_enough_to_separate_distinct_patterns(problems, embeddings):
    """At the default resolution one community held three unrelated patterns.

    On the real corpus a single 34-member community contained queue/stack
    design problems, heap/greedy problems, and anagram/sliding-window problems.
    Minimum Window Substring therefore lived in a family named "Queue / Design",
    and a sliding-window query served a ladder of queue problems. Raising the
    resolution lifted ladder-family accuracy from 0.583 to 0.833.
    """
    assert config.LOUVAIN_RESOLUTION > 1.0

    graph = build_link_graph(problems)
    add_knn_backfill(graph, problems, embeddings)
    communities = detect_families(graph, resolution=config.LOUVAIN_RESOLUTION)
    window = next((c for c in communities if "window-medium" in c), set())
    assert "stack-medium" not in window


def test_positional_artefacts_stay_aligned_with_the_corpus(problems, embeddings):
    """Embedding rows and BM25 doc ids are positional.

    If corpus ordering were not deterministic, a rebuild would permute the rows
    against the problems without any error being raised. Every result would
    just be quietly wrong, which is the hardest kind of failure to notice.
    """
    assert embeddings.shape[0] == len(problems)
    ids = [p.problem_id for p in problems]
    assert ids == sorted(ids)


def test_ladder_ordering_is_stable_under_input_permutation(problems):
    """Ties on difficulty and acceptance rate must not reorder between runs."""
    forward = [p.slug for p in order_ladder(problems)]
    backward = [p.slug for p in order_ladder(list(reversed(problems)))]
    assert forward == backward


def test_normalised_vectors_keep_cosine_and_dot_product_identical(embeddings):
    """Search uses a raw dot product. If any row were not unit length, that
    would silently stop being cosine similarity for that row alone."""
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_blank_query_returns_nothing_rather_than_confident_results(engine):
    """A query with no letters or digits produced a full page of results.

    The dense retriever is exhaustive: it embeds whatever it is given, however
    meaningless, and returns that point's nearest neighbours. So an empty
    string, a run of spaces, or "!!! ???" each came back with five ranked
    problems, per-result explanations, and a ladder -- all of it about nothing.

    The interface happened to guard against empty input, which is exactly why
    this survived: the engine is the public surface and was relying on its
    caller to be careful.
    """
    for query in ("", "   ", "!!! ??? ...", "---"):
        response = engine.search(query, Intent(), use_reranker=False)
        assert response.results == [], f"{query!r} produced results"
        assert response.ladder is None
        assert response.notes


def test_a_real_query_is_not_caught_by_the_blank_guard(engine):
    """The guard must stay narrow. A bare number could be a problem id and a
    single letter is a poor query rather than an empty one; neither is
    nonsense in the way an empty string is."""
    for query in ("42", "a", "stack"):
        response = engine.search(query, Intent(), use_reranker=False)
        assert response.results, f"{query!r} was wrongly rejected"


def test_ladder_floor_survives_a_negative_best_similarity(engine, problems, monkeypatch):
    """A relative floor inverts when the best similarity is negative.

    The filter kept members scoring at least `best * 0.8`. For a positive best
    that is a sensible floor. For a negative one it is *above* the best -- 0.8
    times -0.05 is -0.04 -- so the single most relevant member failed its own
    threshold and the comparison started selecting the least similar members
    instead of the most.

    Reachable on the real corpus: one family in 1,507 query-family pairs probed
    had a negative best similarity. It never raised, because the minimum-rungs
    fallback quietly refilled the ladder, so the only symptom was the relevance
    filter silently ceasing to work.
    """
    import numpy as np

    # Force every similarity negative, worst-case for the floor arithmetic.
    monkeypatch.setattr(
        engine.dense,
        "encode_query",
        lambda _text: -np.ones(engine.dense.matrix.shape[1], dtype=np.float32)
        / np.sqrt(engine.dense.matrix.shape[1]),
    )
    response = engine.search("monotonic stack", Intent(), use_reranker=False)
    if response.ladder is not None:
        assert response.ladder.rungs
        # The most relevant member must be on the ladder it is the best of.
        rows = [
            engine._row_of[s]
            for s in response.ladder.family.members
            if s in engine._row_of
        ]
        similarity = engine.dense.matrix[rows] @ engine.dense.encode_query("x")
        best_row = rows[int(np.argmax(similarity))]
        best_slug = engine.problems[best_row].slug
        assert best_slug in {p.slug for p in response.ladder.rungs}


def test_stale_artefacts_are_refused_at_load_time(problems, embeddings, tmp_path):
    """Mismatched cached artefacts loaded silently and produced wrong answers.

    The embedding matrix and BM25 index are positional -- row 7 is problem 7 --
    and neither file records which corpus it was built from. An interrupted
    rebuild, or embeddings left over from a different indexing setting, gave
    four artefacts that loaded without complaint and disagreed about what
    document 7 was. Every result stayed plausible and every one was wrong; the
    first sign of trouble was an IndexError from inside a matrix slice, if the
    sizes happened to differ enough to reach one.
    """
    import numpy as np
    import pytest

    from pattern_ladder.index.build import _assert_artefacts_aligned
    from pattern_ladder.index.dense import DenseIndex
    from pattern_ladder.index.lexical import LexicalIndex

    lexical = LexicalIndex.build([p.index_text for p in problems])

    # Correctly aligned artefacts must pass.
    _assert_artefacts_aligned(problems, lexical, DenseIndex(embeddings))

    truncated = DenseIndex(np.ascontiguousarray(embeddings[:3]))
    with pytest.raises(ValueError, match="stale"):
        _assert_artefacts_aligned(problems, lexical, truncated)

    with pytest.raises(ValueError, match="stale"):
        _assert_artefacts_aligned(problems[:2], lexical, DenseIndex(embeddings[:2]))


def test_index_build_derivation_runs_without_network(problems, embeddings):
    """Regression: the build's derivation step had no test at all.

    Because graph construction, clustering and naming lived inline inside
    `build_all` -- which downloads a corpus and loads a model, so no offline
    test could reach it -- the whole step was uncovered. A parameter named
    `describe` was later added there, silently shadowing an imported function
    of the same name, so `describe(graph)` became `False(graph)`. The suite
    stayed green; only the linter noticed.

    Extracting the pure computation makes it reachable, which is the actual
    fix. This test would have failed on that shadowing with a TypeError.
    """
    from pattern_ladder.index.build import derive_families

    families, link_stats, final_stats, described = derive_families(
        problems, embeddings, knn_backfill=True, describe_families=False, resolution=1.0
    )
    # Asserting on the families themselves, not just the graph statistics: an
    # earlier version of this test passed while producing an empty list.
    assert families
    assert all(f.members for f in families)
    assert link_stats.nodes == len(problems)
    assert final_stats.edges >= link_stats.edges
    assert final_stats.isolated <= link_stats.isolated
    assert described == 0
    assert all(f.description is None for f in families)


def test_family_descriptions_are_optional_and_additive(problems, embeddings, monkeypatch):
    """A description must never replace or disturb the deterministic name.

    Every measurement that scores against family names scores against `name`,
    so an optional model-written field has to leave that string untouched --
    otherwise enabling descriptions would silently move the evaluation numbers.
    """
    from pattern_ladder.index.build import derive_families

    baseline, _, _, _ = derive_families(problems, embeddings, describe_families=False)

    monkeypatch.setattr(
        "pattern_ladder.index.build.describe_all",
        lambda families, _problems, **_kw: {f.family_id: "a described pattern" for f in families},
    )
    described_families, _, _, count = derive_families(
        problems, embeddings, describe_families=True
    )

    assert count == len(described_families)
    assert [f.name for f in described_families] == [f.name for f in baseline]
    assert [f.members for f in described_families] == [f.members for f in baseline]
    for family in described_families:
        assert family.description == "a described pattern"
        assert family.headline == "a described pattern"


def test_family_headline_falls_back_to_the_deterministic_name():
    """With no description the headline is the tag-derived name, unchanged."""
    from pattern_ladder.graph.families import Family

    plain = Family(family_id=0, name="Stack / Array", tags=["Stack"], members=["a"], size=1)
    assert plain.headline == "Stack / Array"
    assert plain.description is None


def test_a_rebuild_preserves_descriptions_it_did_not_regenerate(problems, embeddings):
    """Regression: rebuilding the index silently discarded 137 API calls.

    Descriptions are written by a model, one call per family. A rebuild
    triggered for an unrelated reason -- adding a field, changing a constant --
    overwrote families.json with undescribed families, and the only sign was a
    manifest counter dropping to zero.

    Carrying them forward is matched on the exact member set rather than on
    family_id, because ids are positional over the sorted community list: one
    inserted or removed community renumbers everything after it, and a
    description would then be attached to the wrong pattern.
    """
    from dataclasses import replace as dc_replace

    from pattern_ladder.index.build import derive_families

    first, _, _, _ = derive_families(
        problems, embeddings, describe_families=False, resolution=1.0
    )
    assert first, "fixture produced no families"

    described = [dc_replace(f, description=f"described {f.family_id}") for f in first]

    second, _, _, count = derive_families(
        problems, embeddings, describe_families=False, existing=described, resolution=1.0
    )
    assert count == len(second)
    for family in second:
        assert family.description is not None

    # A family whose membership changed must not inherit a stale description.
    unrelated = [dc_replace(f, members=["a-slug-that-does-not-exist"]) for f in described]
    third, _, _, third_count = derive_families(
        problems, embeddings, describe_families=False, existing=unrelated, resolution=1.0
    )
    assert third_count == 0
    assert all(f.description is None for f in third)


def test_the_interpreter_is_pinned():
    """Regression: the family count depended on which Python you happened to get.

    The lockfile carries numpy twice, keyed on Python version -- 2.4.6 below
    3.12, 2.5.2 at or above. Louvain's modularity comparisons come down to
    floating-point ties often enough that the numeric stack decides some of
    them, so an identical graph with a fixed seed clustered into 472
    communities under one numpy and 473 under the other.

    With `requires-python = ">=3.11"` and no pin, `uv sync` on a fresh clone
    resolved 3.11 while the published numbers had been measured on 3.12. The
    documented results were not reproducible from the repository, and nothing
    said so.

    CI still exercises both versions explicitly; this only fixes the default.
    """
    from pathlib import Path

    pin = Path(__file__).resolve().parent.parent / ".python-version"
    assert pin.is_file(), ".python-version is missing; builds are not reproducible"
    assert pin.read_text(encoding="utf-8").strip() == "3.12"


def test_the_app_bootstraps_its_own_index(monkeypatch, tmp_path):
    """A fresh deployment has no cached artefacts and must build them.

    The artefacts are derived data and deliberately not committed. Without a
    bootstrap the deployed app starts, calls load_engine(), and dies with a
    message telling the user to run a script the host will never run -- an
    error that is correct locally and useless in production.

    Verified here without touching the network: load_engine must raise
    FileNotFoundError (not something vaguer) when the artefacts are absent, so
    that the app has a specific exception to catch.
    """
    from pattern_ladder import config
    from pattern_ladder.engine import load_engine

    empty = config.Paths.under(tmp_path / "nothing-here")
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_engine(empty)


def test_difficulty_spread_is_shown_in_difficulty_order():
    """Regression: the family's difficulty breakdown was sorted alphabetically.

    A dict of {"Easy": 3, "Hard": 6, "Medium": 13} sorted by key renders as
    "3 Easy, 6 Hard, 13 Medium", which places Hard between Easy and Medium and
    contradicts the ladder printed immediately above it. Harmless in the sense
    that no number is wrong, and confusing in exactly the place the interface
    is trying to teach an ordering.
    """
    from pathlib import Path

    app_path = Path(__file__).resolve().parent.parent / "src" / "pattern_ladder" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    # Read the constant without importing the module, which would start
    # Streamlit; the module runs main() at import time by design.
    assert 'DIFFICULTY_ORDER = ("Easy", "Medium", "Hard")' in source
    assert "for level in DIFFICULTY_ORDER" in source
    assert "sorted(spread.items())" not in source

    spread = {"Easy": 3, "Hard": 6, "Medium": 13}
    order = ("Easy", "Medium", "Hard")
    rendered = ", ".join(f"{spread[level]} {level}" for level in order if level in spread)
    assert rendered == "3 Easy, 13 Medium, 6 Hard"


def test_ci_overrides_the_pinned_interpreter_for_every_uv_command():
    """Regression: pinning .python-version broke CI on 3.11.

    The pin exists so a plain `uv sync` is reproducible. CI deliberately
    overrides it to test both supported versions -- but it only overrode the
    sync step. The lint step ran a bare `uv run ruff check .`, which read
    .python-version, disagreed with the 3.11 environment the previous step had
    built, and failed before ruff was ever invoked.

    The failure looked like a lint error and was not one: ruff 0.16.5 passes on
    both interpreters, verified by running the binary directly against a 3.11
    environment. Three pushes went out with CI red while the README claimed it
    was green.
    """
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    # A job-level UV_PYTHON covers every uv invocation, not just the one that
    # happens to carry an explicit flag.
    assert "UV_PYTHON:" in workflow
    assert "${{ matrix.python-version }}" in workflow


def test_encode_batch_is_small_enough_to_build_within_the_hosting_ceiling():
    """Regression: a cold index build peaked at 976MB against a ~1GB ceiling.

    Steady-state memory had been measured at ~700MB and looked comfortable.
    The peak had not, and the peak is what a host kills you for: 48MB of
    headroom before the web server's own footprint, during the one operation a
    fresh deployment performs before serving anything.

    The cost is entirely the activations of a single forward pass, and it is
    free to fix. Encoding all 2,830 documents took 110 seconds at every batch
    size tried, while peak memory ran 951MB at 64, 799MB at 32 and 687MB at 16.

    Chunking the input was tried first, on the theory that the corpus list and
    the encoder's internal copies of it were the cost. It moved the peak by
    1MB, and was removed.
    """
    from pattern_ladder import config

    assert config.ENCODE_BATCH <= 16


def test_partial_artefacts_are_reported_as_recoverable(problems, embeddings, tmp_path):
    """A half-finished build must be repairable by the app, not fatal.

    A build interrupted partway -- a network timeout, a container restarted
    mid-encode -- leaves some artefacts written and others missing or the wrong
    size. Loading rejects that with a ValueError, which is correct; but the
    interface originally caught only FileNotFoundError when deciding whether to
    bootstrap, so an interrupted first run left the deployment permanently
    broken in a state it could have repaired itself. On a hosted service there
    is nobody to log in and rerun a script.

    This pins the two exception types the app must treat as "rebuild", and that
    a size mismatch really does raise the second of them.
    """
    import numpy as np

    from pattern_ladder.index.build import _assert_artefacts_aligned
    from pattern_ladder.index.dense import DenseIndex
    from pattern_ladder.index.lexical import LexicalIndex

    lexical = LexicalIndex.build([p.index_text for p in problems])
    truncated = DenseIndex(np.ascontiguousarray(embeddings[:2]))
    with pytest.raises(ValueError):
        _assert_artefacts_aligned(problems, lexical, truncated)

    app_source = (
        Path(__file__).resolve().parent.parent / "src" / "pattern_ladder" / "app.py"
    ).read_text(encoding="utf-8")
    assert "except (FileNotFoundError, ValueError):" in app_source


def test_pdf_default_output_matches_the_committed_filenames():
    """Regenerating a document must overwrite the copy that is committed.

    The renderer defaulted its output to the source name with a .pdf suffix, so
    `render_pdf.py docs/report.md` -- the exact command the documentation gives
    for rebuilding the report -- wrote docs/report.pdf, while the file tracked
    in the repository was docs/Pattern-Ladder-report.pdf. Following the
    documented instructions therefore produced an untracked file and left the
    committed PDF untouched, which is the quiet version of the failure: no
    error, no diff, and a stale document that still looks freshly built.

    Asserting only that the committed PDFs exist would not have caught this --
    they existed throughout, which is precisely why the bug went unnoticed. So
    the naming rule itself is asserted, and then checked against what is really
    in docs/, which is what ties the rule to the repository.
    """
    import runpy

    root = Path(__file__).resolve().parent.parent
    module = runpy.run_path(str(root / "scripts" / "render_pdf.py"), run_name="__pinned__")
    default_output_path = module["default_output_path"]

    assert default_output_path(Path("docs/report.md")).name == "Pattern-Ladder-report.pdf"
    assert default_output_path(Path("a/b/notes.md")) == Path("a/b/Pattern-Ladder-notes.pdf")

    sources = sorted((root / "docs").glob("*.md"))
    assert sources, "no documents to render"
    for source in sources:
        rendered = default_output_path(source)
        assert rendered.is_file(), f"{source.name} renders to {rendered.name}, not in docs/"


def test_no_rule_is_stranded_before_a_part_heading():
    """A horizontal rule must never be the last thing before a forced break.

    Level-1 headings force a page break, and the report puts a `---` rule
    immediately before each one. When the preceding part happened to end near
    the bottom of a page, the rule spilled onto the next page and the break then
    pushed the heading past it -- leaving a page holding one horizontal line and
    a footer. The renderer's own blank-page check caught it at 45 pages, which
    is the only reason it was noticed; extracted text was empty, so nothing that
    read the text would have seen a problem.

    Pinned at the story level rather than by counting pages, because pagination
    depends on font metrics and on how much prose happens to sit above the
    break. The invariant is what matters: no rule directly before a page break.
    """
    import runpy

    from reportlab.platypus import PageBreak
    from reportlab.platypus.flowables import HRFlowable

    root = Path(__file__).resolve().parent.parent
    module = runpy.run_path(str(root / "scripts" / "render_pdf.py"), run_name="__pinned__")
    body, mono, _ = module["register_fonts"]()
    renderer = module["MarkdownRenderer"](body, mono)

    story = renderer.convert("Some prose.\n\n---\n\n# Part 2 — Next\n\nMore prose.\n")
    kinds = [type(f) for f in story]
    assert PageBreak in kinds, "a part heading must still start a new page"
    assert HRFlowable not in kinds, "the rule before a part heading must be dropped"

    # A rule that is not followed by a part heading is still a rule.
    plain = renderer.convert("Some prose.\n\n---\n\nMore prose.\n")
    assert HRFlowable in [type(f) for f in plain]

    # And the real document, which is what actually regressed.
    report = renderer.convert((root / "docs" / "report.md").read_text(encoding="utf-8"))
    for earlier, later in zip(report, report[1:], strict=False):
        assert not (isinstance(earlier, HRFlowable) and isinstance(later, PageBreak))


def _family(name, tags, description):
    from pattern_ladder.graph.families import Family

    return Family(
        family_id=1, name=name, tags=tags, members=["a"], size=1, description=description
    )


def test_a_mixed_family_does_not_lead_with_the_wrong_technique():
    """The headline must not name a technique the student did not ask about.

    A community can hold more than one technique. One is tagged both Sliding
    Window and Prefix Sum, and its single model-written description -- produced
    once per family, with no knowledge of any query -- says "use prefix sums to
    compute subarray sums quickly". The interface led with that, so the query
    the whole project is named for ("shrink a window from the left") was
    answered with the wrong pattern name while the deterministic name that does
    say Sliding Window sat beneath it in small print.

    Nothing caught it because nothing measured it: `family_at_1` scores
    `family.name` and the family's tags, and the string the interface actually
    leads with is `family.headline`, which is the description whenever one
    exists. The metric read 0.833 and was right about the string it checked.
    """
    mixed = _family(
        "Subarray / Sliding Window / Prefix Sum",
        ["Sliding Window", "Prefix Sum", "Array"],
        "use prefix sums to compute subarray sums quickly",
    )
    assert mixed.headline_for("sliding window") == mixed.name
    # Asked about the technique the description does name, it still leads.
    assert mixed.headline_for("prefix sum") == mixed.description
    # With no technique parsed there is nothing to disagree with.
    assert mixed.headline_for(None) == mixed.description
    # A technique this family does not cover is not evidence against it.
    assert mixed.headline_for("dijkstra") == mixed.description


def test_a_good_description_is_not_demoted_for_paraphrasing():
    """Only a sibling technique demotes a description, not different wording.

    The first version of the rule above asked merely whether the description
    mentioned the technique. That is too broad and made the display worse on
    queries it already handled: "reordering nodes by pointer manipulation"
    never says "linked list" and is clearly the better of the two lines, yet it
    would have been replaced by "Linked / Linked List / Recursion".

    Naming a *sibling tag of the same family* is the signal that the
    description picked one technique out of several, rather than paraphrasing
    the only one there is.
    """
    paraphrase = _family(
        "Linked / Linked List / Recursion",
        ["Linked List"],
        "reordering nodes by pointer manipulation",
    )
    assert paraphrase.headline_for("linked list") == paraphrase.description
    assert paraphrase.headline_for("linked list") == paraphrase.headline


def test_the_interface_uses_the_query_aware_headline():
    """The app must call headline_for, not headline, for the ladder title.

    The fix is worthless if the interface keeps reading the plain property, and
    that substitution is invisible in any unit test of the dataclass.
    """
    source = (
        Path(__file__).resolve().parent.parent / "src" / "pattern_ladder" / "app.py"
    ).read_text(encoding="utf-8")
    assert "family.headline_for(intent.technique)" in source
    assert 'f"**{headline}**' in source


def test_the_result_reason_names_the_technique_that_was_asked_about():
    """The per-result reason must not name the wrong technique either.

    The query-aware headline was added for the ladder title and the ladder
    title only, which left the same wrong label in the sentence attached to
    every individual result -- the one the brief calls the reason per
    recommendation, and the place a student is most likely to read it. Fixing
    the heading and not the reasons is the more embarrassing half-fix, because
    the heading appears once and the reason appears five times.

    Pinned here rather than through the interface because explain() is where
    the sentence is assembled, and it took the family but not the query.
    """
    from pattern_ladder.data import Problem
    from pattern_ladder.explain import MATCH_BOTH, explain

    mixed = _family(
        "Subarray / Sliding Window / Prefix Sum",
        ["Sliding Window", "Prefix Sum", "Array"],
        "use prefix sums to compute subarray sums quickly",
    )
    problem = Problem(
        slug="minimum-size-subarray-sum",
        problem_id=209,
        title="Minimum Size Subarray Sum",
        difficulty="Medium",
        acceptance_rate=48.0,
        topics=["Array", "Sliding Window"],
        description="find the shortest subarray whose sum is at least target",
        index_text="minimum size subarray sum",
    )
    asked = explain(problem, kind=MATCH_BOTH, family=mixed, technique="sliding window")
    assert "Sliding Window" in asked
    assert "prefix sums" not in asked

    # With no technique parsed the description is still the better label.
    unasked = explain(problem, kind=MATCH_BOTH, family=mixed, technique=None)
    assert "prefix sums" in unasked


def test_search_passes_the_technique_into_the_reason():
    """The reason is only query-aware if the pipeline actually forwards it."""
    source = (
        Path(__file__).resolve().parent.parent
        / "src" / "pattern_ladder" / "retrieval" / "search.py"
    ).read_text(encoding="utf-8")
    assert "technique=intent.technique" in source


def test_make_clean_cannot_delete_the_committed_index():
    """`make clean` must not remove tracked files.

    The target was written when artifacts/ held nothing but derived data, and
    said so: "everything here is rebuildable". Committing the built index made
    that false without touching the Makefile, so `make clean` would have
    deleted fifteen megabytes of tracked content and left the working tree full
    of deletions -- recoverable, but only by someone who realised what had
    happened. The deployment depends on that index being present, so a checkout
    that has been cleaned would silently go back to rebuilding on every cold
    start, which is the throttle this all started with.

    `git clean` is used because it will not touch tracked files whatever the
    ignore rules say. This test pins the absence of the blunt instrument rather
    than the presence of the careful one, since any `rm -rf artifacts` is wrong
    regardless of what replaces it.
    """
    makefile = (Path(__file__).resolve().parent.parent / "Makefile").read_text(encoding="utf-8")
    assert "rm -rf artifacts " not in makefile
    assert "rm -rf artifacts\n" not in makefile
    assert "git clean -xdf artifacts" in makefile


def test_make_test_does_not_double_apply_quiet():
    """pyproject already sets -q; a second one gives -qq and hides the summary.

    The same mistake was found and fixed in scripts/check_docs.py, where -qq
    suppressed the "N passed" line the script parses and made it report that it
    could not determine the test count while the suite was passing. The Makefile
    had the identical bug and was not checked at the time.
    """
    root = Path(__file__).resolve().parent.parent
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '-q' in pyproject.split("addopts")[1].split("\n")[0]
    assert "uv run pytest -q" not in makefile


def test_the_readme_layout_lists_every_module():
    """The file map must not silently fall behind the code.

    Three modules were missing from it: engine.py, graph/naming.py and
    understand/cache.py. All three were added after the map was written, and
    naming.py is the one that produces the family descriptions -- so the map
    omitted the module behind the defect in Part 6g while claiming to show the
    shape of the project.

    A file map is exactly the kind of documentation that rots without
    complaining, because nothing reads it except a person forming their first
    impression of the codebase, and they have no way to know what is absent.
    """
    root = Path(__file__).resolve().parent.parent
    layout = (root / "README.md").read_text(encoding="utf-8").split("## Layout")[1].split("\n##")[0]
    missing = [
        f.name
        for f in sorted((root / "src" / "pattern_ladder").rglob("*.py"))
        if f.name != "__init__.py" and f.name not in layout
    ]
    assert not missing, f"not listed in the README layout: {missing}"


def test_the_image_ships_the_committed_index_rather_than_rebuilding_it():
    """The container must carry the same artefacts as the repository.

    The Dockerfile ran build_index.py, which looked like thoroughness and was
    the opposite. A build inside the image has no API key, so it produced 137
    families with *zero* model-written descriptions where the committed index
    has 120 -- three minutes of build time spent manufacturing a degraded copy
    of an artefact that was already sitting in the repository. Both manifests
    recorded the difference the whole time, and nothing ever compared them.

    Copying the index instead makes the image byte-identical to the deployment
    and takes the step from 192 seconds to 53. Loading it during the build is
    what turns a bad copy into a failed build rather than a failed first query.
    """
    root = Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    ignore = (root / ".dockerignore").read_text(encoding="utf-8")

    assert "COPY artifacts/index/" in dockerfile
    assert "scripts/build_index.py" not in dockerfile
    # The index has to survive .dockerignore or the COPY above silently
    # produces nothing and the failure surfaces as a missing-artefact error.
    assert "!artifacts/index/" in ignore
    # A bad copy must fail the build, not the first query.
    assert "load_engine()" in dockerfile


def test_the_image_does_not_ship_the_dependency_cache_twice():
    """uv's cache must not be baked into the image alongside the virtualenv.

    UV_LINK_MODE=copy is set so uv copies each wheel out of its cache into the
    virtualenv instead of hardlinking, which is what you want in a container
    where the two may sit on different filesystems. The consequence nobody
    looked for is that both copies then persist: 1,367 MB of cache beside a
    1,365 MB virtualenv, in an image of 4.25 GB. Roughly a third of the image
    was a build artefact with nothing at runtime reaching it, and every layer
    export paid to compress and write it.

    Passing --no-cache removed 1.81 GB, taking the image to 2.44 GB with
    byte-identical behaviour. Docker's own layer cache is unaffected, since
    that is a different cache from uv's.
    """
    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(encoding="utf-8")
    syncs = [line for line in dockerfile.splitlines() if "uv sync" in line and "RUN" in line]
    assert syncs, "no uv sync in the Dockerfile"
    for line in syncs:
        assert "--no-cache" in line, f"uv sync without --no-cache: {line.strip()}"


def test_rendering_with_no_arguments_covers_every_document():
    """The no-argument render must produce every committed PDF, not just one.

    Two documents are committed as PDFs, and the default rendered only the
    report. `make report` therefore refreshed one and left the other at
    whatever it had been, without any indication -- the run it printed was a
    success, for the file it happened to be rendering. This is the same shape
    as the output-path defect above: a command that regenerates most of what it
    claims to.
    """
    import runpy

    root = Path(__file__).resolve().parent.parent
    module = runpy.run_path(str(root / "scripts" / "render_pdf.py"), run_name="__pinned__")
    default_output_path = module["default_output_path"]

    sources = sorted((root / "docs").glob("*.md"))
    assert len(sources) >= 2, "this test is only meaningful with more than one document"
    for source in sources:
        assert default_output_path(source).is_file()

    makefile = (root / "Makefile").read_text(encoding="utf-8")
    # The bare form is what must cover everything; a named source would not.
    assert "render_pdf.py\n" in makefile


def test_a_blank_page_fails_the_render(monkeypatch, tmp_path):
    """A blank page must exit non-zero, not merely print a warning.

    The renderer rasterises each page and checks for the signature of a glyph
    failure, which is the right check and is how the stranded-rule defect was
    caught. It then printed "WARNING blank pages: [43]" and returned 0. A check
    that reports a failure through stdout while telling its caller everything
    is fine cannot gate anything, and the only reason that defect was noticed
    is that a person happened to read the output.
    """
    import runpy

    root = Path(__file__).resolve().parent.parent
    module = runpy.run_path(str(root / "scripts" / "render_pdf.py"), run_name="__pinned__")

    source = tmp_path / "doc.md"
    source.write_text("# Title\n\nSome prose.\n", encoding="utf-8")

    calls = {}

    def fake_build(markdown, out_path, title):
        calls["built"] = True
        return out_path, set()

    def fake_verify(pdf_path, image_dir=None):
        return {"pages": 2, "blank": [2], "suspicious": [], "chars": 10}

    monkeypatch.setattr(sys, "argv", ["render_pdf.py", str(source)])
    module["build_pdf"] = fake_build
    module["verify"] = fake_verify
    main = module["main"]
    main.__globals__["build_pdf"] = fake_build
    main.__globals__["verify"] = fake_verify

    assert main() == 1, "a blank page must fail the render"
    assert calls["built"]


def test_the_pdf_render_is_deterministic():
    """The same markdown must render to the same bytes.

    reportlab stamps the wall-clock time into /CreationDate and /ModDate and
    generates a random document /ID unless told not to. The PDFs are committed,
    so that turned every render into a modification: the working tree went dirty
    whether or not a word had changed, a diff could not distinguish a real edit
    from a rebuild, and there was no way to answer whether the committed PDF
    matched its source short of reading it.

    Pinned by asserting the mechanism rather than by rendering twice, which
    would add twenty seconds to an offline suite that finishes in seventeen.
    scripts/check_docs.py does the real comparison, where the cost is
    affordable.
    """
    import runpy

    root = Path(__file__).resolve().parent.parent
    source = (root / "scripts" / "render_pdf.py").read_text(encoding="utf-8")
    assert "rl_config.invariant = 1" in source

    module = runpy.run_path(str(root / "scripts" / "render_pdf.py"), run_name="__pinned__")
    assert "build_pdf" in module


def test_the_readme_is_not_in_the_dependency_layer():
    """Editing prose must not invalidate the dependency install.

    pyproject sets readme = "README.md", so hatchling needs it to build the
    project wheel, and it was copied alongside pyproject.toml and uv.lock --
    which put it in the layer immediately above `uv sync`. Every edit to the
    README therefore invalidated the dependency install and the model warm-up
    beneath it: a documentation typo cost 150 seconds of rebuild instead of 70.

    The README belongs with the source it describes, which is also what changes
    at the same time as it.
    """
    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(encoding="utf-8")
    manifest_copy = next(
        line for line in dockerfile.splitlines()
        if line.startswith("COPY") and "pyproject.toml" in line
    )
    assert "README.md" not in manifest_copy, (
        "README.md shares a layer with the dependency manifest"
    )
    # It still has to arrive before the project is installed, or the wheel build
    # fails on a missing readme.
    body = dockerfile.split("COPY README.md")[1]
    assert "uv sync --frozen --no-dev --no-cache" in body
