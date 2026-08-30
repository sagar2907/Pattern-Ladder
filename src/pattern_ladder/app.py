"""Streamlit interface.

The layout follows one rule: the system shows its own reading before it shows
its answers. A student who cannot name the technique they are missing has no
way to tell a good result from a bad one, so the parsed intent, the retrieval
arm that found each result, and the provenance of the ladder are all on screen.
A tool that reveals its reasoning is correctable; one that is silently right is
only trusted until the first time it is silently wrong.

Run: streamlit run src/pattern_ladder/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit executes this file as a script rather than importing it as part of
# the package, so the package root has to be on the path before the imports
# below resolve.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_ladder import config  # noqa: E402
from pattern_ladder.engine import load_engine  # noqa: E402
from pattern_ladder.index.build import build_all  # noqa: E402
from pattern_ladder.understand import groq_client  # noqa: E402

EXAMPLES = [
    "I keep failing problems where you shrink a window from the left",
    "how do I find the next greater element to the right of each item",
    "detect whether a linked list has a cycle in it",
    "problems where you binary search on the answer instead of an index",
]

DIFFICULTY_COLOUR = {"Easy": "#1a7f37", "Medium": "#9a6700", "Hard": "#cf222e"}

# Difficulty order for display. Dict order is the source of truth here so
# that anything derived from it reads in the order a student climbs.
DIFFICULTY_ORDER = ("Easy", "Medium", "Hard")


@st.cache_resource(show_spinner="Loading indexes and models...")
def _engine():
    """Loaded once per process. The cache is what makes the second query fast.

    Builds the index first if it is not there. The built index is committed, so
    on a normal deployment this path never runs and a cold start is a file
    read. It exists for the case where the artefacts are absent or damaged --
    someone cleaning the working tree, a partial checkout -- because the
    alternative is an app that starts and immediately dies with a message about
    running a script the host will never run.

    That the index is committed at all is a reversal driven by a real
    deployment. It is derived data and was excluded on principle, which meant a
    hosted instance rebuilt it on every cold start: roughly three minutes of
    sustained CPU to download the corpus and encode 2,830 documents. The free
    host throttles for exactly that, and did.

    Both absent *and* inconsistent artefacts trigger a rebuild. A build
    interrupted partway -- a network timeout, a container restarted mid-encode
    -- leaves some files written and others not, which loading rejects with a
    ValueError rather than a FileNotFoundError. Catching only the latter would
    leave a deployment permanently broken in a state it could have repaired by
    itself, and on a hosted service nobody can log in to rerun a script.
    """
    try:
        return load_engine()
    except (FileNotFoundError, ValueError):
        with st.spinner(
            "First run: downloading the corpus and building the indexes. "
            "This takes a couple of minutes and happens once."
        ):
            build_all()
        return load_engine()


def _load_env() -> None:
    """Read .env if present. Never fails if the file or the package is absent."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    except ImportError:
        pass


def _difficulty_badge(difficulty: str) -> str:
    colour = DIFFICULTY_COLOUR.get(difficulty, "#57606a")
    return (
        f"<span style='background:{colour};color:white;padding:1px 7px;"
        f"border-radius:9px;font-size:0.72rem;font-weight:600'>{difficulty}</span>"
    )


def main() -> None:
    st.set_page_config(page_title="Pattern Ladder", page_icon=":ladder:", layout="wide")
    _load_env()

    st.title("Pattern Ladder")
    st.caption(
        "Describe what you are stuck on, in your own words. "
        "You get the pattern, and an ordered set of problems that drill it."
    )

    with st.sidebar:
        st.subheader("Try")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state["query"] = example

        st.divider()
        st.subheader("Settings")
        top_k = st.slider("Results", 3, 20, config.DEFAULT_TOP_K)
        use_reranker = st.checkbox(
            "Cross-encoder rerank",
            value=True,
            help=(
                "Rescores the top 50 candidates by reading each query-problem "
                "pair together. Fused with the retrieval order rather than "
                "replacing it."
            ),
        )
        use_model = st.checkbox(
            "Use Groq for query understanding",
            value=groq_client.available(),
            disabled=not groq_client.available(),
            help=(
                "Unavailable without GROQ_API_KEY in .env. The rule-based "
                "parser is used instead and the parse is labelled as such."
            ),
        )
        if not groq_client.available():
            st.caption("No GROQ_API_KEY found. Falling back to the offline parser.")

    query = st.text_input(
        "What are you stuck on?",
        key="query",
        placeholder="e.g. I keep failing problems where you shrink a window from the left",
    )
    if not query:
        st.info(
            "The corpus is 2,830 free LeetCode problems. Families are discovered "
            "from the similar-questions graph, not from LeetCode's tags."
        )
        return

    engine = _engine()
    intent = groq_client.understand(query, allow_network=use_model)
    response = engine.search(query, intent, top_k=top_k, use_reranker=use_reranker)

    # The parsed reading, before any results: a wrong reading should be visible
    # at the moment it happens, not inferred from bad results further down.
    source_label = {
        "model": "parsed by the language model",
        "fallback": "parsed by the offline rule-based parser",
        "none": "not parsed",
    }.get(intent.source, intent.source)
    st.markdown(f"**Understood as** &nbsp; `{intent.describe()}` &nbsp; — _{source_label}_")
    for note in intent.notes:
        st.caption(f"note: {note}")
    for note in response.notes:
        st.warning(note, icon=":material/info:")

    if not response.results:
        st.error("Nothing matched. Try describing the mechanic rather than the topic.")
        return

    results_column, ladder_column = st.columns([3, 2], gap="large")

    with results_column:
        st.subheader("Results")
        for position, result in enumerate(response.results, start=1):
            problem = result.problem
            st.markdown(
                f"**{position}. [{problem.title}]({problem.url})** &nbsp; "
                f"{_difficulty_badge(problem.difficulty)} &nbsp; "
                f"<span style='color:#57606a;font-size:0.8rem'>#{problem.problem_id} · "
                f"{problem.acceptance_rate:.0f}% accepted</span>",
                unsafe_allow_html=True,
            )
            st.caption(result.reason)
            if problem.topics:
                st.caption("tags: " + ", ".join(problem.topics))
            st.divider()

    with ladder_column:
        ladder = response.ladder
        if ladder is None:
            st.subheader("Ladder")
            st.info(
                "These results are not in a discovered family, so there is no "
                "ladder for this query. About 14% of the corpus sits outside "
                "any family."
            )
            return

        st.subheader("Your ladder")
        family = ladder.family
        # The description, where one exists, says what the pattern *is*; the
        # tag-derived name says which shelf it sits on. Both are shown, because
        # the name is the deterministic one and a reader should be able to see
        # it -- and because ten families share a name with another, so the name
        # alone cannot always identify which pattern this is.
        #
        # Which of the two leads depends on the query. A family holding more
        # than one technique has a description that committed to one of them,
        # and leading with it names the wrong pattern for a student who asked
        # about the other. See Family.headline_for.
        headline = family.headline_for(intent.technique)
        st.markdown(
            f"**{headline}**  \n"
            f"<span style='color:#57606a;font-size:0.82rem'>"
            f"family of {ladder.truncated_from} problems, discovered from the "
            f"similar-questions graph</span>",
            unsafe_allow_html=True,
        )
        if family.description:
            if headline == family.name:
                st.caption(f"most of this family: {family.description}")
            else:
                st.caption(f"grouped under: {family.name}")
        st.caption("tags shared by this family: " + ", ".join(family.tags))
        st.write("")

        for rung in ladder.rungs:
            marker = " &nbsp; **start here**" if rung.slug == ladder.start_here else ""
            st.markdown(
                f"{_difficulty_badge(rung.difficulty)} &nbsp; "
                f"[{rung.title}]({rung.url}) &nbsp; "
                f"<span style='color:#57606a;font-size:0.78rem'>"
                f"{rung.acceptance_rate:.0f}% accepted</span>{marker}",
                unsafe_allow_html=True,
            )
            # Shown only where a curated link actually exists, so the reason a
            # rung follows another is a recorded fact rather than an inference
            # drawn from their positions in the list.
            preceding = ladder.follows_from.get(rung.slug)
            if preceding:
                st.caption(f"follows on from {preceding} - listed as similar")

        st.write("")
        spread = family.difficulty_spread
        if spread:
            # Difficulty order, not alphabetical. Sorting the keys as strings
            # renders "3 Easy, 6 Hard, 13 Medium", which puts Hard between Easy
            # and Medium and quietly contradicts the ladder right above it.
            counts = ", ".join(
                f"{spread[level]} {level}"
                for level in DIFFICULTY_ORDER
                if level in spread
            )
            extra = [level for level in spread if level not in DIFFICULTY_ORDER]
            if extra:
                counts += ", " + ", ".join(f"{spread[level]} {level}" for level in sorted(extra))
            st.caption(f"whole family: {counts}")
        st.caption(
            "Ordered easiest first, and within one difficulty by how approachable "
            "and how well regarded each problem is. Only family members related "
            "to your query are shown."
        )

        # The hint belongs behind a click. It is genuinely useful when a student
        # is stuck and actively harmful if they read it before trying, so it is
        # shown only for the one problem they are being told to start with.
        start = next(
            (r for r in ladder.rungs if r.slug == ladder.start_here and r.hints), None
        )
        if start is not None:
            with st.expander(f"Stuck on {start.title}? First hint"):
                st.write(start.hints[0])

        if ladder.related_families:
            st.write("")
            st.markdown("**Where this leads**")
            for neighbour in ladder.related_families:
                st.caption(
                    f"{neighbour.headline} - {neighbour.size} problems, "
                    "linked to this family"
                )


main()
