"""Run the smoke evaluation and report retrieval and ladder accuracy.

Two numbers, measured on the twenty queries in eval/smoke_queries.json:

  hit@k          -- the expected problem appears in the top k results.
  ladder_hit     -- the expected problem appears on the ladder itself. This is
                    the one that catches a ladder built from the right family
                    but populated with that family's off-topic members.
  family@1       -- the ladder was built from a family whose name or tags
                    contain the expected keyword. Only scored on the queries
                    that declare one; the rest report as not applicable rather
                    than being counted as passes, because a query with no
                    obviously-correct family cannot be evidence either way.

Both are reported per phrasing style. Oblique queries -- the ones that describe
a technique without naming it -- are the case this project exists for, and they
are the ones a lexical-only system fails, so a single blended average would
hide the result that matters.

Run: python scripts/evaluate.py [--no-rerank] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pattern_ladder.engine import load_engine  # noqa: E402
from pattern_ladder.understand.groq_client import understand  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "smoke_queries.json"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env() -> None:
    """Read .env directly rather than depending on python-dotenv being present.

    Deliberately does not overwrite a variable already set in the environment,
    so a key exported for one run cannot be silently replaced by a stale file.
    """
    if not ENV_PATH.is_file():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def run(k: int = 5, *, use_reranker: bool = True, allow_network: bool = False) -> dict:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["queries"]
    engine = load_engine()

    rows = []
    latencies = []
    for case in cases:
        query = case["query"]
        intent = understand(query, allow_network=allow_network)
        started = time.perf_counter()
        response = engine.search(query, intent, top_k=k, use_reranker=use_reranker)
        latencies.append(time.perf_counter() - started)

        slugs = [r.problem.slug for r in response.results]
        expected = case["expect_slug"]
        rank = slugs.index(expected) + 1 if expected in slugs else None

        ladder_slugs = (
            [r.slug for r in response.ladder.rungs] if response.ladder is not None else []
        )
        ladder_has_expected = expected in ladder_slugs if ladder_slugs else False
        rungs = response.ladder.rungs if response.ladder is not None else []
        ladder_approval = (
            sum(r.approval for r in rungs) / len(rungs) if rungs else None
        )

        wanted_family = case.get("expect_family_contains")
        family_ok: bool | None = None
        family_name = None
        family_headline = None
        description_drops_technique = None
        if response.ladder is not None:
            family = response.ladder.family
            family_name = family.name
            family_headline = family.headline
            if wanted_family:
                haystack = f"{family.name} {' '.join(family.tags)}".lower()
                family_ok = wanted_family.lower() in haystack
                # The metric above scores the tag-derived name. The interface
                # leads with family.headline, which is the model-written
                # description whenever there is one -- so the string a student
                # actually reads was never measured by anything. That is how a
                # family named "Subarray / Sliding Window / Prefix Sum" came to
                # be introduced as "use prefix sums to compute subarray sums
                # quickly" for a query about shrinking a window, while family@1
                # recorded a hit.
                #
                # This is a tripwire, not a score. Asking merely whether the
                # description contains the expected words does not work: a good
                # description names the mechanic instead of repeating the tags,
                # so it often shares no words with the expectation.
                # "reordering nodes by pointer manipulation" for a linked-list
                # query, or "DP to find minimum number of items" for a dynamic
                # programming one, are both better than the names they replace,
                # and a plain substring test calls both of them misses. Scored
                # that way this flagged three queries, of which two were the
                # test being wrong.
                #
                # The condition is therefore narrower: the description must
                # name a *different tag of the same family* while omitting the
                # one asked about. That separates "chose a sibling technique",
                # which misinforms, from "used different words", which does not.
                # Only the sliding-window query survives it, and that one is a
                # genuine defect -- a family tagged both Sliding Window and
                # Prefix Sum is introduced as prefix sums to a student who
                # asked about windows.
                #
                # Scored against the headline the interface actually shows,
                # which is query-aware, rather than against the raw
                # description. Measuring the description would repeat the
                # original mistake in a smaller way: it would report a defect
                # that a reader can no longer encounter, and it would keep
                # reporting one no matter how the display was fixed.
                if family.description:
                    shown = family.headline_for(intent.technique).lower()
                    wanted = wanted_family.lower()
                    description_drops_technique = bool(
                        family_ok
                        and wanted not in shown
                        and any(
                            tag.lower() in shown
                            for tag in family.tags
                            if tag.lower() != wanted
                        )
                    )
        elif wanted_family:
            family_ok = False

        rows.append(
            {
                "query": query,
                "parse_source": intent.source,
                "phrasing": case["phrasing"],
                "expected": expected,
                "rank": rank,
                "hit": rank is not None,
                "expect_family_contains": wanted_family,
                "family": family_name,
                "family_headline": family_headline,
                "family_ok": family_ok,
                "description_drops_technique": description_drops_technique,
                "technique": intent.technique,
                "ladder_has_expected": ladder_has_expected,
                "ladder_size": len(ladder_slugs),
                "ladder_approval": ladder_approval,
            }
        )

    return {"k": k, "reranker": use_reranker, "rows": rows, "latencies": latencies}


def summarise(result: dict) -> dict:
    rows = result["rows"]
    lat = sorted(result["latencies"])

    def hit_rate(subset):
        return round(sum(r["hit"] for r in subset) / len(subset), 3) if subset else None

    scored = [r for r in rows if r["family_ok"] is not None]
    return {
        "k": result["k"],
        "reranker": result["reranker"],
        "queries": len(rows),
        "hit_at_k": hit_rate(rows),
        "hit_at_k_literal": hit_rate([r for r in rows if r["phrasing"] == "literal"]),
        "hit_at_k_oblique": hit_rate([r for r in rows if r["phrasing"] == "oblique"]),
        "mean_rank_of_hits": round(
            sum(r["rank"] for r in rows if r["hit"]) / max(1, sum(r["hit"] for r in rows)), 2
        ),
        "family_scored": len(scored),
        "family_at_1": round(sum(bool(r["family_ok"]) for r in scored) / len(scored), 3)
        if scored
        else None,
        # Count, not a rate: see the note where this is computed. Each one is a
        # query whose family was right and whose displayed headline named a
        # different technique than the one asked about.
        "headline_names_wrong_technique": sum(
            bool(r["description_drops_technique"]) for r in rows
        ),
        # Whether the canonical problem for the query is on the ladder itself.
        # Independent of family naming, so it catches the case where the right
        # family is chosen but the rungs are its off-topic members.
        "ladder_hit": round(sum(r["ladder_has_expected"] for r in rows) / len(rows), 3),
        "ladder_len_mean": round(sum(r["ladder_size"] for r in rows) / len(rows), 2),
        # Mean approval of the problems actually recommended. Acceptance rate
        # alone is blind to whether a problem is well regarded, and a ladder
        # that opens with a widely-disliked problem is a bad recommendation
        # however approachable it is.
        "ladder_approval_mean": round(
            sum(r["ladder_approval"] for r in rows if r["ladder_approval"] is not None)
            / max(1, sum(1 for r in rows if r["ladder_approval"] is not None)),
            4,
        ),
        "ladder_worst_approval": round(
            min((r["ladder_approval"] for r in rows if r["ladder_approval"] is not None),
                default=0.0),
            4,
        ),
        "ladder_len_min": min((r["ladder_size"] for r in rows), default=0),
        # How the query was read matters for interpreting everything above: a
        # run where the model silently fell back is a run of the offline parser
        # wearing a live label.
        "parsed_by_model": sum(r["parse_source"] == "model" for r in rows),
        "parsed_by_fallback": sum(r["parse_source"] == "fallback" for r in rows),
        "latency_median_s": round(lat[len(lat) // 2], 3) if lat else None,
        "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 3) if lat else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use the Groq API for query understanding instead of the offline parser",
    )
    args = parser.parse_args()

    if args.live:
        _load_env()

    result = run(k=args.k, use_reranker=not args.no_rerank, allow_network=args.live)
    summary = summarise(result)
    print(json.dumps(summary, indent=2))

    if args.verbose:
        print()
        for row in result["rows"]:
            mark = "OK " if row["hit"] else "MISS"
            fam = "" if row["family_ok"] is None else ("  fam:OK" if row["family_ok"] else "  fam:MISS")
            src = row["parse_source"][:5]
            print(
                f"{mark} rank={row['rank']}  [{src:5s}] {row['technique'] or '-':<22.22} "
                f"{row['query'][:44]:<44}{fam}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
