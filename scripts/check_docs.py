"""Check the numbers quoted in the documentation against the built artefacts.

The README opens by saying every number in it was measured by a script in this
repository. That is a promise worth being able to verify mechanically, because
prose drifts: the test count was quoted twice, in two different places, and
both were wrong -- one by a single test, the other by sixty-eight -- while
every measured figure around them was correct.

Numbers that come from the manifest are compared to the manifest. The test
count is obtained by running the suite. Anything that cannot be derived
automatically is not checked here, and is not claimed to be.

Exits non-zero on the first mismatch so it can gate a release.

Run: python scripts/check_docs.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "docs" / "report.md"]


def _fmt(value: float | int) -> list[str]:
    """Renderings a number might plausibly take in prose."""
    if isinstance(value, int):
        return [f"{value:,}", str(value)]
    return [f"{value:.4f}", f"{value:.3f}", f"{value:.2f}", f"{value * 100:.1f}%"]


def _load_manifest() -> dict | None:
    path = ROOT / "artifacts" / "index" / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _test_count() -> int | None:
    """Run the suite and read the count back, rather than trusting a comment."""
    # No -q here. pyproject already sets it in addopts, and passing it again
    # gives -qq, which suppresses the very summary line this parses -- the
    # reason a first version of this check reported "could not determine the
    # test count" while the suite was passing perfectly well.
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "pytest", str(ROOT / "tests")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    match = re.search(r"(\d+) passed", result.stdout + result.stderr)
    return int(match.group(1)) if match else None


def _stale_pdfs() -> list[str]:
    """Re-render every document and compare it to the committed PDF.

    This is only possible because the renderer is deterministic: invariant mode
    fixes the timestamp and derives the document ID from the content, so an
    unchanged document renders to identical bytes. Before that, every PDF
    differed from every other render of itself and this check could not have
    existed.

    It answers the question a diff cannot: whether the committed PDF was built
    from the committed markdown. Editing the report and forgetting to re-render
    is otherwise invisible -- the source is right, the PDF is wrong, and the
    only symptom is a reader receiving a document that disagrees with the
    repository.
    """
    import runpy
    import tempfile

    module = runpy.run_path(str(ROOT / "scripts" / "render_pdf.py"), run_name="__checked__")
    build_pdf = module["build_pdf"]
    default_output_path = module["default_output_path"]

    stale: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for source in sorted((ROOT / "docs").glob("*.md")):
            committed = default_output_path(source)
            if not committed.is_file():
                stale.append(f"{source.name}: {committed.name} is missing")
                continue
            title = source.stem.replace("-", " ").replace("_", " ").title()
            fresh = Path(tmp) / committed.name
            build_pdf(source.read_text(encoding="utf-8"), fresh, title)
            if fresh.read_bytes() != committed.read_bytes():
                stale.append(
                    f"{committed.name} does not match a fresh render of {source.name}"
                    " -- re-run scripts/render_pdf.py"
                )
    return stale


def main() -> int:
    text = "\n".join(d.read_text(encoding="utf-8") for d in DOCS if d.is_file())
    problems: list[str] = []

    manifest = _load_manifest()
    if manifest is None:
        print("no manifest; run scripts/build_index.py first", file=sys.stderr)
        return 2

    families = manifest["families"]
    graph = manifest["graph"]
    checks: list[tuple[str, int | float]] = [
        ("corpus size", manifest["corpus"]["problems"]),
        ("curated edges", graph["link_only"]["edges"]),
        ("final edges", graph["final"]["edges"]),
        ("inferred edges", graph["final"]["knn_edges"]),
        ("isolated before backfill", graph["link_only"]["isolated"]),
        ("isolated after backfill", graph["final"]["isolated"]),
        ("family count", families["count"]),
        ("largest family", families["largest"]),
        ("families described", families["described_by_model"]),
        ("families with related", families["with_related_families"]),
        ("duplicate names", families["duplicate_names"]),
        ("coverage", families["coverage_fraction"]),
        ("tag agreement", families["tag_independence"]["nmi"]),
        ("name coherence", families["name_coherence"]["coherence"]),
    ]

    for label, value in checks:
        if not any(rendering in text for rendering in _fmt(value)):
            problems.append(f"{label}: manifest says {value}, not found in the docs")

    problems.extend(_stale_pdfs())

    counted = _test_count()
    if counted is None:
        problems.append("could not determine the test count")
    else:
        quoted = {int(n) for n in re.findall(r"(\d+) tests passing", text)}
        quoted |= {int(n) for n in re.findall(r"\|\s*Tests\s*\|\s*(\d+)", text)}
        wrong = {n for n in quoted if n != counted}
        if wrong:
            problems.append(
                f"test count: suite reports {counted}, docs quote {sorted(wrong)}"
            )
        elif not quoted:
            problems.append(f"test count: suite reports {counted}, docs quote none")

    if problems:
        print("documentation is out of step with the artefacts:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    rendered = len(list((ROOT / "docs").glob("*.md")))
    print(
        f"all {len(checks)} manifest figures and the test count appear in the docs; "
        f"{rendered} committed PDFs match a fresh render"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
