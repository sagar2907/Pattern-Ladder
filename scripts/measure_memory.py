"""Measure resident memory against the deployment ceiling.

The free hosting tier this targets (Streamlit Community Cloud) allows roughly
1 GB per app. The two models are small, but PyTorch is not, and "it should
fit" is not a measurement. This reports the resident set at each stage so it is
clear which component costs what.

Note what this does *not* include: Streamlit's own server process. The number
below is the engine in a bare Python process, so the real figure under the app
is higher. Treat the headroom as smaller than it looks.

Run: python scripts/measure_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# The ceiling this is measured against, in megabytes.
DEPLOY_CEILING_MB = 1024


def main() -> int:
    try:
        import psutil
    except ImportError:
        print("psutil is required: uv sync --extra dev")
        return 1

    process = psutil.Process()

    def rss() -> float:
        return process.memory_info().rss / 1e6

    stages = [("baseline python", rss())]

    from pattern_ladder.engine import load_engine
    from pattern_ladder.understand.groq_client import understand

    stages.append(("after imports", rss()))

    engine = load_engine()
    stages.append(("after indexes loaded", rss()))

    query = "I keep failing problems where you shrink a window from the left"
    engine.search(query, understand(query, allow_network=False), top_k=10)
    stages.append(("after first query (models resident)", rss()))

    for follow_up in ("next greater element", "binary search", "linked list cycle"):
        engine.search(follow_up, understand(follow_up, allow_network=False), top_k=10)
    peak = rss()
    stages.append(("after four queries", peak))

    width = max(len(name) for name, _ in stages)
    for name, value in stages:
        print(f"{name:<{width}}  {value:7.0f} MB")

    print()
    print(f"ceiling {DEPLOY_CEILING_MB} MB, headroom {DEPLOY_CEILING_MB - peak:.0f} MB")
    print("excludes the Streamlit server process; real headroom is smaller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
