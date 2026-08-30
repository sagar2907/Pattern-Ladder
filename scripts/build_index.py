"""Build all cached artefacts. Run once before serving or testing end-to-end.

With --describe, each discovered family also gets a one-line description
written by the model. That is an offline, one-time cost of roughly 137 calls;
no query ever waits on it, and without an API key the build produces the same
artefacts without that one optional field.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pattern_ladder.index.build import build_all  # noqa: E402


def _load_env() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--describe",
        action="store_true",
        help="also write a one-line description per family using the model",
    )
    args = parser.parse_args()

    progress = None
    if args.describe:
        _load_env()

        def progress(done, total, family, description):  # noqa: ARG001
            status = description or "(kept tag name)"
            print(f"  [{done:3d}/{total}] {family.name[:38]:38s} -> {status}", file=sys.stderr)

    manifest = build_all(describe_families=args.describe, progress=progress)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
