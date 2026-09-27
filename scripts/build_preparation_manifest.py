"""Explicitly snapshot reviewed preparation artifacts; default refuses replacement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_research_artifacts import safe_path
from src.research.manifest import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", default="research/preparation/manifest.json")
    parser.add_argument(
        "--replace", action="store_true", help="Replace only after reviewing changed evidence"
    )
    args = parser.parse_args()
    try:
        target = safe_path(args.root, args.output)
        value = json.dumps(build_manifest(args.root), indent=2, allow_nan=False) + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w" if args.replace else "x", encoding="utf-8") as handle:
            handle.write(value)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Preparation snapshot written: {target}; no research stage admitted or executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
