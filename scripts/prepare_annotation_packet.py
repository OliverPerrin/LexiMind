"""Prepare metadata references without collecting labels or evaluating models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.annotations import build_annotation_packet, read_json, validate_packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "create", "validate"], default="plan")
    parser.add_argument("--catalog", type=Path, default=ROOT / "web/data/books.json")
    parser.add_argument(
        "--recommendation-rubric", type=Path, default=ROOT / "docs/recommendation_judgments.md"
    )
    parser.add_argument("--mood-rubric", type=Path, default=ROOT / "docs/mood_annotation_guide.md")
    parser.add_argument(
        "--output", type=Path, help="Required for create; destination must not exist"
    )
    parser.add_argument("--packet", type=Path, help="Required for validate; read-only")
    args = parser.parse_args(argv)
    if args.mode == "create" and args.output is None:
        parser.error("create requires an explicit --output")
    if args.mode == "validate" and args.packet is None:
        parser.error("validate requires --packet")
    if args.mode != "create" and args.output is not None:
        parser.error("--output is only allowed in create mode")
    if args.mode != "validate" and args.packet is not None:
        parser.error("--packet is only allowed in validate mode")
    rubrics = {"recommendation": args.recommendation_rubric, "mood": args.mood_rubric}
    try:
        if args.mode == "validate":
            validate_packet(read_json(args.packet), args.catalog, rubrics, root=ROOT)
            print(
                "Preparation packet matches source bytes; human review still required; no gold verified."
            )
            return 0
        packet = build_annotation_packet(args.catalog, rubrics, root=ROOT)
        encoded = (json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        if args.mode == "plan":
            sys.stdout.buffer.write(encoded)
        else:
            # Exclusive creation also refuses pre-existing symlinks and avoids overwrite races.
            descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
            print(
                f"Created unlabelled packet with {len(packet['items'])} work references: {args.output}"
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Annotation preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
