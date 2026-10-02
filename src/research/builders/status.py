"""Check the source-linked preparation packet; never execute a research run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.preparation import inspect_preparation

from . import ROOT


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("research/preparation/manifest.json"))
    parser.add_argument("--target", choices=["model_study", "book_study"], default="model_study")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument(
        "--check-archive",
        action="store_true",
        help="Also inspect optional retained source controls",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        report = inspect_preparation(
            args.root, args.manifest, args.target, check_archive=args.check_archive
        )
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(
            json.dumps(
                {"artifacts_valid": False, "errors": [str(exc)], "execution_performed": False},
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, indent=2))
    if not report["artifacts_valid"]:
        return 1
    return 2 if args.require_ready and not report["ready_for_requested_stage"] else 0
