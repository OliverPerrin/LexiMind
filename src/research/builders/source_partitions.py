"""Prepare source-group assignments only; no models, scoring or network calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.catalog.storage import write_json_atomic
from src.research.partitions import prepare_partitions

from . import ROOT


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    candidate_root = ROOT / "data/research_candidates"
    if not args.output_dir.resolve().is_relative_to(candidate_root):
        parser.error("Assignment indices must remain in ignored data/research_candidates")
    if (
        args.report.resolve().is_relative_to(ROOT / "data")
        or args.report.resolve() == args.candidate_manifest.resolve()
    ):
        parser.error("Report must be separate from source data and the candidate manifest")
    try:
        report = prepare_partitions(ROOT, args.candidate_manifest, args.output_dir)
        write_json_atomic(args.report, report)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "cross_partition_text_groups": report["cross_partition_normalized_text_groups"],
            }
        )
    )
    return 0
