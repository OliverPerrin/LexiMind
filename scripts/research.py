"""Prepare, inspect and review research evidence without running model training."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A command module exposes configure_parser(parser) and run(args, parser).
# Keep imports lazy: even top-level help must work without the model stack.
COMMANDS = {
    "status": ("status", "Inspect preparation evidence and remaining admission gates"),
    "snapshot": ("snapshot", "Snapshot reviewed preparation files and code hashes"),
    "bgc-source": ("bgc_source", "Audit the pinned BGC source archive"),
    "bgc-groups": ("bgc_groups", "Prepare conservative BGC identity groups"),
    "book-fields": ("book_fields", "Prepare partial book-field label references"),
    "licensed-books": ("licensed_books", "Prepare licensed book text candidates"),
    "bookdash": ("bookdash", "Prepare the expanded licensed Book Dash cohort"),
    "cr4": ("cr4", "Audit raw CR4 character-emotion annotations"),
    "book-groups-review": ("book_groups_review", "Prepare bounded cross-source group evidence"),
    "field-review": ("field_review", "Prepare or validate authored field reviews"),
    "review": ("src.research.review_app", "Build or import the local human-review worksheet"),
    "book-partitions": ("book_partitions", "Prepare the candidate work-group partition overlay"),
    "annotation": ("annotation", "Plan, create or validate a book annotation packet"),
    "ledger": ("ledger", "Validate a proposed or observed compute ledger"),
    "data-audit": ("data_audit", "Audit retained legacy processed data"),
    "ag-news": ("ag_news", "Reproduce the optional AG News source control"),
    "goemotions": ("goemotions", "Reproduce the optional GoEmotions source control"),
    "arxiv": ("arxiv", "Reproduce the optional arXiv source index"),
    "source-partitions": ("source_partitions", "Prepare role assignments for a source control"),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="research.py", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selected = argv[0] if argv else None
    module = None
    command_parser = None
    for command, (module_name, description) in COMMANDS.items():
        subparser = commands.add_parser(command, help=description, description=description)
        if command == selected:
            # Fully qualified modules also support optional UI/integration commands.
            module = import_module(
                module_name if "." in module_name else f"src.research.builders.{module_name}"
            )
            module.configure_parser(subparser)
            command_parser = subparser
    args = parser.parse_args(argv)
    assert module is not None and command_parser is not None
    result: int = module.run(args, command_parser)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
