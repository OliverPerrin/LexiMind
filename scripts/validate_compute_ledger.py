"""Validate/report recorded compute; never run profiling, training, or evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.catalog.storage import write_json_atomic
from src.research.ledger import LedgerValidationError, summarize_ledger


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LedgerValidationError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger",
        nargs="?",
        type=Path,
        default=ROOT / "research/preparation/compute_ledger_template.json",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-observed", action="store_true")
    args = parser.parse_args()
    if args.report and args.report.resolve() == args.ledger.resolve():
        parser.error("A report cannot overwrite the source ledger")
    try:
        raw = args.ledger.read_bytes()
        ledger = json.loads(raw, object_pairs_hook=unique_object)
        report = summarize_ledger(ledger)
        report["source_ledger_sha256"] = hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, LedgerValidationError) as error:
        print(json.dumps({"schema_valid": False, "error": str(error)}), file=sys.stderr)
        return 2
    if args.report:
        write_json_atomic(args.report, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if args.require_observed and ledger["status"] != "observed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
