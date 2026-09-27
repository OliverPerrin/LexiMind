"""Synthetic schema fixtures only: no production query, rater, label or gold data."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.research.annotations import (
    build_annotation_packet,
    canonical_record_bytes,
    read_json,
    sha256,
    validate_future_records,
    validate_judgment,
    validate_packet,
    validate_query,
)

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/prepare_annotation_packet.py"


class AnnotationPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "fixture-books.json"
        self.records = [
            {
                "id": work_id,
                "title": "Synthetic fixture only",
                "description": "PRIVATE FIXTURE BLURB MUST NOT BE COPIED",
                "source": {"url": f"https://example.org/fixture/{work_id}"},
                "descriptionSource": f"https://example.org/fixture/{work_id}",
                "identifiers": {"openLibraryWork": f"/works/{work_id}"},
                "sourceRevision": 1,
                "sourceContentHash": "a" * 64,
            }
            for work_id in ("OL900002W", "OL900001W")
        ]
        self.catalog.write_text(json.dumps(self.records), encoding="utf-8")
        self.rubrics = {
            kind: self.root / f"fixture-{kind}-rubric.md" for kind in ("recommendation", "mood")
        }
        for kind, path in self.rubrics.items():
            path.write_text(f"Draft synthetic {kind} rubric.\n", encoding="utf-8")
        self.packet = build_annotation_packet(self.catalog, self.rubrics, root=self.root)

    def query(self) -> dict:
        item = self.packet["items"][0]
        return {
            "query_id": "query:test-only",
            "query_family_id": "family:test-only",
            "text": "Synthetic fixture query; not collected research data.",
            "intent": "Exercise schema validation only.",
            "constraints": [
                {"constraint_id": "constraint:test-only", "description": "Fixture constraint."}
            ],
            "seed_works": [{"work_id": item["work_id"], "record_sha256": item["record_sha256"]}],
            "catalog_sha256": self.packet["catalog"]["sha256"],
            "rubric_sha256": self.packet["rubrics"]["recommendation"]["sha256"],
            "author_id": "author:test-only",
            "source_kind": "test_fixture",
            "fixture_only": True,
        }

    def judgment(self) -> dict:
        item = self.packet["items"][1]
        return {
            "judgment_id": "judgment:test-only",
            "kind": "recommendation",
            "work_id": item["work_id"],
            "record_sha256": item["record_sha256"],
            "catalog_sha256": self.packet["catalog"]["sha256"],
            "rubric_sha256": self.packet["rubrics"]["recommendation"]["sha256"],
            "rater_id": "rater:test-only",
            "source_kind": "test_fixture",
            "fixture_only": True,
            "rationale": "Synthetic fixture assertion, not a human judgment.",
            "evidence": {
                "scope": "description",
                "reading_coverage": "description_only",
                "source_urls": ["https://example.org/fixture/evidence"],
                "locations": [],
                "edition_id": None,
                "language": "en",
            },
            "query_id": "query:test-only",
            "relevance": 0,
            "abstention_reason": None,
            "constraint_violations": [],
        }

    def mood(self) -> dict:
        row = self.judgment()
        for key in ("query_id", "relevance", "abstention_reason", "constraint_violations"):
            row.pop(key)
        row.update(
            {
                "kind": "mood",
                "rubric_sha256": self.packet["rubrics"]["mood"]["sha256"],
                "mood_observations": ["tense"],
                "evidence_status": "uncertain",
                "eligible_for_whole_work_gold": False,
            }
        )
        row["evidence"].update(
            {
                "scope": "excerpt",
                "reading_coverage": "partial",
                "edition_id": "edition:test-only",
                "locations": ["Synthetic passage only"],
            }
        )
        return row

    def check_judgment(self, row: dict) -> None:
        validate_judgment(row, self.packet, [self.query()], allow_test_fixtures=True)

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(CLI),
                "--catalog",
                str(self.catalog),
                "--recommendation-rubric",
                str(self.rubrics["recommendation"]),
                "--mood-rubric",
                str(self.rubrics["mood"]),
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_packet_is_deterministic_complete_empty_and_does_not_copy_text(self) -> None:
        before = {path: path.read_bytes() for path in self.root.iterdir()}
        self.assertEqual(
            self.packet, build_annotation_packet(self.catalog, self.rubrics, root=self.root)
        )
        self.assertEqual(self.packet["catalog"]["sha256"], sha256(self.catalog.read_bytes()))
        self.assertEqual(self.packet["catalog"]["record_count"], 2)
        self.assertEqual(self.packet["queries"], [])
        self.assertEqual(self.packet["judgments"], [])
        self.assertEqual([i["work_id"] for i in self.packet["items"]], ["OL900001W", "OL900002W"])
        for item in self.packet["items"]:
            source = next(r for r in self.records if r["id"] == item["work_id"])
            self.assertEqual(item["record_sha256"], sha256(canonical_record_bytes(source)))
        encoded = json.dumps(self.packet)
        for forbidden in ("PRIVATE FIXTURE", "Synthetic fixture only", "train", "held_out"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.iterdir()})
        validate_packet(self.packet, self.catalog, self.rubrics, root=self.root)

    def test_exact_catalog_bytes_and_rubric_changes_invalidate_packet(self) -> None:
        self.catalog.write_bytes(self.catalog.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            validate_packet(self.packet, self.catalog, self.rubrics, root=self.root)
        self.packet = build_annotation_packet(self.catalog, self.rubrics, root=self.root)
        self.rubrics["mood"].write_text("Changed draft rubric", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_packet(self.packet, self.catalog, self.rubrics, root=self.root)

    def test_packet_tampering_and_type_substitution_are_rejected(self) -> None:
        for field, value in (
            ("collection_authorized", True),
            ("schema_version", True),
            ("schema_version", 1.0),
        ):
            packet = copy.deepcopy(self.packet)
            packet[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_packet(packet, self.catalog, self.rubrics, root=self.root)
        packet = copy.deepcopy(self.packet)
        packet["items"].pop()
        with self.assertRaises(ValueError):
            validate_query(self.query(), packet, allow_test_fixtures=True)

    def test_duplicate_keys_nonfinite_values_and_duplicate_works_fail_closed(self) -> None:
        for content in ('{"key":1,"key":2}', '{"key":NaN}', '{"key":Infinity}'):
            self.catalog.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_json(self.catalog)
        self.catalog.write_text(json.dumps(self.records + self.records[:1]), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_annotation_packet(self.catalog, self.rubrics, root=self.root)

    def test_plan_has_no_writes_and_create_is_explicit_and_exclusive(self) -> None:
        before = set(self.root.iterdir())
        first = self.run_cli()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, self.run_cli().stdout)
        self.assertEqual(before, set(self.root.iterdir()))
        self.assertEqual(self.run_cli("--mode", "create").returncode, 2)
        output = self.root / "packet.json"
        self.assertEqual(self.run_cli("--output", str(output)).returncode, 2)
        created = self.run_cli("--mode", "create", "--output", str(output))
        self.assertEqual(created.returncode, 0, created.stderr)
        data = output.read_bytes()
        self.assertEqual(self.run_cli("--mode", "create", "--output", str(output)).returncode, 1)
        self.assertEqual(output.read_bytes(), data)
        validated = self.run_cli("--mode", "validate", "--packet", str(output))
        self.assertEqual(validated.returncode, 0, validated.stderr)
        self.assertIn("no gold verified", validated.stdout)
        link = self.root / "symlink.json"
        link.symlink_to(output)
        self.assertEqual(self.run_cli("--mode", "create", "--output", str(link)).returncode, 1)
        self.assertEqual(output.read_bytes(), data)

    def test_null_is_not_zero_and_constraints_are_separate(self) -> None:
        zero = self.judgment()
        zero["constraint_violations"] = ["constraint:test-only"]
        self.check_judgment(zero)
        missing = self.judgment()
        missing["relevance"] = None
        missing["abstention_reason"] = "Synthetic insufficient evidence."
        self.check_judgment(missing)
        missing["abstention_reason"] = None
        with self.assertRaises(ValueError):
            self.check_judgment(missing)
        zero["abstention_reason"] = "Synthetic abstention cannot also be zero."
        with self.assertRaises(ValueError):
            self.check_judgment(zero)

    def test_hard_constraint_violations_cannot_receive_positive_relevance(self) -> None:
        for value in (1, 2, 3):
            row = self.judgment()
            row["relevance"] = value
            row["constraint_violations"] = ["constraint:test-only"]
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "hard-constraint"):
                self.check_judgment(row)
        row["relevance"] = None
        row["abstention_reason"] = "Synthetic overall abstention with known constraint violation."
        self.check_judgment(row)

    def test_invalid_ratings_ids_and_hashes_are_rejected(self) -> None:
        for value in (True, False, 1.0, "1", -1, 4):
            row = self.judgment()
            row["relevance"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check_judgment(row)
        for field, value in (
            ("work_id", "OL999999W"),
            ("work_id", "OL0W"),
            ("record_sha256", "b" * 64),
            ("catalog_sha256", "b" * 64),
            ("rubric_sha256", "b" * 64),
            ("rater_id", "rater: bad"),
            ("query_id", "query:unknown"),
            ("constraint_violations", ["constraint:unknown"]),
        ):
            row = self.judgment()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check_judgment(row)

    def test_origins_and_fixture_flags_do_not_create_gold(self) -> None:
        with self.assertRaises(ValueError):
            validate_query(self.query(), self.packet)
        for kind in ("system_generated", "illustrative", "model", "human"):
            row = self.judgment()
            row["source_kind"] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.check_judgment(row)
        row = self.mood()
        row.update({"evidence_status": "supported", "eligible_for_whole_work_gold": True})
        row["evidence"].update({"scope": "whole_work", "reading_coverage": "complete"})
        with self.assertRaises(ValueError):
            self.check_judgment(row)

    def test_mood_requires_reading_evidence_and_preserves_scope(self) -> None:
        row = self.mood()
        self.check_judgment(row)
        changes = [
            (
                "evidence",
                {**row["evidence"], "scope": "description", "reading_coverage": "description_only"},
            ),
            ("evidence", {**row["evidence"], "scope": "whole_work"}),
            ("evidence", {**row["evidence"], "edition_id": None}),
            ("evidence", {**row["evidence"], "locations": []}),
            ("mood_observations", ["invented-label"]),
            ("eligible_for_whole_work_gold", True),
        ]
        for key, value in changes:
            changed = copy.deepcopy(row)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_judgment(changed)

    def test_duplicate_records_and_missing_provenance_are_rejected(self) -> None:
        query, row = self.query(), self.judgment()
        validate_future_records([query], [row], self.packet, allow_test_fixtures=True)
        for queries, rows in (([query, query], []), ([query], [row, row])):
            with self.assertRaises(ValueError):
                validate_future_records(queries, rows, self.packet, allow_test_fixtures=True)
        other = copy.deepcopy(row)
        other["judgment_id"] = "judgment:other-fixture"
        with self.assertRaises(ValueError):
            validate_future_records([query], [row, other], self.packet, allow_test_fixtures=True)
        other["rater_id"] = "rater:other-fixture"
        validate_future_records([query], [row, other], self.packet, allow_test_fixtures=True)
        for field in ("source_kind", "rater_id", "rubric_sha256", "record_sha256", "evidence"):
            broken = self.judgment()
            broken.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check_judgment(broken)

    def test_query_references_and_constraint_ids_fail_closed(self) -> None:
        for change in ("unknown_seed", "duplicate_seed", "duplicate_constraint", "bad_id", "extra"):
            query = self.query()
            if change == "unknown_seed":
                query["seed_works"][0]["work_id"] = "OL999999W"
            elif change == "duplicate_seed":
                query["seed_works"] *= 2
            elif change == "duplicate_constraint":
                query["constraints"] *= 2
            elif change == "bad_id":
                query["query_id"] = "../unsafe"
            else:
                query["split"] = "test"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_query(query, self.packet, allow_test_fixtures=True)


if __name__ == "__main__":
    unittest.main()
