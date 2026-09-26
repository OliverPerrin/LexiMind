"""Synthetic receipt contracts, never collected queries, readers or research gold.

Positive mechanical-contract cases explicitly patch the validator to enable its
test-fixture mode. Production admission has no such bypass and rejects the same
fixtures. Human-review receipt text is explicitly synthetic inside a temp folder.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.research.annotations import build_annotation_packet, sha256, validate_future_records
from src.research.book_admission import validate_book_admission


class BookAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / "fixture-catalog.json"
        records = [
            {
                "id": work,
                "identifiers": {"openLibraryWork": f"/works/{work}"},
                "source": {"url": f"https://example.org/fixture/{work}"},
                "sourceRevision": 1,
                "sourceContentHash": "a" * 64,
            }
            for work in ("OL900001W", "OL900002W")
        ]
        self.catalog.write_text(json.dumps(records), encoding="utf-8")
        rubrics = {kind: self.root / f"{kind}.md" for kind in ("recommendation", "mood")}
        for path in rubrics.values():
            path.write_text("SYNTHETIC DRAFT RUBRIC ONLY", encoding="utf-8")
        self.packet = build_annotation_packet(self.catalog, rubrics, root=self.root)
        self.plan = {
            "study_id": "synthetic-admission-test-only",
            "applied_study": {"generalization": "new_query_families_fixed_catalogue"},
        }
        self.items = {item["work_id"]: item for item in self.packet["items"]}
        queries = [self.query("test", "OL900001W"), self.query("development", "OL900002W")]
        judgments = [self.judgment("one"), self.judgment("two")]
        self.documents = {
            "collection_manifest": self.receipt(
                "book_collection",
                {
                    "queries": queries,
                    "judgments": judgments,
                    "primary_query_ids": ["query:test"],
                },
            ),
            "partition_manifest": self.receipt(
                "book_partitions",
                {
                    "generalization": "new_query_families_fixed_catalogue",
                    "policy": {
                        "status": "frozen",
                        "unit": "query_family_and_seed_work",
                        "seed_work_overlap": "disjoint",
                        "query_family_overlap": "disjoint",
                        "seed_policy": "same_partition_as_query",
                        "candidate_population": "fixed_catalogue",
                    },
                    "catalogue_work_ids": list(self.items),
                    "seed_work_assignments": [
                        {"work_id": "OL900001W", "partition": "test"},
                        {"work_id": "OL900002W", "partition": "development"},
                    ],
                    "query_assignments": [
                        {"query_id": query["query_id"], "partition": partition}
                        for query, partition in zip(queries, ("test", "development"), strict=True)
                    ],
                    "query_family_assignments": [
                        {"query_family_id": query["query_family_id"], "partition": partition}
                        for query, partition in zip(queries, ("test", "development"), strict=True)
                    ],
                },
            ),
            "eligibility_manifest": self.receipt(
                "book_eligibility",
                {
                    "policy": {
                        "status": "frozen",
                        "population": "all_packet_works",
                        "seed_policy": "exclude",
                        "candidate_partition_policy": "shared_catalogue",
                        "other_exclusions": "explicit_reason",
                    },
                    "queries": [
                        {
                            "query_id": query["query_id"],
                            "eligible_work_ids": [other],
                            "excluded_works": [
                                {
                                    "work_id": query["seed_works"][0]["work_id"],
                                    "reason": "seed_work",
                                    "detail": "SYNTHETIC own-seed exclusion only.",
                                }
                            ],
                        }
                        for query, other in zip(queries, ("OL900002W", "OL900001W"), strict=True)
                    ],
                },
            ),
            "rubric_review": self.receipt(
                "book_rubric_review", {"status": "frozen", "attestation": None}
            ),
        }
        self.attestation = self.receipt(
            "book_human_review",
            {
                "source_kind": "human_review",
                "reviewer_id": "reviewer:synthetic-fixture-only",
                "completed_at": "2026-09-26T00:00:00Z",
                "decision": "approved_for_primary_relevance",
                "scope": "metadata_supported_relevance_without_whole_work_mood",
                "review_notes": "SYNTHETIC TEST ATTESTATION. No human review occurred.",
                "reviewed_artifacts": {},
                "reviewed_judgment_ids": [row["judgment_id"] for row in judgments],
                "adjudications": [
                    {
                        "query_id": "query:test",
                        "work_id": "OL900002W",
                        "relevance": 0,
                        "basis_judgment_ids": [row["judgment_id"] for row in judgments],
                        "rationale": "SYNTHETIC contract fixture, not gold relevance.",
                    }
                ],
            },
        )
        self.save_receipts()

    def receipt(self, kind: str, extra: dict) -> dict:
        return {
            "schema_version": 1,
            "kind": kind,
            "purpose": "book_relevance_primary",
            "study_id": self.plan["study_id"],
            "catalog_sha256": self.packet["catalog"]["sha256"],
            "rubric_sha256": {
                key: value["sha256"] for key, value in self.packet["rubrics"].items()
            },
            **extra,
        }

    def query(self, name: str, seed: str) -> dict:
        return {
            "query_id": f"query:{name}",
            "query_family_id": f"family:{name}",
            "text": "Synthetic query fixture only.",
            "intent": "Test schemas, not book quality.",
            "constraints": [],
            "seed_works": [{"work_id": seed, "record_sha256": self.items[seed]["record_sha256"]}],
            "catalog_sha256": self.packet["catalog"]["sha256"],
            "rubric_sha256": self.packet["rubrics"]["recommendation"]["sha256"],
            "author_id": "author:test-fixture",
            "source_kind": "test_fixture",
            "fixture_only": True,
        }

    def judgment(self, name: str) -> dict:
        return {
            "judgment_id": f"judgment:{name}",
            "kind": "recommendation",
            "work_id": "OL900002W",
            "record_sha256": self.items["OL900002W"]["record_sha256"],
            "catalog_sha256": self.packet["catalog"]["sha256"],
            "rubric_sha256": self.packet["rubrics"]["recommendation"]["sha256"],
            "rater_id": f"rater:test-fixture-{name}",
            "source_kind": "test_fixture",
            "fixture_only": True,
            "rationale": "Synthetic fixture only.",
            "evidence": {
                "scope": "metadata",
                "reading_coverage": "metadata_only",
                "source_urls": ["https://example.org/fixture"],
                "locations": [],
                "edition_id": None,
                "language": "en",
            },
            "query_id": "query:test",
            "relevance": 0,
            "abstention_reason": None,
            "constraint_violations": [],
        }

    def write(self, name: str, document: dict) -> dict:
        raw = json.dumps(document, sort_keys=True).encode("utf-8")
        path = self.root / f"{name}.json"
        path.write_bytes(raw)
        return {
            "kind": document["kind"],
            "path": path.name,
            "sha256": sha256(raw),
            "bytes": len(raw),
        }

    def save_receipts(self) -> None:
        for name in ("collection_manifest", "partition_manifest", "eligibility_manifest"):
            self.plan["applied_study"][name] = self.write(name, self.documents[name])
        self.attestation["reviewed_artifacts"] = {
            name: self.plan["applied_study"][name]
            for name in ("collection_manifest", "partition_manifest", "eligibility_manifest")
        }
        self.documents["rubric_review"]["attestation"] = self.write("attestation", self.attestation)
        self.plan["applied_study"]["rubric_review"] = self.write(
            "rubric_review", self.documents["rubric_review"]
        )

    def check_fixture(self) -> list[str]:
        def fixture_validator(queries, judgments, packet):
            return validate_future_records(queries, judgments, packet, allow_test_fixtures=True)

        with patch(
            "src.research.book_admission.validate_future_records", side_effect=fixture_validator
        ) as checker:
            result = validate_book_admission(self.root, self.plan, self.packet)
            checker.assert_called_once()
            return result

    def test_production_rejects_fixtures_but_mechanical_path_accepts_separate_receipts(
        self,
    ) -> None:
        self.assertTrue(validate_book_admission(self.root, self.plan, self.packet))
        before = copy.deepcopy(self.packet)
        self.assertEqual(self.check_fixture(), [])
        self.assertEqual(self.packet, before)
        self.assertEqual(self.packet["queries"], [])
        self.assertEqual(self.packet["judgments"], [])
        # The test query legitimately retrieves the work used as a development seed.
        self.assertEqual(
            self.documents["eligibility_manifest"]["queries"][0]["eligible_work_ids"], ["OL900002W"]
        )

    def test_boolean_flags_neither_create_evidence_nor_block_a_complete_receipt(self) -> None:
        for flag in (False, True):
            self.plan["applied_study"]["judgments_collected"] = flag
            self.assertEqual(self.check_fixture(), [])
        for name in self.documents:
            self.plan["applied_study"][name] = None
        blockers = validate_book_admission(self.root, self.plan, self.packet)
        self.assertEqual(len(blockers), 4)

    def test_empty_packet_cannot_be_repurposed_as_collection(self) -> None:
        self.packet["queries"] = [self.query("test", "OL900001W")]
        self.assertTrue(validate_book_admission(self.root, self.plan, self.packet))

    def test_hash_kind_purpose_study_and_catalogue_linkage(self) -> None:
        original = copy.deepcopy(self.documents["collection_manifest"])
        for key, value in (
            ("kind", "compute_ledger"),
            ("purpose", "other"),
            ("study_id", "other"),
            ("catalog_sha256", "b" * 64),
        ):
            self.documents["collection_manifest"] = {**original, key: value}
            self.save_receipts()
            with self.subTest(key=key):
                self.assertTrue(validate_book_admission(self.root, self.plan, self.packet))
        self.documents["collection_manifest"] = original
        self.save_receipts()
        (self.root / "collection_manifest.json").write_bytes(b"{}")
        self.assertTrue(validate_book_admission(self.root, self.plan, self.packet))

    def test_generalization_must_be_explicit_and_full_catalogue_tracked(self) -> None:
        self.plan["applied_study"]["generalization"] = "unseen_candidate_works"
        self.assertTrue(self.check_fixture())
        self.plan["applied_study"]["generalization"] = "new_query_families_fixed_catalogue"
        self.documents["partition_manifest"]["catalogue_work_ids"].pop()
        self.save_receipts()
        self.assertTrue(self.check_fixture())

    def test_query_family_cannot_cross_partitions(self) -> None:
        queries = self.documents["collection_manifest"]["queries"]
        queries[1]["query_family_id"] = queries[0]["query_family_id"]
        self.documents["partition_manifest"]["query_family_assignments"].pop()
        self.save_receipts()
        self.assertTrue(self.check_fixture())

    def test_seed_work_cannot_cross_query_partitions(self) -> None:
        queries = self.documents["collection_manifest"]["queries"]
        queries[1]["seed_works"] = copy.deepcopy(queries[0]["seed_works"])
        self.documents["partition_manifest"]["seed_work_assignments"].pop()
        self.save_receipts()
        self.assertTrue(self.check_fixture())

    def test_identical_query_content_cannot_hide_behind_different_family_ids(self) -> None:
        queries = self.documents["collection_manifest"]["queries"]
        for query in queries:
            query["seed_works"] = []
        self.documents["partition_manifest"]["seed_work_assignments"] = []
        for row in self.documents["eligibility_manifest"]["queries"]:
            for exclusion in row["excluded_works"]:
                exclusion["reason"] = "insufficient_evidence"
        # IDs, whitespace, and constraint ordering cannot change content identity.
        queries[0]["constraints"] = [
            {"constraint_id": "constraint:first", "description": "Synthetic requirement one."},
            {"constraint_id": "constraint:second", "description": "Synthetic requirement two."},
        ]
        queries[1]["constraints"] = [
            {"constraint_id": "constraint:other-two", "description": "Synthetic requirement two."},
            {"constraint_id": "constraint:other-one", "description": "Synthetic  requirement one."},
        ]
        queries[1]["text"] = queries[1]["text"].replace("query fixture", "query  fixture")
        self.save_receipts()
        blockers = self.check_fixture()
        self.assertTrue(any("query content crosses partitions" in blocker for blocker in blockers))

    def test_adjudication_cannot_ignore_a_retained_constraint_violation(self) -> None:
        query = self.documents["collection_manifest"]["queries"][0]
        query["constraints"] = [
            {"constraint_id": "constraint:required", "description": "Synthetic hard requirement."}
        ]
        judgments = self.documents["collection_manifest"]["judgments"]
        extra = copy.deepcopy(judgments[0])
        extra.update(
            judgment_id="judgment:third",
            rater_id="rater:test-fixture-third",
            constraint_violations=["constraint:required"],
        )
        judgments.append(extra)
        self.attestation["reviewed_judgment_ids"].append("judgment:third")
        # The contradictory judgment is reviewed, but omitted from the basis list.
        for value in (1, 2, 3):
            self.attestation["adjudications"][0]["relevance"] = value
            self.save_receipts()
            with self.subTest(value=value):
                self.assertTrue(
                    any("hard-constraint violations" in blocker for blocker in self.check_fixture())
                )
        self.attestation["adjudications"][0]["relevance"] = 0
        self.save_receipts()
        self.assertEqual(self.check_fixture(), [])

    def test_eligibility_is_complete_and_excludes_own_seed(self) -> None:
        row = self.documents["eligibility_manifest"]["queries"][0]
        original = copy.deepcopy(row)
        row["eligible_work_ids"] = []
        self.save_receipts()
        self.assertTrue(self.check_fixture())
        row.update(original)
        row["eligible_work_ids"].append("OL900001W")
        row["excluded_works"] = []
        self.save_receipts()
        self.assertTrue(self.check_fixture())

    def test_primary_coverage_and_null_are_not_silently_zero(self) -> None:
        row = self.documents["collection_manifest"]["judgments"][1]
        row["relevance"] = None
        row["abstention_reason"] = "Synthetic insufficient evidence."
        self.save_receipts()
        self.assertTrue(self.check_fixture())
        row["relevance"], row["abstention_reason"] = 0, None
        for value in (None, False, 0.0):
            self.attestation["adjudications"][0]["relevance"] = value
            self.save_receipts()
            self.assertTrue(self.check_fixture())
        self.attestation["adjudications"] = []
        self.save_receipts()
        self.assertTrue(self.check_fixture())

    def test_review_needs_typed_source_all_judgments_and_two_raters(self) -> None:
        self.attestation["reviewed_judgment_ids"].pop()
        self.save_receipts()
        self.assertTrue(self.check_fixture())
        self.attestation["reviewed_judgment_ids"].append("judgment:two")
        self.attestation["adjudications"][0]["basis_judgment_ids"].pop()
        self.save_receipts()
        self.assertTrue(self.check_fixture())
        self.attestation["adjudications"][0]["basis_judgment_ids"].append("judgment:two")
        self.documents["rubric_review"]["attestation"]["kind"] = "generic_approval"
        self.plan["applied_study"]["rubric_review"] = self.write(
            "rubric_review", self.documents["rubric_review"]
        )
        self.assertTrue(self.check_fixture())

    def test_refs_cannot_escape_root_or_use_boolean_lengths(self) -> None:
        reference = self.plan["applied_study"]["collection_manifest"]
        original = copy.deepcopy(reference)
        for key, value in (("path", "../outside.json"), ("path", "/outside.json"), ("bytes", True)):
            reference.update(original)
            reference[key] = value
            self.assertTrue(validate_book_admission(self.root, self.plan, self.packet))


if __name__ == "__main__":
    unittest.main()
