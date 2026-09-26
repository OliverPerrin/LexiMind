"""Synthetic arXiv-source fixtures only; no network, models, or research scoring."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_arxiv_candidate as arxiv


def fixture(
    identifier: str, article: str = "SYNTHETIC SOURCE TEXT", abstract: str = "SYNTHETIC ABSTRACT"
) -> dict:
    return {
        "article_id": identifier,
        "article_text": [article, "A second synthetic sentence."],
        "abstract_text": [abstract],
        "section_names": ["Fixture section"],
        "sections": [[article, "A second synthetic sentence."]],
    }


class Response(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int = 200, headers: dict | None = None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}
        self.url = arxiv.ARCHIVE_URL


class ArxivPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "fixture.zip"
        self.candidate = self.root / "candidate"
        self.records = {
            "train": [fixture("1001.0001v1"), fixture("1001.0002", "Repeated synthetic source")],
            "val": [fixture("1001.0001v2")],
            "test": [fixture("hep-th/9901001", "Repeated synthetic source")],
        }

    def archive(
        self, records: dict | None = None, *, extras: dict | None = None
    ) -> dict[str, bytes]:
        source = self.records if records is None else records
        raw = {
            split: b"".join((json.dumps(row, ensure_ascii=False) + "\r\n").encode() for row in rows)
            for split, rows in source.items()
        }
        with zipfile.ZipFile(self.source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for split, value in raw.items():
                archive.writestr(f"arxiv-dataset/{split}.txt", value)
            archive.writestr("arxiv-dataset/vocab", "fixture vocabulary")
            for name, value in (extras or {}).items():
                archive.writestr(name, value)
        return raw

    def test_stream_preserves_ids_splits_offsets_and_source_bytes_without_copying_text(
        self,
    ) -> None:
        raw = self.archive()
        source_before = self.source.read_bytes()
        audit = arxiv.index_archive(self.source, self.candidate)
        self.assertEqual(audit["rows"], 4)
        self.assertEqual(set(audit["splits"]), {"train", "val", "test"})
        for split, expected in self.records.items():
            index_path = self.candidate / "index" / f"{split}.jsonl"
            indexed = [json.loads(line) for line in index_path.read_text().splitlines()]
            self.assertEqual(
                [row["article_id"] for row in indexed], [row["article_id"] for row in expected]
            )
            for row in indexed:
                offset, size = row["uncompressed_byte_offset"], row["source_line_bytes"]
                original = raw[split][offset : offset + size]
                self.assertEqual(arxiv.sha(original), row["source_line_sha256"])
                self.assertEqual(json.loads(original)["article_id"], row["article_id"])
                self.assertNotIn("article_text", row)
            self.assertEqual(audit["splits"][split]["uncompressed_sha256"], arxiv.sha(raw[split]))
            self.assertNotIn("SYNTHETIC SOURCE TEXT", index_path.read_text())
            self.assertNotIn("SYNTHETIC ABSTRACT", index_path.read_text())
        self.assertEqual(source_before, self.source.read_bytes())
        self.assertFalse(list(self.candidate.glob(".arxiv-audit-*")))

    def test_version_and_duplicate_audits_do_not_reassign_or_drop_rows(self) -> None:
        self.archive()
        audit = arxiv.index_archive(self.source, self.candidate)
        groups = audit["duplicate_groups"]
        self.assertEqual(audit["unique_original_article_ids"], 4)
        self.assertEqual(audit["unique_recognized_arxiv_base_ids"], 3)
        self.assertEqual(groups["article_id"]["groups"], 0)
        self.assertEqual(groups["arxiv_base_id"]["cross_split_groups"], 1)
        self.assertEqual(groups["article_hash"]["cross_split_groups"], 2)
        self.assertEqual(audit["splits"]["train"]["flags"]["version_not_stated"], 1)
        test = json.loads((self.candidate / "index/test.jsonl").read_text())
        self.assertEqual(test["article_id"], "hep-th/9901001")
        self.assertEqual(test["source_url"], "https://arxiv.org/abs/hep-th/9901001")
        self.assertIsNone(test["article_version"])

    def test_normalized_duplicate_audit_preserves_case_and_punctuation(self) -> None:
        self.archive(
            {
                "train": [fixture("1001.0001", "Café   fixture")],
                "val": [fixture("1001.0002", "Cafe\u0301 fixture")],
                "test": [fixture("1001.0003", "café fixture!")],
            }
        )
        groups = arxiv.index_archive(self.source, self.candidate)["duplicate_groups"]
        self.assertEqual(groups["article_hash"]["groups"], 0)
        self.assertEqual(
            groups["normalized_hash"], {"groups": 1, "rows": 2, "cross_split_groups": 1}
        )

    def test_repeated_preparation_verifies_outputs_and_refuses_changed_existing_index(self) -> None:
        self.archive()
        first = arxiv.index_archive(self.source, self.candidate)
        self.assertEqual(first, arxiv.index_archive(self.source, self.candidate))
        index = self.candidate / "index/train.jsonl"
        index.write_bytes(b"existing source evidence must survive\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            arxiv.index_archive(self.source, self.candidate)
        self.assertEqual(index.read_bytes(), b"existing source evidence must survive\n")

    def test_missing_id_and_duplicate_json_keys_fail_without_invented_rows(self) -> None:
        row = fixture("1001.0001")
        row.pop("article_id")
        self.archive(
            {"train": [row], "val": [fixture("1001.0002")], "test": [fixture("1001.0003")]}
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            arxiv.index_archive(self.source, self.candidate)
        self.assertFalse((self.candidate / "index/train.jsonl").exists())
        with zipfile.ZipFile(self.source, "w") as archive:
            for split in arxiv.SPLITS:
                archive.writestr(f"{split}.txt", '{"article_id":"one","article_id":"two"}\n')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            arxiv.index_archive(self.source, self.candidate)

    def test_unrecognized_provider_identifier_is_preserved_and_marked_unresolved(self) -> None:
        row = fixture("provider-fixture-not-a-valid-arxiv-id")
        raw = json.dumps(row).encode()
        result = arxiv.record_metadata(row, "train", 1, 0, raw)
        self.assertEqual(result["article_id"], row["article_id"])
        self.assertIsNone(result["arxiv_base_id"])
        self.assertIsNone(result["source_url"])
        self.assertIn("unresolved_arxiv_identifier_format", result["flags"])

    def test_empty_text_and_section_disagreement_are_retained_audit_flags(self) -> None:
        row = fixture("1001.0001")
        row.update({"article_text": [], "abstract_text": [], "section_names": []})
        result = arxiv.record_metadata(row, "train", 1, 0, json.dumps(row).encode())
        self.assertEqual(
            set(result["flags"]),
            {
                "empty_article",
                "empty_abstract",
                "section_name_count_mismatch",
                "section_sentence_count_mismatch",
            },
        )

    def test_member_traversal_and_duplicate_split_files_are_rejected(self) -> None:
        for extras in ({"../unsafe.txt": "fixture"}, {"different/train.txt": "fixture"}):
            self.archive(extras=extras)
            with self.subTest(extras=extras), self.assertRaises(ValueError):
                arxiv.index_archive(self.source, self.candidate)
        self.archive()
        with zipfile.ZipFile(self.source, "a") as archive:
            link = zipfile.ZipInfo("link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, "outside")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            arxiv.index_archive(self.source, self.candidate)

    def test_line_row_and_expansion_limits_are_enforced(self) -> None:
        self.archive()
        for bound, value in (
            ("MAX_LINE_BYTES", 20),
            ("MAX_ROWS", 1),
            ("MAX_UNCOMPRESSED_BYTES", 10),
        ):
            with (
                self.subTest(bound=bound),
                patch.object(arxiv, bound, value),
                self.assertRaises(ValueError),
            ):
                arxiv.index_archive(self.source, self.candidate)

    def test_empty_split_fails_before_any_empty_index_is_published(self) -> None:
        self.archive({"train": [], "val": [fixture("1001.0002")], "test": [fixture("1001.0003")]})
        with self.assertRaisesRegex(ValueError, "empty"):
            arxiv.index_archive(self.source, self.candidate)
        self.assertFalse((self.candidate / "index/train.jsonl").exists())

    def test_no_fetch_does_not_open_network_or_create_a_source(self) -> None:
        with patch.object(arxiv, "urlopen") as network, self.assertRaises(FileNotFoundError):
            arxiv.acquire_archive(self.candidate, fetch=False, progress=lambda event: None)
        network.assert_not_called()
        self.assertFalse(self.candidate.exists())

    def test_download_pins_declared_and_observed_checksums_separately_and_reuses_source(
        self,
    ) -> None:
        payload = b"synthetic immutable archive fixture bytes"
        expected = {name: hashlib.new(name, payload).hexdigest() for name in ("sha1", "md5")}
        with (
            patch.object(arxiv, "EXPECTED_BYTES", len(payload)),
            patch.object(arxiv, "PROVIDER_CHECKSUMS", expected),
        ):
            with patch.object(arxiv, "urlopen", return_value=Response(payload)) as network:
                receipt = arxiv.acquire_archive(
                    self.candidate, fetch=True, progress=lambda event: None
                )
                network.assert_called_once()
            self.assertEqual(receipt["provider_declared_checksums"], expected)
            self.assertEqual(receipt["observed_checksums"]["sha256"], arxiv.sha(payload))
            with patch.object(arxiv, "urlopen") as network:
                self.assertEqual(
                    receipt,
                    arxiv.acquire_archive(self.candidate, fetch=False, progress=lambda event: None),
                )
                network.assert_not_called()
            archive = self.candidate / "raw/arxiv-dataset.zip"
            archive.write_bytes(b"x" * len(payload))
            with self.assertRaisesRegex(ValueError, "checksums"):
                arxiv.acquire_archive(self.candidate, fetch=False, progress=lambda event: None)

    def test_ignored_range_never_publishes_source(self) -> None:
        raw = self.candidate / "raw"
        raw.mkdir(parents=True)
        partial = raw / "arxiv-dataset.zip.partial"
        partial.write_bytes(b"prefix")
        with (
            patch.object(arxiv, "EXPECTED_BYTES", 20),
            patch.object(arxiv, "urlopen", return_value=Response(b"ignored-range")),
        ):
            with self.assertRaisesRegex(ValueError, "range"):
                arxiv.acquire_archive(self.candidate, fetch=True, progress=lambda event: None)
        self.assertEqual(partial.read_bytes(), b"prefix")
        self.assertFalse((raw / "arxiv-dataset.zip").exists())

    def test_empty_partial_is_reused_without_retry_or_overwrite_failure(self) -> None:
        raw = self.candidate / "raw"
        raw.mkdir(parents=True)
        (raw / "arxiv-dataset.zip.partial").touch()
        payload = b"synthetic complete source"
        expected = {name: hashlib.new(name, payload).hexdigest() for name in ("sha1", "md5")}
        with (
            patch.object(arxiv, "EXPECTED_BYTES", len(payload)),
            patch.object(arxiv, "PROVIDER_CHECKSUMS", expected),
            patch.object(arxiv, "urlopen", return_value=Response(payload)) as network,
        ):
            receipt = arxiv.acquire_archive(self.candidate, fetch=True, progress=lambda event: None)
        network.assert_called_once()
        self.assertEqual(receipt["observed_checksums"]["sha256"], arxiv.sha(payload))
        self.assertEqual((raw / "arxiv-dataset.zip").read_bytes(), payload)
        self.assertFalse((raw / "arxiv-dataset.zip.partial").exists())

    def test_resume_rejects_inconsistent_range_total_before_writing(self) -> None:
        raw = self.candidate / "raw"
        raw.mkdir(parents=True)
        partial = raw / "arxiv-dataset.zip.partial"
        partial.write_bytes(b"prefix")
        response = Response(
            b"suffix", status=206, headers={"Content-Length": "6", "Content-Range": "bytes 6-11/13"}
        )
        with (
            patch.object(arxiv, "EXPECTED_BYTES", 12),
            patch.object(arxiv, "urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(ValueError, "range"):
                arxiv.acquire_archive(self.candidate, fetch=True, progress=lambda event: None)
        self.assertEqual(partial.read_bytes(), b"prefix")
        self.assertFalse((raw / "arxiv-dataset.zip").exists())

    def test_wrong_response_length_never_publishes_source(self) -> None:
        raw = self.candidate / "raw"
        with (
            patch.object(arxiv, "EXPECTED_BYTES", 20),
            patch.object(arxiv, "urlopen", return_value=Response(b"too-short")),
        ):
            with self.assertRaisesRegex(ValueError, "length"):
                arxiv.acquire_archive(self.candidate, fetch=True, progress=lambda event: None)
        self.assertFalse((raw / "arxiv-dataset.zip").exists())

    def test_partial_parallel_source_is_never_used_or_downloaded_again(self) -> None:
        raw = self.candidate / "raw"
        raw.mkdir(parents=True)
        (raw / "arxiv-dataset.zip.ranged.partial").write_bytes(b"staged only")
        with (
            patch.object(arxiv, "urlopen") as network,
            self.assertRaisesRegex(ValueError, "incomplete"),
        ):
            arxiv.acquire_archive(self.candidate, fetch=True, progress=lambda event: None)
        network.assert_not_called()

    def test_aliased_or_nonregular_partial_never_changes_external_sentinel(self) -> None:
        raw = self.candidate / "raw"
        raw.mkdir(parents=True)
        protected = self.root / "protected-fixture.txt"
        protected.write_bytes(b"prefix")
        partial = raw / "arxiv-dataset.zip.partial"
        for kind in ("symlink", "hardlink", "directory"):
            if kind == "symlink":
                partial.symlink_to(protected)
            elif kind == "hardlink":
                os.link(protected, partial)
            else:
                partial.mkdir()
            with self.subTest(kind=kind), patch.object(arxiv, "urlopen") as network:
                with self.assertRaises(ValueError):
                    arxiv.acquire_archive(self.candidate, fetch=True, progress=lambda event: None)
                network.assert_not_called()
                self.assertEqual(protected.read_bytes(), b"prefix")
            partial.rmdir() if kind == "directory" else partial.unlink()

    def test_raw_directory_escape_is_rejected_before_network_or_write(self) -> None:
        self.candidate.mkdir()
        outside = self.root / "outside-fixture"
        outside.mkdir()
        (self.candidate / "raw").symlink_to(outside, target_is_directory=True)
        with patch.object(arxiv, "urlopen") as network, self.assertRaises(ValueError):
            arxiv.acquire_archive(self.candidate, fetch=True, progress=lambda event: None)
        network.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_partial_alias_introduced_during_request_is_not_followed(self) -> None:
        raw = self.candidate / "raw"
        raw.mkdir(parents=True)
        partial = raw / "arxiv-dataset.zip.partial"
        partial.write_bytes(b"prefix")
        protected = self.root / "protected-fixture.txt"
        protected.write_bytes(b"prefix")

        def response_after_swap(*args, **kwargs):
            partial.unlink()
            partial.symlink_to(protected)
            return Response(
                b"suffix",
                status=206,
                headers={"Content-Length": "6", "Content-Range": "bytes 6-11/12"},
            )

        with (
            patch.object(arxiv, "EXPECTED_BYTES", 12),
            patch.object(arxiv, "urlopen", side_effect=response_after_swap) as network,
        ):
            with self.assertRaises(ValueError):
                arxiv.acquire_archive(self.candidate, fetch=True, progress=lambda event: None)
        network.assert_called_once()
        self.assertEqual(protected.read_bytes(), b"prefix")
        self.assertFalse((raw / "arxiv-dataset.zip").exists())

    def test_index_and_document_directory_aliases_are_not_followed(self) -> None:
        self.archive()
        self.candidate.mkdir()
        outside = self.root / "outside-fixture"
        outside.mkdir()
        (self.candidate / "index").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            arxiv.index_archive(self.source, self.candidate)
        (self.candidate / "index").unlink()
        (self.candidate / "raw").mkdir()
        (self.candidate / "raw/documents").symlink_to(outside, target_is_directory=True)
        with patch.object(arxiv, "urlopen") as network, self.assertRaises(ValueError):
            arxiv.provenance_documents(self.candidate, fetch=True)
        network.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_original_record_types_are_strict(self) -> None:
        for field, value in (
            ("article_id", None),
            ("article_text", "not an array"),
            ("abstract_text", [False]),
            ("sections", ["not an array"]),
        ):
            row = copy.deepcopy(fixture("1001.0001"))
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                arxiv.record_metadata(row, "train", 1, 0, json.dumps(row).encode())


if __name__ == "__main__":
    unittest.main()
