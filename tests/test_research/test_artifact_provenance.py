"""No ML imports: verify evidence preservation, report handling and pause gates."""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.audit_research_artifacts import (
    DEFAULT_MANIFEST,
    ROOT,
    audit_archive,
    load_json,
    preparation_blockers,
    safe_path,
)
from scripts.build_tables import historical_rows, latex_escape, metric_at, render


class ResearchProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = load_json(ROOT / DEFAULT_MANIFEST)
        for artifact in self.manifest["artifacts"]:
            output = self.root / artifact["path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / artifact["path"], output)

    def test_archive_is_self_contained_without_untracked_checkpoints(self):
        self.assertEqual(audit_archive(self.root, self.manifest), [])
        self.assertTrue(audit_archive(self.root, self.manifest, check_local=True))

    def test_changed_report_bytes_block_table_generation(self):
        path = self.root / self.manifest["artifacts"][0]["path"]
        original = path.read_text()
        # A valid JSON metric change must be detected even if byte length stays equal.
        changed = original.replace("0.30640212017899815", "0.90640212017899815")
        self.assertNotEqual(original, changed)
        path.write_text(changed)
        self.assertTrue(
            any("SHA-256 changed" in error for error in audit_archive(self.root, self.manifest))
        )
        with self.assertRaisesRegex(ValueError, "Archive verification failed"):
            historical_rows(self.root, self.manifest)

    def test_missing_evidence_is_not_silently_skipped(self):
        (self.root / self.manifest["artifacts"][0]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing file"):
            historical_rows(self.root, self.manifest)

    def test_duplicate_source_ids_are_rejected(self):
        self.manifest["artifacts"].append(self.manifest["artifacts"][0])
        self.assertTrue(
            any(
                "Duplicate artifact id" in error
                for error in audit_archive(self.root, self.manifest)
            )
        )

    def test_paths_cannot_escape_by_parent_absolute_or_symlink(self):
        for path in ("../elsewhere.json", "/tmp/elsewhere.json"):
            with self.assertRaises(ValueError):
                safe_path(self.root, path)
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            safe_path(self.root, "escape/elsewhere.json")

    def test_invalid_or_missing_metrics_are_not_zero_filled(self):
        for value in (None, "0.7", True, float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                metric_at({"metric": value}, ("metric",))
        with self.assertRaises(KeyError):
            metric_at({}, ("metric",))

    def test_nonstandard_json_numbers_are_rejected(self):
        path = self.root / "bad.json"
        path.write_text('{"metric": NaN}')
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            load_json(path)

    def test_tables_identify_threshold_and_run_limitations(self):
        rows = historical_rows(self.root, self.manifest)
        markdown = render(rows, "markdown")
        self.assertIn("incomplete checkpoint/data provenance", markdown)
        self.assertIn("Calibration protocols differ", markdown)
        self.assertIn("threshold 0.3", markdown)
        self.assertIn("fixed threshold unpinned", markdown)
        self.assertNotIn("seed 17", markdown.lower())
        self.assertIn("leximind_test:/emotion/frozen_tuned_macro_f1", markdown)
        latex = render(rows, "latex")
        self.assertIn("% Source: leximind_test:/emotion/frozen_tuned_macro_f1", latex)
        self.assertIn(r"\noindent\textit{", latex)
        self.assertEqual(latex_escape("field_name & score"), r"field\_name \& score")

    def test_status_flags_alone_cannot_pass_preparation(self):
        preparation = {
            "schema_version": 1,
            "paused_by_user": False,
            "status": "ready_for_review",
            "gates": {},
        }
        blockers = preparation_blockers(self.root, preparation)
        self.assertIn("Unresolved gate: dataset_manifest", blockers)
        preparation["gates"]["dataset_manifest"] = {"status": "resolved", "evidence": []}
        self.assertIn(
            "Missing pinned evidence: dataset_manifest",
            preparation_blockers(self.root, preparation),
        )

    def test_current_pause_is_preserved_even_with_resolved_evidence(self):
        preparation = load_json(ROOT / "configs/research/preparation.json")
        preparation = copy.deepcopy(preparation)
        preparation["status"] = "ready_for_review"
        for gate in preparation["gates"].values():
            gate.update(status="resolved", evidence=[self.manifest["artifacts"][0]])
        self.assertEqual(
            preparation_blockers(self.root, preparation),
            ["Training and experiments remain paused by the user"],
        )

    def test_readiness_command_is_fail_closed_and_imports_no_models(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/audit_research_artifacts.py"), "--require-ready"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["archive_valid"])
        self.assertFalse(result["preparation_ready"])
        self.assertIn("no models loaded", result["scope"])


if __name__ == "__main__":
    unittest.main()
