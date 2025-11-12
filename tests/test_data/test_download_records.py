"""Unit tests for dataset record helpers in scripts.download_data."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any, Dict, Iterator, List, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_data.py"

spec = importlib.util.spec_from_file_location("download_data", DOWNLOAD_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load scripts/download_data.py for testing")
download_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_data)


class DummyDataset:
    def __init__(self, records: List[Dict[str, object]]) -> None:
        self._records = records

    def __iter__(self) -> Iterator[Dict[str, object]]:
        return iter(self._records)


class DownloadDataRecordTests(unittest.TestCase):
    def test_emotion_records_handles_out_of_range_labels(self) -> None:
        dataset_split = DummyDataset([
            {"text": "sample", "label": 1},
            {"text": "multi", "label": [0, 5]},
            {"text": "string", "label": "2"},
        ])
        label_names = ["sadness", "joy", "love"]
        records = list(
            download_data._emotion_records(
                cast(Any, dataset_split),
                label_names,
            )
        )
        self.assertEqual(records[0]["emotions"], ["joy"])
        # Out-of-range index falls back to string representation
        self.assertEqual(records[1]["emotions"], ["sadness", "5"])
        # Non-int values fall back to string
        self.assertEqual(records[2]["emotions"], ["2"])

    def test_topic_records_handles_varied_label_inputs(self) -> None:
        dataset_split = DummyDataset([
            {"text": "news", "label": 3},
            {"text": "list", "label": [1]},
            {"text": "unknown", "label": "5"},
            {"text": "missing", "label": []},
        ])
        label_names = ["World", "Sports", "Business", "Sci/Tech"]
        records = list(
            download_data._topic_records(
                cast(Any, dataset_split),
                label_names,
            )
        )
        self.assertEqual(records[0]["topic"], "Sci/Tech")
        self.assertEqual(records[1]["topic"], "Sports")
        # Out-of-range string label falls back to original string value
        self.assertEqual(records[2]["topic"], "5")
        # Empty list yields empty string
        self.assertEqual(records[3]["topic"], "")


if __name__ == "__main__":
    unittest.main()
