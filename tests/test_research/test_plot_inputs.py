import json

import pytest

from scripts.plot_inputs import load_confusion_report


def test_confusion_matrix_keeps_its_recorded_label_order(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "topic": {
                    "labels": ["B", "A"],
                    "confusion_matrix": [[2, 1], [0, 3]],
                    "num_samples": 6,
                }
            }
        )
    )
    assert load_confusion_report(path, "topic") == (["B", "A"], [[2, 1], [0, 3]])


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"labels": ["A"], "confusion_matrix": [[True]]},
        {"labels": ["A"], "confusion_matrix": [[-1]]},
        {"labels": ["A"], "confusion_matrix": [[0]]},
        {"labels": ["A", "A"], "confusion_matrix": [[1, 0], [0, 1]]},
        {"labels": ["A", "B"], "confusion_matrix": [[1]]},
        {"labels": ["A"], "confusion_matrix": [[1]], "num_samples": 9},
    ],
)
def test_absent_or_inconsistent_data_never_becomes_a_synthetic_matrix(tmp_path, record):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"topic": record}))
    with pytest.raises(ValueError):
        load_confusion_report(path, "topic")
