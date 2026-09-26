import hashlib
from pathlib import Path

import pytest

from src.research.io import check_file, read_json, safe_path

ROOT = Path(__file__).resolve().parents[2]


def test_retained_historical_reports_match_their_original_hashes():
    manifest = read_json(ROOT / "research/results/manifest.json")
    for item in manifest["artifacts"]:
        assert check_file(ROOT, item) == []


def test_safe_path_rejects_parent_absolute_and_symlink_escape(tmp_path):
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    for name in ("../outside.json", "/tmp/outside.json", "escape/outside.json"):
        with pytest.raises(ValueError):
            safe_path(tmp_path, name)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"score":NaN}', '{"score":Infinity}'])
def test_json_rejects_ambiguous_or_nonfinite_data(tmp_path, raw):
    path = tmp_path / "data.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_json(path)


def test_same_length_mutation_and_boolean_size_do_not_pass(tmp_path):
    path = tmp_path / "data.json"
    original = b'{"a":1}'
    path.write_bytes(original)
    reference = {
        "path": path.name,
        "bytes": len(original),
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    assert check_file(tmp_path, reference) == []
    path.write_text('{"a":2}')
    assert any("SHA-256 changed" in message for message in check_file(tmp_path, reference))
    assert check_file(tmp_path, {**reference, "bytes": True})
