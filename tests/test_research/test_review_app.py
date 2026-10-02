"""Offline human-review drafts remain source-bound, explicit and unadmitted."""

from __future__ import annotations

import json
import re

import pytest

from src.research import review_app as app
from src.research.book_fields import FACETS, input_hash
from src.research.builders.book_fields import reference
from src.research.builders.field_review import CONTRACT
from src.research.candidate_io import json_bytes
from src.research.field_reviews import evidence_span, human_reviewed_states
from src.research.io import read_json


@pytest.fixture
def review_bundle(tmp_path):
    root = tmp_path
    local = root / "data/research_candidates/synthetic"
    local.mkdir(parents=True)
    mapping_path = local / "mapping.json"
    mapping = read_json(app.ROOT / "research/preparation/book_field_mapping.json")
    mapping_path.write_bytes(json_bytes(mapping))
    source = local / "source.txt"
    source.write_text("Synthetic source fixture, not book data.")
    bindings = {
        name: reference(root, source) for name in ("archive", "field_manifest", "field_references")
    }
    bindings["mapping"] = reference(root, mapping_path)
    payload = {
        "title": "Synthetic 🐈 café </script><script>throw Error('unsafe')</script>",
        "description": "Synthetic story for adults; explicitly not for children.",
    }
    candidate = {
        "record_id": "synthetic:1",
        "input_sha256": input_hash(payload),
        "source": {"provider_url": "https://example.invalid/synthetic"},
        "fields": {facet: {"positive": [], "negative": []} for facet in FACETS},
    }
    candidate["fields"]["genre"]["positive"] = ["fantasy"]
    review = {
        "record_id": candidate["record_id"],
        "input_sha256": candidate["input_sha256"],
        "source_url": candidate["source"]["provider_url"],
        "selection": ["synthetic fixture"],
        "agent_review": {
            "reviewer": {
                "kind": "agent",
                "id": "synthetic-agent",
                "method": "assistant_source_inspection",
                "reviewed_at": "2026-10-02",
            },
            "decisions": [
                {
                    "field": "genre",
                    "label": "fantasy",
                    "state": "positive",
                    "evidence": [evidence_span(payload, "title", 0, 9)],
                    "rationale": "Synthetic candidate only.",
                }
            ],
        },
        "human_review": None,
    }
    packet = {
        "schema_version": 1,
        "status": "candidate_field_reviews_not_admitted",
        "training_authorized": False,
        "human_gold": False,
        "bindings": bindings,
        "contract": dict(CONTRACT),
        "selection": "Synthetic fixture only",
        "records": [review],
    }
    packet_path = root / "research/preparation/book_field_review.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_bytes(json_bytes(packet))
    worksheet = [
        {
            "review": review,
            "input": payload,
            "source": candidate["source"],
            "source_fields": candidate["fields"],
            "source_labels": [[0, "Synthetic source category"]],
            "group": {"group_id": "synthetic-group"},
        }
    ]
    worksheet_path = local / "worksheet.json"
    worksheet_path.write_bytes(json_bytes(worksheet))
    manifest = {
        "schema_version": 1,
        "status": packet["status"],
        "training_authorized": False,
        "human_gold": False,
        "review_packet": reference(root, packet_path),
        "bindings": bindings,
        "local_worksheet": reference(root, worksheet_path),
    }
    manifest_path = packet_path.parent / "review_manifest.json"
    manifest_path.write_bytes(json_bytes(manifest))
    return root, manifest_path, packet_path, candidate, payload, mapping


def human_draft(bundle):
    root, manifest, _, candidate, payload, _ = bundle
    start = payload["title"].index("🐈")
    return {
        "schema_version": 1,
        "kind": app.DRAFT_KIND,
        "bindings": app._load(root, manifest)["bindings"],
        "records": [
            {
                "record_id": candidate["record_id"],
                "input_sha256": candidate["input_sha256"],
                "human_review": {
                    "reviewer": {
                        "kind": "human",
                        "id": "synthetic-test-reviewer",
                        "method": "direct_source_review",
                        "reviewed_at": "2026-10-02",
                    },
                    "decisions": [
                        {
                            "field": "audience",
                            "label": "children",
                            "state": "unknown",
                            "evidence": [evidence_span(payload, "title", start, start + 1)],
                            "rationale": "Synthetic abstention fixture; the selected symbol does not establish an audience.",
                        }
                    ],
                },
            }
        ],
    }


def test_build_is_offline_inert_and_preserves_unknown_defaults(review_bundle):
    root, manifest, packet_path, _, payload, _ = review_bundle
    original = packet_path.read_bytes()
    output = root / "data/research_candidates/synthetic/review.html"
    result = app.build(root, manifest, output)
    html = output.read_text()
    assert result == app.build(root, manifest, output)
    assert html.count("</script>") == 2
    embedded = json.loads(
        re.search(r'<script id="review-data" type="application/json">(.*?)</script>', html).group(1)
    )
    assert embedded["rows"][0]["input"] == payload
    assert sum(len(spec["labels"]) for spec in embedded["facets"].values()) == 48
    assert embedded["rows"][0]["human_review"] is None
    assert "connect-src 'none'" in html and "fetch(" not in html
    assert '<details id="prior"><summary>' in html
    assert "kind:'human'" in html
    assert packet_path.read_bytes() == original


def test_human_import_is_explicit_preserves_agent_evidence_and_stays_unadmitted(review_bundle):
    root, manifest, packet_path, candidate, payload, mapping = review_bundle
    original = packet_path.read_bytes()
    draft = root / "draft.json"
    value = human_draft(review_bundle)
    value["records"][0]["human_review"]["decisions"][0]["state"] = "negative"
    draft.write_bytes(json_bytes(value))
    output = root / "data/research_candidates/synthetic/human.json"
    result = app.import_draft(root, manifest, draft, output)
    packet = read_json(output)
    assert result == app.import_draft(root, manifest, draft, output)
    assert packet_path.read_bytes() == original
    assert packet["training_authorized"] is packet["human_gold"] is False
    assert (
        packet["records"][0]["agent_review"] == read_json(packet_path)["records"][0]["agent_review"]
    )
    states = human_reviewed_states(packet["records"][0], candidate, payload, mapping)
    assert states["audience"]["negative"] == ["children"]
    assert states["genre"]["positive"] == []
    span = packet["records"][0]["human_review"]["decisions"][0]["evidence"][0]
    assert payload["title"][span["start"] : span["end"]] == "🐈"
    assert b"Synthetic story" not in output.read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "bindings",
        "input",
        "agent_slot",
        "duplicate_record",
        "duplicate_decision",
        "empty",
        "null",
        "provenance",
        "offset_utf16",
        "bool_offset",
        "hash",
        "rationale",
        "copied_text",
        "date",
        "source_conflict",
        "nan",
        "duplicate_json_key",
    ],
)
def test_malformed_stale_or_conflicting_drafts_never_publish(review_bundle, mutation):
    root, manifest, packet, _, _, _ = review_bundle
    original = packet.read_bytes()
    value = human_draft(review_bundle)
    entry = value["records"][0]
    decision = entry["human_review"]["decisions"][0]
    if mutation == "bindings":
        value["bindings"]["packet"]["sha256"] = "0" * 64
    elif mutation == "input":
        entry["input_sha256"] = "0" * 64
    elif mutation == "agent_slot":
        entry["agent_review"] = None
    elif mutation == "duplicate_record":
        value["records"] *= 2
    elif mutation == "duplicate_decision":
        entry["human_review"]["decisions"] *= 2
    elif mutation == "empty":
        value["records"] = []
    elif mutation == "null":
        entry["human_review"] = None
    elif mutation == "provenance":
        entry["human_review"]["reviewer"]["kind"] = "agent"
    elif mutation == "offset_utf16":
        decision["evidence"][0]["end"] += 1
    elif mutation == "bool_offset":
        decision["evidence"][0]["start"] = True
    elif mutation == "hash":
        decision["evidence"][0]["sha256"] = "0" * 64
    elif mutation == "rationale":
        decision["rationale"] = ""
    elif mutation == "copied_text":
        decision["evidence"][0]["text"] = "copied source text"
    elif mutation == "date":
        entry["human_review"]["reviewer"]["reviewed_at"] = "2026-02-30"
    elif mutation == "source_conflict":
        decision.update(field="genre", label="fantasy", state="negative")
    encoded = json_bytes(value)
    if mutation == "nan":
        encoded = encoded.replace(b'"start": 10', b'"start": NaN')
    elif mutation == "duplicate_json_key":
        encoded = encoded.replace(
            b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1'
        )
    draft = root / "draft.json"
    draft.write_bytes(encoded)
    output = root / "data/research_candidates/new/human.json"
    with pytest.raises(ValueError):
        app.import_draft(root, manifest, draft, output)
    assert not output.exists()
    assert packet.read_bytes() == original


@pytest.mark.parametrize(
    "target", ["outside", "packet", "worksheet", "draft", "existing", "symlink"]
)
def test_import_never_overwrites_evidence_or_existing_work(review_bundle, target):
    root, manifest, packet, _, _, _ = review_bundle
    value = human_draft(review_bundle)
    draft = root / "data/research_candidates/draft.json"
    draft.write_bytes(json_bytes(value))
    output = root / "data/research_candidates/output.json"
    if target == "outside":
        output = root / "tracked.json"
    elif target == "packet":
        output = packet
    elif target == "worksheet":
        output = root / read_json(manifest)["local_worksheet"]["path"]
    elif target == "draft":
        output = draft
    elif target == "existing":
        output.write_text("preexisting work")
    else:
        outside = root / "outside"
        outside.mkdir()
        (root / "data/research_candidates/escape").symlink_to(outside, target_is_directory=True)
        output = root / "data/research_candidates/escape/output.json"
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError):
        app.import_draft(root, manifest, draft, output)
    assert (output.read_bytes() if output.exists() else None) == before


def test_changed_source_rejected_before_build_and_import(review_bundle):
    root, manifest, _, _, _, _ = review_bundle
    draft = root / "draft.json"
    draft.write_bytes(json_bytes(human_draft(review_bundle)))
    (root / "data/research_candidates/synthetic/source.txt").write_text("changed source")
    output = root / "data/research_candidates/new/result.json"
    with pytest.raises(ValueError, match="changed"):
        app.import_draft(root, manifest, draft, output)
    with pytest.raises(ValueError, match="changed"):
        app.build(root, manifest, output.with_suffix(".html"))
    assert not output.parent.exists()


def test_source_changed_during_render_never_publishes(review_bundle, monkeypatch):
    root, manifest, _, _, _, _ = review_bundle
    original = app._load

    def changed(*args):
        bundle = original(*args)
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return bundle

    monkeypatch.setattr(app, "_load", changed)
    output = root / "data/research_candidates/new/result.html"
    with pytest.raises(ValueError, match="changed"):
        app.build(root, manifest, output)
    assert not output.exists()
