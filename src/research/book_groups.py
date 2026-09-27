"""Connected leakage constraints for source records, never inferred work truth."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from .candidate_io import create_or_verify, sha

POLICY = "bgc-leakage-groups-v1"
KEYS = (
    "isbn13",
    "provider_book_id",
    "exact_blurb",
    "normalized_blurb",
    "normalized_title_author_candidate",
)
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class BookGroupRecord:
    record_id: str
    source_split: str
    source_row: int
    keys: dict[str, str]
    labels: tuple[tuple[int, str], ...]


def text_keys(title: str, author: str, body: str) -> dict[str, str]:
    """NFC/whitespace only: retain case/punctuation, skip empty matching fields."""
    title, author, blurb = (
        " ".join(unicodedata.normalize("NFC", value).split()) for value in (title, author, body)
    )
    keys = {"exact_blurb": sha(body), "normalized_blurb": sha(blurb)} if blurb else {}
    if title and author:
        keys["normalized_title_author_candidate"] = sha(
            json.dumps([title, author], ensure_ascii=False, separators=(",", ":"))
        )
    return keys


def proposed_split(group_id: str) -> str:
    """Fixed 64/16/20 hash allocation, without labels or input-order dependence."""
    bucket = int(sha(f"{POLICY}:split:{group_id}"), 16) % 10000
    return "train" if bucket < 6400 else "dev" if bucket < 8000 else "test"


def _line(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def prepare_book_groups(
    records: Iterable[BookGroupRecord], output: Path, namespace: str, *, max_rows: int = 100_000
) -> dict:
    """Index bounded hashes on disk; publish immutable assignments without copying text."""
    if not namespace.strip():
        raise ValueError("Grouping requires a pinned source namespace")
    parents: list[int] = []
    ranks: list[int] = []

    def find(value):
        while value != parents[value]:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(first, second):
        first, second = find(first), find(second)
        if first == second:
            return
        if ranks[first] < ranks[second]:
            first, second = second, first
        parents[second] = first
        if ranks[first] == ranks[second]:
            ranks[first] += 1

    with tempfile.TemporaryDirectory(prefix="leximind-book-groups-") as temporary:
        with sqlite3.connect(Path(temporary) / "groups.sqlite") as db:
            db.execute(
                "CREATE TABLE records(id INTEGER PRIMARY KEY, record_id TEXT UNIQUE, source_split TEXT, source_row INTEGER, labels TEXT, group_id TEXT, proposed_split TEXT, UNIQUE(source_split,source_row))"
            )
            db.execute("CREATE TABLE keys(kind TEXT,value TEXT,row_id INTEGER)")
            missing: Counter[str] = Counter()
            for index, row in enumerate(records):
                if index >= max_rows:
                    raise ValueError("Source exceeds bounded grouping row count")
                if (
                    not row.record_id.strip()
                    or row.source_split not in SPLITS
                    or type(row.source_row) is not int
                    or row.source_row < 1
                    or set(row.keys) - set(KEYS)
                    or any(not value.strip() for value in row.keys.values())
                    or not row.labels
                    or any(
                        type(depth) is not int or depth < 0 or not label.strip()
                        for depth, label in row.labels
                    )
                ):
                    raise ValueError("Invalid source record identity, labels or matching keys")
                labels = json.dumps(sorted(set(row.labels)), ensure_ascii=False)
                try:
                    db.execute(
                        "INSERT INTO records VALUES(?,?,?,?,?,NULL,NULL)",
                        (index, row.record_id, row.source_split, row.source_row, labels),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError("Repeated source row identity") from error
                db.executemany(
                    "INSERT INTO keys VALUES(?,?,?)",
                    ((kind, value, index) for kind, value in row.keys.items()),
                )
                missing.update(set(KEYS) - set(row.keys))
                parents.append(index)
                ranks.append(0)
            if not parents:
                raise ValueError("Cannot prepare groups for an empty source")
            db.execute("CREATE INDEX keys_by_value ON keys(kind,value,row_id)")
            for _, matches in groupby(
                db.execute("SELECT kind,value,row_id FROM keys ORDER BY kind,value,row_id"),
                key=lambda item: item[:2],
            ):
                first = None
                for _, _, other in matches:
                    if first is None:
                        first = other
                    else:
                        union(first, other)
            members: dict[int, list[tuple]] = {}
            for row in db.execute(
                "SELECT id,record_id,source_split,source_row,labels FROM records"
            ):
                members.setdefault(find(row[0]), []).append(row)
            group_sizes: Counter[int] = Counter()
            for group in members.values():
                identifiers = sorted(row[1] for row in group)
                group_id = "bgc-group:" + sha(
                    _line({"namespace": namespace, "members": identifiers})
                )
                split = proposed_split(group_id)
                db.executemany(
                    "UPDATE records SET group_id=?,proposed_split=? WHERE id=?",
                    ((group_id, split, row[0]) for row in group),
                )
                group_sizes[len(group)] += 1
            db.execute("CREATE INDEX records_by_group ON records(group_id)")
            db.execute("CREATE INDEX keys_by_row ON keys(row_id)")
            db.execute(
                "CREATE TEMP TABLE shared_keys AS SELECT kind,value,MIN(group_id) group_id,COUNT(*) rows,COUNT(DISTINCT source_split) source_splits,COUNT(DISTINCT proposed_split) proposed_splits,COUNT(DISTINCT labels) label_sets FROM keys JOIN records ON records.id=keys.row_id GROUP BY kind,value HAVING COUNT(*)>1"
            )
            overlap = {
                kind: dict(
                    zip(
                        (
                            "shared_keys",
                            "original_cross_split_keys",
                            "proposed_cross_split_keys",
                            "different_label_set_keys",
                        ),
                        db.execute(
                            "SELECT COUNT(*),COALESCE(SUM(source_splits>1),0),COALESCE(SUM(proposed_splits>1),0),COALESCE(SUM(label_sets>1),0) FROM shared_keys WHERE kind=?",
                            (kind,),
                        ).fetchone(),
                        strict=True,
                    )
                )
                for kind in KEYS
            }
            if any(value["proposed_cross_split_keys"] for value in overlap.values()):
                raise ValueError("Connected matching keys cross proposed splits")
            review_counts: Counter[str] = Counter(
                dict.fromkeys(
                    (
                        "multi_record_groups",
                        "groups_with_different_label_sets",
                        "groups_with_identity_label_conflicts",
                        "original_cross_split_groups",
                    ),
                    0,
                )
            )

            def reviews():
                for group_id, count in db.execute(
                    "SELECT group_id,COUNT(*) FROM records GROUP BY group_id HAVING COUNT(*)>1 ORDER BY group_id"
                ):
                    rows = list(
                        db.execute(
                            "SELECT record_id,source_split,source_row,labels FROM records WHERE group_id=? ORDER BY record_id",
                            (group_id,),
                        )
                    )
                    conflicts = [
                        {"kind": kind, "value": value}
                        for kind, value in db.execute(
                            "SELECT kind,value FROM shared_keys WHERE group_id=? AND kind IN ('isbn13','provider_book_id') AND label_sets>1 ORDER BY kind,value",
                            (group_id,),
                        )
                    ]
                    differs = len({row[3] for row in rows}) > 1
                    review_counts["multi_record_groups"] += 1
                    review_counts["groups_with_different_label_sets"] += differs
                    review_counts["groups_with_identity_label_conflicts"] += bool(conflicts)
                    review_counts["original_cross_split_groups"] += (
                        len({row[1] for row in rows}) > 1
                    )
                    yield _line(
                        {
                            "group_id": group_id,
                            "proposed_split": proposed_split(group_id),
                            "records": count,
                            "status": "unadjudicated_leakage_constraint",
                            "different_source_label_sets": differs,
                            "identity_label_conflicts": conflicts,
                            "shared_key_counts": dict(
                                db.execute(
                                    "SELECT kind,COUNT(*) FROM shared_keys WHERE group_id=? GROUP BY kind ORDER BY kind",
                                    (group_id,),
                                )
                            ),
                            "members": [
                                {
                                    "record_id": rid,
                                    "source_split": split,
                                    "source_row": number,
                                    "source_labels": json.loads(labels),
                                }
                                for rid, split, number, labels in rows
                            ],
                        }
                    )

            review_ref = create_or_verify(output / "review_groups.jsonl", reviews())

            def assignments():
                for rid, split, number, group_id, proposed, count in db.execute(
                    "SELECT r.record_id,r.source_split,r.source_row,r.group_id,r.proposed_split,g.n FROM records r JOIN (SELECT group_id,COUNT(*) n FROM records GROUP BY group_id) g ON r.group_id=g.group_id ORDER BY r.source_split,r.source_row"
                ):
                    yield _line(
                        {
                            "record_id": rid,
                            "source_split": split,
                            "source_row": number,
                            "group_id": group_id,
                            "proposed_split": proposed,
                            "review_required": count > 1,
                        }
                    )

            assignment_ref = create_or_verify(output / "assignments.jsonl", assignments())
            split_counts = dict(
                db.execute("SELECT proposed_split,COUNT(*) FROM records GROUP BY proposed_split")
            )
            moved = db.execute(
                "SELECT COUNT(*) FROM records WHERE proposed_split!=source_split"
            ).fetchone()[0]
            counts = {
                "records": len(parents),
                "groups": len(members),
                "group_size_counts": {
                    str(size): count for size, count in sorted(group_sizes.items())
                },
                "proposed_split_records": {split: split_counts.get(split, 0) for split in SPLITS},
                "rows_whose_proposed_split_differs": moved,
                **review_counts,
            }
            label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
            for split, labels in db.execute("SELECT proposed_split,labels FROM records"):
                label_counts[split].update({label for _, label in json.loads(labels)})
            all_labels = set().union(*label_counts.values())
            label_support = {
                split: {
                    "unique_labels": len(labels),
                    "minimum_present_count": min(labels.values(), default=0),
                    "missing_labels": sorted(all_labels - set(labels)),
                    "labels_with_fewer_than_five_records": {
                        label: labels[label] for label in sorted(all_labels) if labels[label] < 5
                    },
                }
                for split, labels in label_counts.items()
            }
            largest = [
                {
                    "group_id": group_id,
                    "records": size,
                    "distinct_source_label_sets": labels,
                    "original_split_count": splits,
                }
                for group_id, size, labels, splits in db.execute(
                    "SELECT group_id,COUNT(*) n,COUNT(DISTINCT labels),COUNT(DISTINCT source_split) FROM records GROUP BY group_id HAVING n>1 ORDER BY n DESC,group_id LIMIT 3"
                )
            ]
    return {
        "counts": counts,
        "matching_key_overlap": overlap,
        "records_without_matching_key": {key: missing[key] for key in KEYS},
        "proposed_split_source_label_support": label_support,
        "largest_group_examples": largest,
        "assignments": assignment_ref,
        "review_groups": review_ref,
    }
