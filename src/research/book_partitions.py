"""Sparse, immutable cross-source split constraints; no work or label adjudication."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from .candidate_io import sha

POLICY = "book-component-overlay-v1"
STRONG_KEYS = {"isbn13", "provider_book_id", "normalized_title_creator_candidate"}
SPLITS = {"train", "dev", "test"}


def component_split(identifier: str) -> str:
    bucket = int(sha(f"{POLICY}:{identifier}"), 16) % 10_000
    return "train" if bucket < 6400 else "dev" if bucket < 8000 else "test"


def build_components(
    bgc_groups: Mapping[str, str],
    external_ids: Iterable[str],
    identity_pairs: Iterable[tuple[str, str]],
    title_only_pairs: Iterable[tuple[str, str]],
) -> list[dict]:
    """Only touched BGC groups need overrides; all other base assignments survive.

    External IDs must include their source namespace. Future sources can provide
    evidence-backed pairs through this interface, but are never joined by bare
    IDs or an assumption that a source document is a globally identified work.
    Title-only edges do not merge identities. Both connected endpoints are held
    out of every proposed split until that ambiguity is separately resolved.
    """
    external = list(external_ids)
    if (
        len(set(external)) != len(external)
        or any(
            not isinstance(node, str)
            or ":" not in node
            or node.startswith("bgc-group:")
            or not all(node.split(":", 1))
            or "\n" in node
            or "\r" in node
            for node in external
        )
        or any(
            not isinstance(node, str)
            or not node.startswith("bgc-group:")
            or not node.removeprefix("bgc-group:")
            or "\n" in node
            or "\r" in node
            or split not in SPLITS
            for node, split in bgc_groups.items()
        )
    ):
        raise ValueError("Expected unique namespaced identities and valid BGC base assignments")
    known = set(bgc_groups) | set(external)
    parents = {node: node for node in external}

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def checked(pairs):
        seen = set()
        for pair in pairs:
            if len(pair) != 2 or pair[0] == pair[1] or any(node not in known for node in pair):
                raise ValueError("Constraint requires two distinct known identities")
            pair = tuple(sorted(pair))
            if pair in seen:
                continue
            seen.add(pair)
            for node in pair:
                parents.setdefault(node, node)
            yield pair

    for left, right in checked(identity_pairs):
        first, second = sorted((find(left), find(right)))
        parents[second] = first
    unresolved = list(checked(title_only_pairs))
    quarantined = {find(node) for pair in unresolved for node in pair}
    members = defaultdict(list)
    for node in sorted(parents):
        members[find(node)].append(node)
    result = []
    for root, nodes in members.items():
        identifier = "book-component:" + sha("\n".join(nodes))
        quarantine = root in quarantined
        result.append(
            {
                "component_id": identifier,
                "members": nodes,
                "proposed_split": None if quarantine else component_split(identifier),
                "status": "quarantined_title_only_ambiguity"
                if quarantine
                else "candidate_not_admitted",
                "training_eligible": False,
                "bgc_base_proposed_splits": dict(
                    Counter(bgc_groups[node] for node in nodes if node in bgc_groups)
                ),
            }
        )
    return sorted(result, key=lambda row: row["component_id"])


def group_overrides(components: Iterable[dict]) -> dict[str, dict]:
    """Resolve BGC record -> original group -> overlay without copying corpus rows."""
    result = {}
    for component in components:
        for node in component["members"]:
            if node.startswith("bgc-group:"):
                if node in result:
                    raise ValueError("A BGC group occurs in multiple components")
                result[node] = component
    return result
