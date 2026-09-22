from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .domain import ExpansionCandidate, base_name, normalize_name, read_json, write_json


@dataclass
class VerifiedNode:
    party_id: str
    party_name: str
    parent_id: str | None = None
    candidates: list[ExpansionCandidate] = field(default_factory=list)
    suggested_parent: dict[str, Any] | None = None
    parent_evidence: dict[str, Any] | None = None


class VerifiedGraph:
    """A persisted verified-only, account-scoped forest."""

    schema_version = 1

    def __init__(self, account_id: str, path: str | Path, load_existing: bool = True):
        self.account_id = account_id
        self.path = Path(path)
        self.nodes: dict[str, VerifiedNode] = {}
        self.pending_parent_ids: set[str] = set()
        self._candidate_owners: dict[str, set[str]] = {}
        if load_existing:
            self._load()

    def _load(self) -> None:
        payload = read_json(self.path, {}) or {}
        if payload and payload.get("account_id") != self.account_id:
            raise ValueError("Graph account does not match requested account")
        for raw in payload.get("nodes", []):
            raw["candidates"] = [ExpansionCandidate(**item) for item in raw.get("candidates", [])]
            node = VerifiedNode(**raw)
            self.nodes[node.party_id] = node
            if node.suggested_parent and not node.parent_id:
                self.pending_parent_ids.add(node.party_id)
        self._rebuild_indexes()

    def save(self) -> None:
        write_json(self.path, {
            "schema_version": self.schema_version,
            "account_id": self.account_id,
            "nodes": [asdict(self.nodes[key]) for key in sorted(self.nodes)],
        })

    def add_parties(self, parties: Iterable[dict[str, str]]) -> list[str]:
        added: list[str] = []
        for party in parties:
            party_id, party_name = str(party["partyId"]), str(party["partyName"]).strip()
            if party_id in self.nodes:
                if self.nodes[party_id].party_name != party_name:
                    self.nodes[party_id].party_name = party_name
                continue
            self.nodes[party_id] = VerifiedNode(party_id=party_id, party_name=party_name)
            added.append(party_id)
        self._rebuild_indexes()
        return added

    def apply_expansions(self, entities: Iterable[dict[str, Any]]) -> None:
        for entity in entities:
            node = self.nodes.get(str(entity.get("party_id", "")))
            if not node:
                continue
            seen: set[str] = set()
            candidates: list[ExpansionCandidate] = []
            for item in entity.get("match_candidates", []):
                name = str(item.get("name", "")).strip()
                key = normalize_name(name)
                if not name or not key or key in seen or key == normalize_name(node.party_name):
                    continue
                seen.add(key)
                candidates.append(ExpansionCandidate(
                    name=name,
                    candidate_type=str(item.get("candidate_type", "alias")),
                    llm_confidence=_float_or_none(item.get("llm_confidence")),
                    source=str(item.get("source", "llm")),
                ))
            node.candidates = candidates
            parent = entity.get("suggested_parent")
            if isinstance(parent, dict) and str(parent.get("name", "")).strip():
                node.suggested_parent = {
                    "name": str(parent["name"]).strip(),
                    "llm_confidence": _float_or_none(parent.get("llm_confidence")),
                    "source": str(parent.get("source", "llm")),
                }
                if not node.parent_id:
                    self.pending_parent_ids.add(node.party_id)
            elif not node.parent_id:
                node.suggested_parent = None
                self.pending_parent_ids.discard(node.party_id)
        self._rebuild_indexes()

    def reconcile_suggested_parents(self) -> list[dict[str, Any]]:
        """Resolve only unique deterministic official/base-name matches."""
        official: dict[str, set[str]] = {}
        for node in self.nodes.values():
            for key in {normalize_name(node.party_name), base_name(node.party_name)} - {""}:
                official.setdefault(key, set()).add(node.party_id)
        updates: list[dict[str, Any]] = []
        for child_id in sorted(self.pending_parent_ids):
            child = self.nodes.get(child_id)
            if not child or child.parent_id or not child.suggested_parent:
                continue
            suggested = child.suggested_parent["name"]
            matches = set()
            for key in {normalize_name(suggested), base_name(suggested)} - {""}:
                matches.update(official.get(key, set()))
            matches.discard(child_id)
            if len(matches) != 1:
                continue
            parent_id = next(iter(matches))
            if self._would_cycle(child_id, parent_id):
                continue
            child.parent_id = parent_id
            child.parent_evidence = {
                "method": "unique_normalized_official_name",
                "suggested_name": suggested,
            }
            updates.append({
                "child_id": child_id,
                "child_name": child.party_name,
                "parent_id": parent_id,
                "parent_name": self.nodes[parent_id].party_name,
            })
        self.pending_parent_ids = {
            node_id for node_id in self.pending_parent_ids
            if node_id in self.nodes and not self.nodes[node_id].parent_id
        }
        return updates

    def root_id(self, party_id: str) -> str:
        current, visited = party_id, set()
        while self.nodes[current].parent_id:
            if current in visited:
                raise ValueError(f"Cycle detected at {current}")
            visited.add(current)
            current = str(self.nodes[current].parent_id)
        return current

    def path_to_root(self, party_id: str) -> list[str]:
        path, current = [], party_id
        while True:
            if current in path:
                raise ValueError(f"Cycle detected at {current}")
            path.append(current)
            parent = self.nodes[current].parent_id
            if not parent:
                return path
            current = parent

    def variants(self, party_id: str) -> list[tuple[str, str, str, float | None]]:
        node = self.nodes[party_id]
        values = [(node.party_name, "official", "verified", None)]
        values.extend(
            (candidate.name, candidate.candidate_type, candidate.source, candidate.llm_confidence)
            for candidate in node.candidates
        )
        return values

    def candidate_collision_count(self, candidate_name: str) -> int:
        return len(self._candidate_owners.get(normalize_name(candidate_name), set())) or 1

    def _would_cycle(self, child_id: str, parent_id: str) -> bool:
        current = parent_id
        while current:
            if current == child_id:
                return True
            node = self.nodes.get(current)
            current = node.parent_id if node else None
        return False

    def _rebuild_indexes(self) -> None:
        self._candidate_owners = {}
        for node in self.nodes.values():
            for name, _, _, _ in self.variants(node.party_id):
                key = normalize_name(name)
                if key:
                    self._candidate_owners.setdefault(key, set()).add(node.party_id)


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
