"""Read-only verified-party hierarchy used by the rules matcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .domain import ExpansionCandidate, normalize_name


@dataclass
class VerifiedNode:
    party_id: str
    party_name: str
    parent_id: str | None = None
    candidates: list[ExpansionCandidate] = field(default_factory=list)


class VerifiedGraph:
    def __init__(self, parties: Iterable[dict[str, str | list[str] | None]]) -> None:
        self.nodes: dict[str, VerifiedNode] = {}
        for party in parties:
            party_id = str(party["verified_id"]).strip()
            party_name = str(party["verified_name"]).strip()
            parent_id = str(party.get("parent_id") or "").strip() or None
            if not party_id or not party_name:
                raise ValueError("Verified parties need nonblank verified_id and verified_name")
            if party_id in self.nodes:
                raise ValueError(f"Duplicate verified_id: {party_id}")
            aliases = party.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [value.strip() for value in aliases.split("|") if value.strip()]
            candidates = [
                ExpansionCandidate(name=str(alias).strip(), candidate_type="alias",
                                   llm_confidence=1.0, source="supplied_csv")
                for alias in aliases if normalize_name(alias) != normalize_name(party_name)
            ]
            self.nodes[party_id] = VerifiedNode(party_id, party_name, parent_id, candidates)
        for node in self.nodes.values():
            if node.parent_id and node.parent_id not in self.nodes:
                raise ValueError(f"Unknown parent_id {node.parent_id} for {node.party_id}")
        for party_id in self.nodes:
            self.root_id(party_id)  # validate cycles before building indexes
        self._candidate_owners: dict[str, set[str]] = {}
        for party_id in self.nodes:
            for name, _, _, _ in self.variants(party_id):
                key = normalize_name(name)
                if key:
                    self._candidate_owners.setdefault(key, set()).add(party_id)

    def root_id(self, party_id: str) -> str:
        current, visited = party_id, set()
        while self.nodes[current].parent_id:
            if current in visited:
                raise ValueError(f"Cycle in verified parent hierarchy at {current}")
            visited.add(current)
            current = str(self.nodes[current].parent_id)
        return current

    def path_to_root(self, party_id: str) -> list[str]:
        path = [party_id]
        while self.nodes[path[-1]].parent_id:
            parent = str(self.nodes[path[-1]].parent_id)
            if parent in path:
                raise ValueError(f"Cycle in verified parent hierarchy at {parent}")
            path.append(parent)
        return path

    def variants(self, party_id: str) -> list[tuple[str, str, str, float | None]]:
        node = self.nodes[party_id]
        result = [(node.party_name, "official", "verified", None)]
        result.extend((candidate.name, candidate.candidate_type, candidate.source,
                       candidate.llm_confidence) for candidate in node.candidates)
        return result

    def candidate_collision_count(self, candidate_name: str) -> int:
        return len(self._candidate_owners.get(normalize_name(candidate_name), set())) or 1

    def candidate_owner_ids(self, candidate_name: str) -> set[str]:
        return self._candidate_owners.get(normalize_name(candidate_name), set())
