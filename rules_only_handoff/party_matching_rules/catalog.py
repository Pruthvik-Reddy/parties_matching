"""Flat, read-only verified names and their explicitly supplied aliases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .domain import normalize_name


@dataclass
class VerifiedParty:
    party_id: str
    party_name: str
    aliases: list[str] = field(default_factory=list)


class VerifiedCatalog:
    def __init__(self, parties: Iterable[dict[str, str | list[str] | None]]) -> None:
        self.parties: dict[str, VerifiedParty] = {}
        for party in parties:
            party_id = str(party["verified_id"]).strip()
            party_name = str(party["verified_name"]).strip()
            if not party_id or not party_name:
                raise ValueError("Verified parties need nonblank verified_id and verified_name")
            if party_id in self.parties:
                raise ValueError(f"Duplicate verified_id: {party_id}")
            # A nonblank parent must fail loudly rather than silently changing
            # which verified ID the caller expects to receive.
            if str(party.get("parent_id") or "").strip():
                raise ValueError("parent_id is not supported by the flat alias matcher")
            aliases = party.get("aliases") or []
            if isinstance(aliases, str):
                aliases = aliases.split("|")
            names = [str(alias).strip() for alias in aliases]
            names = [name for name in names if name and normalize_name(name) != normalize_name(party_name)]
            self.parties[party_id] = VerifiedParty(party_id, party_name, names)

        # One alias can belong to multiple verified IDs. Retain that ambiguity
        # so the ordinary collision/abstention safeguards can see it.
        self._candidate_owners: dict[str, set[str]] = {}
        for party_id in self.parties:
            for name, _, _, _ in self.variants(party_id):
                key = normalize_name(name)
                if key:
                    self._candidate_owners.setdefault(key, set()).add(party_id)

    def variants(self, party_id: str) -> list[tuple[str, str, str, float | None]]:
        party = self.parties[party_id]
        return [(party.party_name, "official", "verified", None)] + [
            (alias, "alias", "supplied_csv", 1.0) for alias in party.aliases
        ]

    def candidate_collision_count(self, candidate_name: str) -> int:
        return len(self._candidate_owners.get(normalize_name(candidate_name), set())) or 1

    def candidate_owner_ids(self, candidate_name: str) -> set[str]:
        return self._candidate_owners.get(normalize_name(candidate_name), set())
