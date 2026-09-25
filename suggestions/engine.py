"""Provisional name suggestions as a verified-party list changes.

This module deliberately does not use labels, graph relationships, or the POC's
accepted mappings. A shared brand is evidence for review, not proof that two
legal entities are identical or even part of the same corporate family.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


_LEGAL_ENDINGS = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "plc", "gmbh", "sa", "sas",
    "bv", "pte", "pty", "na",
}
_GENERIC_ANCHORS = {
    "bank", "capital", "company", "corporation", "financial", "global",
    "group", "health", "international", "national", "services",
    "solutions", "technology", "university", "world",
}
_TOKEN_EQUIVALENTS = {"labs": "laboratory", "laboratories": "laboratory"}
_LEGAL_EQUIVALENTS = {
    "incorporated": "inc", "corporation": "corp", "company": "co",
    "limited": "ltd",
}


def _words(name: str) -> list[str]:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", ascii_name.casefold().replace("&", " and "))
    # Keep common dotted legal forms as one terminal token.
    for ending in (("l", "l", "c"), ("n", "a"), ("s", "a")):
        if tuple(words[-len(ending):]) == ending:
            words[-len(ending):] = ["".join(ending)]
            break
    return words


def name_tokens(name: str) -> tuple[str, ...]:
    """Normalize presentation differences, but do not discard business words."""
    words = _words(name)
    if words and words[0] == "the":
        words.pop(0)
    words = [_TOKEN_EQUIVALENTS.get(word, word) for word in words]
    while len(words) > 1 and words[-1] in _LEGAL_ENDINGS:
        words.pop()
    return tuple(words)


def _legal_form(name: str) -> str:
    words = _words(name)
    if not words or words[-1] not in _LEGAL_ENDINGS:
        return ""
    return _LEGAL_EQUIVALENTS.get(words[-1], words[-1])


@dataclass(frozen=True)
class VerifiedParty:
    party_id: str
    name: str


@dataclass(frozen=True)
class Suggestion:
    status: str  # SUGGESTED, AMBIGUOUS, or NONE; never a confirmed mapping.
    party_id: str = ""
    party_name: str = ""
    evidence: str = ""
    alternative_names: tuple[str, ...] = ()


def _evidence(raw: tuple[str, ...], verified: tuple[str, ...],
              raw_form: str, verified_form: str) -> tuple[int, str]:
    if not raw or not verified:
        return 0, ""
    if raw == verified:
        if raw_form and raw_form == verified_form:
            return 105, "same meaningful name and stated legal form"
        return 100, "same meaningful name (legal form/punctuation may differ)"
    if (len(raw) > len(verified) and raw[: len(verified)] == verified
            and (len(verified) > 1 or
                 (len(verified[0]) >= 5 and verified[0] not in _GENERIC_ANCHORS))):
        return 85, "full verified name followed by extra words"
    # A single generic word, such as BANK or GROUP, is never enough to suggest
    # a legal entity. This also avoids broad fuzzy matching to unrelated names.
    anchor = verified[0]
    if raw[0] == anchor and len(anchor) >= 5 and anchor not in _GENERIC_ANCHORS:
        return 55, "shared distinctive leading name; relationship unverified"
    return 0, ""


class SuggestionIndex:
    """Index verified names by their first meaningful token for repeated queries."""

    def __init__(self, verified: Iterable[VerifiedParty]) -> None:
        self.by_anchor: dict[str, list[tuple[VerifiedParty, tuple[str, ...], str]]] = {}
        for party in verified:
            tokens = name_tokens(party.name)
            if tokens:
                self.by_anchor.setdefault(tokens[0], []).append(
                    (party, tokens, _legal_form(party.name))
                )

    def suggest(self, raw_name: str) -> Suggestion:
        raw = name_tokens(raw_name)
        if not raw:
            return Suggestion("NONE", evidence="No supported name anchor")
        raw_form = _legal_form(raw_name)
        scored: list[tuple[int, VerifiedParty, str]] = []
        for party, tokens, form in self.by_anchor.get(raw[0], ()):
            score, reason = _evidence(raw, tokens, raw_form, form)
            if score:
                scored.append((score, party, reason))
        if not scored:
            return Suggestion("NONE", evidence="No supported name anchor")
        scored.sort(key=lambda item: (-item[0], item[1].name.casefold(), item[1].party_id))
        top_score, top_party, reason = scored[0]
        # Equal-strength official names or brand anchors are deliberately left
        # for review. We never silently move between competing parties.
        tied = [party.name for score, party, _ in scored if score == top_score]
        if len(tied) > 1:
            return Suggestion(
                "AMBIGUOUS", evidence="Multiple verified names have equal name evidence",
                alternative_names=tuple(tied[:5]),
            )
        return Suggestion("SUGGESTED", top_party.party_id, top_party.name, reason)


def suggest(raw_name: str, verified: Iterable[VerifiedParty]) -> Suggestion:
    return SuggestionIndex(verified).suggest(raw_name)


def change_status(before: Suggestion, after: Suggestion) -> str:
    if before.status == "NONE" and after.status == "SUGGESTED":
        return "NEW_SUGGESTION"
    if before.status == "SUGGESTED" and after.status == "SUGGESTED":
        return "UNCHANGED" if before.party_id == after.party_id else "MOVED_SUGGESTION"
    if before.status == "SUGGESTED" and after.status == "AMBIGUOUS":
        return "NOW_AMBIGUOUS"
    if before.status == "SUGGESTED" and after.status == "NONE":
        return "WITHDRAWN"
    if before.status == "AMBIGUOUS" and after.status == "SUGGESTED":
        return "AMBIGUITY_RESOLVED"
    return "UNCHANGED"


def verified_name_relation(first: VerifiedParty, second: VerifiedParty) -> tuple[str, str]:
    """Flag a verified-name relationship for review, never merge their IDs."""
    left, right = name_tokens(first.name), name_tokens(second.name)
    if left and left == right:
        return "POSSIBLE_DUPLICATE_NAME", "Meaningful names are equal after legal-form normalization"
    if (left and right and left[0] == right[0] and len(left[0]) >= 5
            and left[0] not in _GENERIC_ANCHORS):
        return "POSSIBLE_NAME_FAMILY", "Shared distinctive leading name; ownership not verified"
    return "NO_NAME_RELATION", "No supported common name anchor"
