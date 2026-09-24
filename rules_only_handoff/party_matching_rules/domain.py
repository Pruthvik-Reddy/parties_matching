from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

CONNECTORS = re.compile(r"\b(OBO|VIA)\b", re.IGNORECASE)
LEGAL_SUFFIXES = {
    "ag", "bv", "co", "company", "corp", "corporation", "gmbh", "inc",
    "incorporated", "llc", "llp", "limited", "ltd", "nv", "plc", "pty",
    "sa", "sas", "sarl", "spa",
}

def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    text = re.sub(r"(?<=\b[a-z0-9])\.(?=[a-z0-9](?:\.|\b))", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def base_name(value: object) -> str:
    tokens = normalize_name(value).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def compact_name(value: object) -> str:
    return normalize_name(value).replace(" ", "")


@dataclass
class PartyRecord:
    account_id: str
    adm_party_id: str
    raw_name: str
    source_row: int
    eligible: bool = True


@dataclass
class OrganizationMention:
    mention_id: str
    adm_party_id: str
    text: str
    normalized: str
    base: str
    compact: str
    position: int
    connector_before: str | None = None
    connector_after: str | None = None
    parse_warning: str | None = None


def parse_mentions(record: PartyRecord) -> list[OrganizationMention]:
    raw = str(record.raw_name or "").strip()
    if not raw:
        return [OrganizationMention(
            mention_id=f"{record.adm_party_id}:m0", adm_party_id=record.adm_party_id,
            text="", normalized="", base="", compact="", position=0,
            parse_warning="EMPTY_NAME",
        )]
    pieces = CONNECTORS.split(raw)
    names = [pieces[i].strip() for i in range(0, len(pieces), 2)]
    connectors = [pieces[i].upper() for i in range(1, len(pieces), 2)]
    if any(not name for name in names):
        names, connectors = [raw], []
        warning = "MALFORMED_CONNECTOR"
    else:
        warning = None
    mentions: list[OrganizationMention] = []
    for index, name in enumerate(names):
        mentions.append(OrganizationMention(
            mention_id=f"{record.adm_party_id}:m{index}",
            adm_party_id=record.adm_party_id,
            text=name,
            normalized=normalize_name(name),
            base=base_name(name),
            compact=compact_name(name),
            position=index,
            connector_before=connectors[index - 1] if index else None,
            connector_after=connectors[index] if index < len(connectors) else None,
            parse_warning=warning,
        ))
    return mentions


@dataclass
class MatchProposal:
    adm_party_id: str
    mention_id: str
    mention_text: str
    connector_before: str | None
    connector_after: str | None
    owner_party_id: str
    owner_party_name: str
    root_party_id: str
    root_party_name: str
    matched_name: str
    candidate_type: str
    candidate_source: str
    candidate_confidence: float | None
    retrieval_sources: list[str] = field(default_factory=list)
    exact: float = 0.0
    lexical_score: float = 0.0
    char_tfidf_score: float = 0.0
    word_tfidf_score: float = 0.0
    rrf_score: float = 0.0
    embedding_score: float | None = None
    feature_score: float = 0.0
    rules_score: float = 0.0
    cross_encoder_score: float | None = None
    candidate_collision_count: int = 1
    char_similarity: float = 0.0
    jaro_winkler: float = 0.0
    levenshtein: float = 0.0
    token_jaccard: float = 0.0
    raw_coverage: float = 0.0
    candidate_coverage: float = 0.0
    distinctive_token_conflict: bool = False
    digit_conflict: bool = False


@dataclass
class FinalDecision:
    account_id: str
    adm_party_id: str
    raw_name: str
    decision: str
    confidence: float
    reason: str
    verified_party_id: str | None = None
    verified_party_name: str | None = None
    matched_member_id: str | None = None
    matched_member_name: str | None = None
    matched_candidate_name: str | None = None
    matched_candidate_type: str | None = None
    candidate_expansion_confidence: float | None = None
    matched_mention: str | None = None
    connector: str | None = None
    match_method: str | None = None
    decision_tier: str | None = None
    retrieval_sources: list[str] = field(default_factory=list)
    identity_score: float | None = None
    char_tfidf_score: float | None = None
    word_tfidf_score: float | None = None
    rrf_score: float | None = None
    char_similarity: float | None = None
    jaro_winkler: float | None = None
    levenshtein: float | None = None
    token_jaccard: float | None = None
    raw_coverage: float | None = None
    candidate_coverage: float | None = None
    distinctive_token_conflict: bool | None = None
    digit_conflict: bool | None = None
    cross_encoder_score: float | None = None
    runner_up_party_id: str | None = None
    runner_up_score: float | None = None
    margin: float | None = None
    retrieved_root_ids: list[str] = field(default_factory=list)
    parse_warning: str | None = None
    connector_resolution: str | None = None
    selected_mention_position: int | None = None
    provisional_party_id: str | None = None
    provisional_party_name: str | None = None
    mention_results: list[dict[str, Any]] = field(default_factory=list)
    emitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
