from __future__ import annotations

import hashlib
import json
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


CONNECTORS = re.compile(r"\b(OBO|VIA)\b", re.IGNORECASE)
LEGAL_SUFFIXES = {
    "ag", "bv", "co", "company", "corp", "corporation", "gmbh", "inc",
    "incorporated", "llc", "llp", "limited", "ltd", "nv", "plc", "pty",
    "sa", "sas", "sarl", "spa",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(Path(path).resolve())
    return config


def stable_id(namespace: str, *parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"party-matching:{namespace}:{value}"))


def content_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(target)


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temp.replace(target)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass
class PartyRecord:
    account_id: str
    adm_party_id: str
    raw_name: str
    source_row: int
    eligible: bool = True
    parent_id: str | None = None
    parent_is_verified: bool = False
    is_verified: bool = False


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
class ExpansionCandidate:
    name: str
    candidate_type: str = "alias"
    llm_confidence: float | None = None
    source: str = "llm"


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
    graph_path: list[str] = field(default_factory=list)
    retrieved_root_ids: list[str] = field(default_factory=list)
    parse_warning: str | None = None
    emitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
