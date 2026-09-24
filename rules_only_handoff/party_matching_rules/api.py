"""Public inference API for the standalone enhanced-rules matcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .domain import PartyRecord
from .graph import VerifiedGraph
from .matching import (CrossEncoderReranker, FeatureScorer, MentionRetriever,
                       _token_idf, collect_proposals, decide_records)
from .rules_enhancement import enhance_rules_decisions


@dataclass(frozen=True)
class MatcherSettings:
    # Fixed inference settings from the POC. These are rule cutoffs, not a
    # trained model or calibrated probabilities. No labels are read here.
    connector_cutoff: float = 0.933
    plain_cutoff: float = 0.800
    max_roots_per_mention: int = 50
    max_variants_per_root: int = 2


def match_parties(
    verified: Iterable[dict[str, Any]],
    unverified: Iterable[dict[str, Any]],
    settings: MatcherSettings = MatcherSettings(),
) -> list[dict[str, Any]]:
    """Match supplied names; return one decision per unverified input row.

    Verified rows: verified_id, verified_name, optional parent_id and aliases.
    Unverified rows: unverified_id, unverified_name. No ground truth is used.
    """
    if not 0 <= settings.plain_cutoff <= 1 or not 0 <= settings.connector_cutoff <= 1:
        raise ValueError("Rule cutoffs must be between 0 and 1")
    graph = VerifiedGraph(verified)
    input_rows = list(unverified)
    seen: set[str] = set()
    records = []
    for index, row in enumerate(input_rows, start=1):
        party_id = str(row.get("unverified_id") or f"u{index}").strip()
        if party_id in seen:
            raise ValueError(f"Duplicate unverified_id: {party_id}")
        seen.add(party_id)
        records.append(PartyRecord(
            account_id="csv", adm_party_id=party_id,
            raw_name=str(row.get("unverified_name") or ""), source_row=index,
        ))
    if not records:
        return []
    if not graph.nodes:
        return [
            {"unverified_id": record.adm_party_id, "unverified_name": record.raw_name,
             "decision": "NO_MATCH", "verified_id": "", "verified_name": "",
             "score": 0.0, "reason": "NO_VERIFIED_PARTIES", "decision_tier": "",
             "matched_mention": ""}
            for record in records
        ]

    retrieval = {
        "max_roots_per_mention": settings.max_roots_per_mention,
        "max_variants_per_root": settings.max_variants_per_root,
        "char_tfidf_enabled": True, "word_tfidf_enabled": True,
        "char_min_score": 0.28, "word_min_score": 0.18,
        "rrf_enabled": True, "rrf_k": 60,
        "embedding_enabled": False,
    }
    config = {
        "retrieval": retrieval,
        "matching": {
            "connector_policy": "positional", "cross_encoder_mode": "off",
            "minimum_root_margin": 0.04, "preferred_near_cutoff_band": 0.03,
            "distinctive_token_guard": True, "digit_conflict_guard": True,
            "directional_coverage_threshold": 0.90,
            "directional_jaccard_threshold": 0.45,
        },
        "decision": {
            "rules_enhanced_enabled": True,
            "rules_connector_short_name_enabled": True,
            "rules_containment_enabled": True,
            "rules_containment_min_confidence": 0.40,
            "rules_containment_min_margin": 0.08,
            "rules_containment_min_lexical": 0.50,
            "rules_containment_max_token_df": 3,
        },
    }
    retriever = MentionRetriever(records, retrieval, workers=1)
    idf = _token_idf([node.party_name for node in graph.nodes.values()])
    scorer = FeatureScorer(idf, settings.connector_cutoff, settings.plain_cutoff)
    proposals, _ = collect_proposals(graph, retriever, scorer)
    baseline = decide_records(records, proposals, graph, scorer, CrossEncoderReranker(), config)
    # Same order as the POC: first the ordinary rules decision, then guarded
    # short-name/name-view/core-anchor recovery for rejected plain names only.
    no_request_cutoff = {party_id: {"confidenceCutoff": 0.0} for party_id in graph.nodes}
    enhanced, _ = enhance_rules_decisions(
        records, proposals, baseline, graph, retriever, scorer, config,
        no_request_cutoff, {"confidenceCutoff": 0.0},
    )
    return [
        {
            "unverified_id": record.adm_party_id,
            "unverified_name": record.raw_name,
            "decision": decision.decision,
            "verified_id": decision.verified_party_id if decision.decision == "MATCH" else "",
            "verified_name": decision.verified_party_name if decision.decision == "MATCH" else "",
            "score": round(float(decision.confidence), 6),
            "reason": decision.reason,
            "decision_tier": decision.decision_tier or "",
            "matched_mention": decision.matched_mention or "",
        }
        for record, decision in zip(records, enhanced)
    ]
