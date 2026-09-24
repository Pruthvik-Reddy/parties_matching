"""Opt-in, removable rules audit. Never used to choose matches or emit events."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .domain import FinalDecision, MatchProposal, PartyRecord, parse_mentions, read_jsonl
from .graph import VerifiedGraph


def write_rules_candidate_trace(
    path: Path,
    records: list[PartyRecord],
    proposals_by_record: dict[str, list[MatchProposal]],
    decisions_by_id: dict[str, FinalDecision],
    labels_path: Path,
    graph: VerifiedGraph,
    scorer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Trace missed, labeled plain names using the already-built rules proposals.

    TEMPORARY DIAGNOSTIC: remove this module, the CLI flag, and its run hook
    after root-ranking/decision analysis is complete. Recomputing *read-only*
    target scores keeps the production decision functions untouched. The file
    contains party names, so keep it with other confidential run artifacts.
    """
    # Local import avoids making normal matching runs import audit code.
    from .matching import _best_per_root, _combined_identity, _guard_reason, target_features

    labels = {
        str(row["adm_party_id"]): row for row in read_jsonl(labels_path)
        if row.get("scorable") and row.get("split") in {"CALIBRATION", "TEST_KNOWN", "TEST_UNSEEN"}
    }
    matching_cfg = config.get("matching", {})
    counts: Counter[str] = Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            label = labels.get(record.adm_party_id)
            if label is None:
                continue
            mentions = parse_mentions(record)
            if len(mentions) != 1 or mentions[0].parse_warning:
                continue  # Connector diagnosis is intentionally a separate phase.
            decision = decisions_by_id[record.adm_party_id]
            expected_id = str(label["expected_party_id"])
            expected_root = graph.root_id(expected_id) if expected_id in graph.nodes else expected_id
            counts["plain_labeled"] += 1
            if decision.decision == "MATCH" and decision.verified_party_id == expected_root:
                continue

            root_best = _best_per_root(proposals_by_record.get(record.adm_party_id, []))
            by_identity = sorted(root_best.values(), key=lambda item: (-_combined_identity(item), item.root_party_id))
            best_identity = _combined_identity(by_identity[0]) if by_identity else 0.0
            second_identity = _combined_identity(by_identity[1]) if len(by_identity) > 1 else 0.0
            vectors = [
                target_features(
                    proposal, second_identity if index == 0 else best_identity,
                    1, len(graph.path_to_root(proposal.owner_party_id)),
                )
                for index, proposal in enumerate(by_identity)
            ]
            scores = scorer.score_target_vectors(vectors)
            ranked = sorted(zip(scores, by_identity), key=lambda item: (-float(item[0]), item[1].root_party_id))
            eligible = [item for item in ranked if not _guard_reason(item[1], matching_cfg)]
            expected_item = next((item for item in ranked if item[1].root_party_id == expected_root), None)
            expected_guard = _guard_reason(expected_item[1], matching_cfg) if expected_item else None
            rank_all = next((index for index, item in enumerate(ranked, 1) if item[1].root_party_id == expected_root), None)
            rank_eligible = next((index for index, item in enumerate(eligible, 1) if item[1].root_party_id == expected_root), None)
            winner = (eligible or ranked)[0] if ranked else None

            if expected_item is None:
                bucket = "RETRIEVAL_MISS"
            elif expected_guard:
                bucket = "EXPECTED_GUARDED"
            elif rank_eligible != 1:
                bucket = "RANKING_LOSS"
            elif decision.reason in {"AMBIGUOUS_FINAL_TARGETS", "CANDIDATE_COLLISION"}:
                bucket = "AMBIGUITY"
            elif decision.reason in {"INSUFFICIENT_SUPPORT", "BELOW_REQUEST_CUTOFF"}:
                bucket = "SUPPORT_REJECTION"
            else:
                bucket = "OTHER"

            def describe(item: tuple[float, MatchProposal] | None) -> dict[str, Any] | None:
                if item is None:
                    return None
                score, proposal = item
                return {
                    "root_id": proposal.root_party_id,
                    "root_name": proposal.root_party_name,
                    "matched_name": proposal.matched_name,
                    "candidate_type": proposal.candidate_type,
                    "score": round(float(score), 6),
                    "identity_score": round(float(proposal.feature_score), 6),
                    "guard": _guard_reason(proposal, matching_cfg),
                    "collision_count": proposal.candidate_collision_count,
                    "exact": bool(proposal.exact),
                    "char_tfidf": round(float(proposal.char_tfidf_score), 6),
                    "word_tfidf": round(float(proposal.word_tfidf_score), 6),
                    "candidate_coverage": round(float(proposal.candidate_coverage), 6),
                    "token_jaccard": round(float(proposal.token_jaccard), 6),
                }

            trace = {
                "adm_party_id": record.adm_party_id,
                "raw_name": record.raw_name,
                "split": label["split"],
                "expected_root_id": expected_root,
                "expected_name": label.get("expected_canonical_name"),
                "decision": decision.decision,
                "reason": decision.reason,
                "decision_tier": decision.decision_tier,
                "decision_confidence": decision.confidence,
                "decision_margin": decision.margin,
                "rules_plain_threshold": scorer.plain_threshold,
                "provisional_root_id": decision.verified_party_id,
                "bucket": bucket,
                "candidate_count": len(ranked),
                "expected_rank_all": rank_all,
                "expected_rank_eligible": rank_eligible,
                "expected_candidate": describe(expected_item),
                "winner": describe(winner),
                "top_candidates": [describe(item) for item in ranked[:5]],
            }
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            counts["traced_misses"] += 1
            counts[bucket] += 1
    return {"path": str(path), "plain_labeled": counts["plain_labeled"],
            "traced_misses": counts["traced_misses"],
            "buckets": {key: value for key, value in counts.items()
                        if key not in {"plain_labeled", "traced_misses"}}}
