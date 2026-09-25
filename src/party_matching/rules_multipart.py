"""Optional, reversible rules-only choice for multi-party plain names.

This is deliberately separate from the ML/event pipeline and the frozen handoff.
Only TRAIN/CALIBRATION labels estimate a *delimiter-position* prior; names and
verified IDs are never memorized. TEST labels are used later for reporting only.
Distinct verified IDs remain distinct, even when they share a brand.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

from .domain import FinalDecision, MatchProposal, PartyRecord, normalize_name, parse_mentions, read_jsonl


SEPARATOR = re.compile(r"\s+(?:[-–—|/])\s+|\s+[-–—|/]\s*|\s*[-–—|/]\s+")
PARENS = re.compile(r"^(.+?)\s*\(([^()]+)\)\s*\.?$")


def multipart_fragments(raw_name: str) -> list[tuple[str, int, str]]:
    """Return (delimiter, position, text); do not split brand-internal hyphens."""
    value = raw_name.strip()
    parens = PARENS.match(value)
    if parens:
        pieces = [part.strip() for part in parens.groups()]
        delimiter = "parentheses"
    else:
        pieces = [part.strip() for part in SEPARATOR.split(value)]
        delimiter = "pipe" if "|" in value else "separator"
    pieces = [part for part in pieces if len(normalize_name(part)) >= 3]
    if not 2 <= len(pieces) <= 4:
        return []
    if len({normalize_name(part) for part in pieces}) < 2:
        return []
    return [(delimiter, index, part) for index, part in enumerate(pieces)]


def _best_strong_root(proposals: list[MatchProposal], minimum: float) -> MatchProposal | None:
    """Keep one eligible official/alias proposal per root, then the strongest."""
    eligible = [item for item in proposals
                if item.rules_score >= minimum
                and not item.digit_conflict and not item.distinctive_token_conflict
                and max(item.char_tfidf_score, item.word_tfidf_score, item.exact) >= 0.55]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (-item.rules_score, -item.exact,
                                           item.root_party_id, item.matched_name))


def _choice(
    fragments: list[tuple[str, int, str, MatchProposal]],
    prior: dict[str, list[float]],
) -> tuple[str, int, str, MatchProposal]:
    delimiter = fragments[0][0]
    probabilities = prior.get(delimiter, [])
    return min(fragments, key=lambda item: (
        -(item[3].rules_score + 0.08 * (probabilities[item[1]] if item[1] < len(probabilities) else 0.0)),
        -item[3].rules_score, item[1], item[3].root_party_id,
    ))


def choose_multipart_rules(
    records: list[PartyRecord], decisions: list[FinalDecision], graph: Any,
    scorer: Any, retrieval_config: dict[str, Any], workers: int,
    labels_path: Any, party_job: dict[str, dict], default_job: dict,
    minimum_confidence: float = 0.80,
) -> tuple[list[FinalDecision], dict[str, Any]]:
    """Retrieve fragments separately; make one choice only with strong evidence.

    No generic text-only rule can establish corporate ownership or decide which
    of two genuinely different legal entities an under-specified name denotes.
    """
    from .matching import MentionRetriever, collect_proposals

    by_id = {item.adm_party_id: item for item in decisions}
    metadata: dict[str, tuple[str, int, str]] = {}
    synthetic: list[PartyRecord] = []
    for record in records:
        mentions = parse_mentions(record)
        if len(mentions) != 1 or mentions[0].parse_warning:
            continue  # OBO/VIA and malformed connector policy is untouched.
        for delimiter, position, fragment in multipart_fragments(record.raw_name):
            fragment_id = f"{record.adm_party_id}:multipart:{position}"
            metadata[fragment_id] = (record.adm_party_id, position, delimiter)
            synthetic.append(PartyRecord(record.account_id, fragment_id, fragment, record.source_row))
    if not synthetic:
        return decisions, {"fragment_rows": 0, "competing_rows": 0, "changed": 0}

    # A dedicated reverse index lets each fragment retrieve candidates absent
    # from the original whole-name shortlist. It never changes the baseline index.
    fragment_config = {**retrieval_config, "embedding_enabled": False}
    fragment_retriever = MentionRetriever(synthetic, fragment_config, workers=workers)
    fragment_proposals, retrieval_stats = collect_proposals(graph, fragment_retriever, scorer)
    if not 0.0 <= float(minimum_confidence) <= 1.0:
        raise ValueError("rules_multipart_min_confidence must be between 0 and 1")
    minimum = max(float(minimum_confidence), float(scorer.plain_threshold or 0.0))
    alternatives: dict[str, list[tuple[str, int, str, MatchProposal]]] = defaultdict(list)
    for fragment in synthetic:
        best = _best_strong_root(fragment_proposals.get(fragment.adm_party_id, []), minimum)
        if best is not None:
            original_id, position, delimiter = metadata[fragment.adm_party_id]
            alternatives[original_id].append((delimiter, position, fragment.raw_name, best))

    labels = {row["adm_party_id"]: row for row in read_jsonl(labels_path)
              if row.get("scorable") and row.get("split") in {"TRAIN", "CALIBRATION"}}
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    competing = {
        adm_id: choices for adm_id, choices in alternatives.items()
        if len({item[3].root_party_id for item in choices}) >= 2
    }
    for adm_id, choices in competing.items():
        label = labels.get(adm_id)
        if label is None:
            continue
        expected = label.get("expected_party_id")
        if expected in graph.nodes:
            expected = graph.root_id(expected)
        positions = {item[1] for item in choices if item[3].root_party_id == expected}
        if len(positions) != 1:
            continue
        delimiter = choices[0][0]
        counts[delimiter][next(iter(positions))] += 1
        totals[delimiter] += 1
    # With fewer than 20 unambiguous calibration examples, use no learned
    # position preference. This avoids a handful of rows steering a whole class.
    prior: dict[str, list[float]] = {}
    for delimiter, total in totals.items():
        if total >= 20:
            prior[delimiter] = [(counts[delimiter][index] + 1) / (total + 4)
                                for index in range(4)]

    changed = new_matches = overrides = 0
    for adm_id, choices in competing.items():
        old = by_id[adm_id]
        # Complete official full-name matches have stronger evidence than any
        # fragment and must not be displaced by a delimiter preference.
        if (old.decision == "MATCH" and old.matched_mention == old.raw_name
                and old.matched_candidate_type == "official" and old.confidence >= 0.93
                and normalize_name(old.raw_name) == normalize_name(old.matched_candidate_name)):
            continue
        _, _, fragment, winner = _choice(choices, prior)
        job = party_job.get(str(winner.owner_party_id), default_job)
        if winner.rules_score < max(minimum, float(job.get("confidenceCutoff", 0.0))):
            continue
        if old.decision == "MATCH" and old.verified_party_id == winner.root_party_id:
            continue
        by_id[adm_id] = FinalDecision(
            account_id=old.account_id, adm_party_id=adm_id, raw_name=old.raw_name,
            decision="MATCH", confidence=winner.rules_score,
            reason="MULTIPART_FORCED_CHOICE", verified_party_id=winner.root_party_id,
            verified_party_name=winner.root_party_name,
            matched_member_id=winner.owner_party_id, matched_member_name=winner.owner_party_name,
            matched_candidate_name=winner.matched_name, matched_candidate_type=winner.candidate_type,
            candidate_expansion_confidence=winner.candidate_confidence,
            matched_mention=fragment, match_method="rules:multipart_forced_choice",
            decision_tier="RULES_MULTIPART_FORCED_CHOICE",
            retrieval_sources=winner.retrieval_sources,
            identity_score=winner.rules_score, char_tfidf_score=winner.char_tfidf_score,
            word_tfidf_score=winner.word_tfidf_score, rrf_score=winner.rrf_score,
            char_similarity=winner.char_similarity, jaro_winkler=winner.jaro_winkler,
            levenshtein=winner.levenshtein, token_jaccard=winner.token_jaccard,
            raw_coverage=winner.raw_coverage, candidate_coverage=winner.candidate_coverage,
            distinctive_token_conflict=winner.distinctive_token_conflict,
            digit_conflict=winner.digit_conflict,
            graph_path=graph.path_to_root(winner.owner_party_id),
            retrieved_root_ids=old.retrieved_root_ids,
        )
        changed += 1
        new_matches += int(old.decision != "MATCH")
        overrides += int(old.decision == "MATCH")
    return [by_id[item.adm_party_id] for item in decisions], {
        "fragment_rows": len({item[0] for item in metadata.values()}),
        "competing_rows": len(competing), "changed": changed,
        "new_matches": new_matches, "overrides": overrides,
        "learned_position_counts": {key: dict(value) for key, value in counts.items()},
        "learned_position_prior": prior,
        "scored_proposals": retrieval_stats.get("scored_proposals", 0),
        "retrieval_seconds": retrieval_stats.get("retrieval_seconds", 0.0),
    }
