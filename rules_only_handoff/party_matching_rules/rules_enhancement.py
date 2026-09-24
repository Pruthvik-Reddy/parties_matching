"""Conservative plain-name rules recovery, independent of the trained ML path.

Only previously rejected plain records are reconsidered. No labels are read,
and a view may only strengthen a verified root retrieved for the full name.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any

import numpy as np

from .domain import LEGAL_SUFFIXES, FinalDecision, MatchProposal, PartyRecord, compact_name, normalize_name, parse_mentions


SEPARATOR = re.compile(r"-{2,}|\s+[-–—|/]\s*|\s*[-–—|/]\s+")
TRAILING_PARENS = re.compile(r"^(.+?)\s*\(([^()]+)\)\s*\.?$")
BRAND_CASE = re.compile(r"[a-z][A-Z]")
FUNCTION_WORDS = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "the", "to"}


def name_views(raw_name: str) -> list[tuple[str, str, str]]:
    """Return (kind, view, removed text), bounded and company-name agnostic."""
    views: list[tuple[str, str, str]] = []
    parts = [part.strip() for part in SEPARATOR.split(raw_name) if part.strip()]
    if len(parts) > 1:
        views.append(("first_segment", parts[0], " ".join(parts[1:])))
        views.append(("last_segment", parts[-1], " ".join(parts[:-1])))
    parenthetical = TRAILING_PARENS.match(raw_name.strip())
    if parenthetical:
        views.append(("without_parenthetical", parenthetical.group(1).strip(), parenthetical.group(2).strip()))
    tokens = normalize_name(raw_name).split()
    if len(tokens) == 2 and all(3 <= len(token) <= 12 for token in tokens):
        views.append(("compact_spacing", compact_name(raw_name), ""))
    if any(len(token) >= 6 and token.endswith("ies") for token in tokens):
        singular = " ".join(token[:-3] + "y" if len(token) >= 6 and token.endswith("ies") else token for token in tokens)
        views.append(("plural_ies", singular, ""))
    seen = {normalize_name(raw_name)}
    result = []
    for kind, view, removed in views:
        normalized = normalize_name(view)
        if len(normalized) < 3 or normalized in seen:
            continue
        seen.add(normalized)
        result.append((kind, view, removed))
        if len(result) == 3:
            break
    return result


def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    def singular(token: str) -> str:
        if len(token) >= 6 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) >= 5 and token.endswith("s"):
            return token[:-1]
        return token
    if singular(left) == singular(right):
        return 0.94
    if min(len(left), len(right)) < 5:
        return 0.0
    from .matching import levenshtein_similarity
    value = levenshtein_similarity(left, right)
    return value if value >= 0.78 else 0.0


def soft_token_coverage(left: str, right: str, idf: dict[str, float]) -> tuple[float, float]:
    """One-to-one, IDF-weighted token coverage in both directions."""
    left_tokens = [t for t in normalize_name(left).split() if t not in LEGAL_SUFFIXES | FUNCTION_WORDS]
    right_tokens = [t for t in normalize_name(right).split() if t not in LEGAL_SUFFIXES | FUNCTION_WORDS]
    if not left_tokens or not right_tokens:
        return 0.0, 0.0
    edges = sorted(
        ((similarity, i, j) for i, token in enumerate(left_tokens)
         for j, candidate in enumerate(right_tokens)
         if (similarity := _token_similarity(token, candidate)) > 0),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    left_credit = right_credit = 0.0
    for similarity, i, j in edges:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        left_credit += max(1.0, idf.get(left_tokens[i], 1.0)) * similarity
        right_credit += max(1.0, idf.get(right_tokens[j], 1.0)) * similarity
    left_total = sum(max(1.0, idf.get(token, 1.0)) for token in left_tokens)
    right_total = sum(max(1.0, idf.get(token, 1.0)) for token in right_tokens)
    return left_credit / left_total, right_credit / right_total


def _unsafe_removed(removed: str, candidate_root: str, proposals: list[MatchProposal], idf: dict[str, float]) -> bool:
    if not removed:
        return False
    # Internal capitalization is a brand-like signal (e.g. SpringCM), not a
    # company-specific alias list. Fiscal/code tokens such as FY26 lack it.
    if BRAND_CASE.search(removed):
        return True
    for item in proposals:
        if item.root_party_id == candidate_root:
            continue
        raw_cover, candidate_cover = soft_token_coverage(removed, item.matched_name, idf)
        if raw_cover >= 0.75 and candidate_cover >= 0.75:
            return True
    return False


def _root_token_index(catalog: Any) -> dict[str, set[str]]:
    roots: dict[str, set[str]] = defaultdict(set)
    for party_id, party in catalog.parties.items():
        for token in normalize_name(party.party_name).split():
            if len(token) >= 5 and token not in LEGAL_SUFFIXES | FUNCTION_WORDS:
                roots[token].add(party_id)
    return roots


def _unique_official_short_token(
    mention_text: str, proposal: MatchProposal, root_tokens: dict[str, set[str]],
    minimum_lexical: float,
) -> bool:
    """A single distinctive token must identify one verified root, not an alias."""
    tokens = normalize_name(mention_text).split()
    if len(tokens) != 1 or len(tokens[0]) < 5:
        return False
    token = tokens[0]
    return (
        root_tokens.get(token) == {proposal.root_party_id}
        and proposal.candidate_type == "official"
        and token in normalize_name(proposal.matched_name).split()
        and max(proposal.char_tfidf_score, proposal.word_tfidf_score) >= minimum_lexical
    )


def _unique_short_name(
    record: PartyRecord, baseline: FinalDecision, proposals: list[MatchProposal],
    root_tokens: dict[str, set[str]], request_cutoff: float,
) -> FinalDecision | None:
    tokens = normalize_name(record.raw_name).split()
    if len(tokens) != 1 or len(tokens[0]) < 5 or baseline.verified_party_id is None:
        return None
    token = tokens[0]
    if root_tokens.get(token) != {baseline.verified_party_id}:
        return None
    if baseline.margin is None or baseline.margin < 0.08 or baseline.confidence < request_cutoff:
        return None
    matching = [item for item in proposals if item.root_party_id == baseline.verified_party_id
                and _unique_official_short_token(record.raw_name, item, root_tokens, 0.50)]
    if not matching:
        return None
    winner = copy.copy(baseline)
    winner.decision = winner.reason = "MATCH"
    winner.decision_tier = "RULES_ROOT_UNIQUE_SHORT_NAME"
    winner.match_method = "rules:root_unique_short_name"
    return winner


def _pair_cosines(proposals: list[MatchProposal], vectorizer: Any) -> np.ndarray:
    if vectorizer is None or not proposals:
        return np.zeros(len(proposals), dtype=np.float32)
    left = vectorizer.transform([item.mention_text for item in proposals])
    right = vectorizer.transform([item.matched_name for item in proposals])
    return np.asarray(left.multiply(right).sum(axis=1)).ravel().astype(np.float32)


def _anchored_official_prefix(raw_name: str, official_name: str) -> tuple[str, str] | None:
    """Find a complete official name at the start, followed by extra context."""
    official_tokens = normalize_name(official_name).split()
    informative = [token for token in official_tokens if token not in LEGAL_SUFFIXES | FUNCTION_WORDS]
    if len(informative) < 2 and not (
        len(informative) == 1 and len(informative[0]) >= 8
        and any(token in LEGAL_SUFFIXES for token in official_tokens)
    ):
        return None
    normalized_official = " ".join(official_tokens)
    for word in re.finditer(r"[^\W_]+", raw_name, flags=re.UNICODE):
        prefix = raw_name[:word.end()]
        normalized_prefix = normalize_name(prefix)
        if normalized_prefix == normalized_official:
            remainder = raw_name[word.end():].strip(" \t.,:;-/–—|()")
            return (prefix, remainder) if len(normalize_name(remainder)) >= 2 else None
        if len(normalized_prefix.split()) > len(official_tokens) + 2:
            break
    return None


def _unsafe_anchor_remainder(
    remainder: str, chosen_root: str, proposals: list[MatchProposal], idf: dict[str, float],
) -> bool:
    if _unsafe_removed(remainder, chosen_root, proposals, idf):
        return True
    # A full competing official name inside a longer suffix can be diluted in
    # whole-suffix coverage (e.g. "as successor to Acme Software Limited").
    suffix = normalize_name(remainder)
    for item in proposals:
        if item.root_party_id == chosen_root or item.candidate_type != "official":
            continue
        words = [token for token in normalize_name(item.matched_name).split()
                 if token not in LEGAL_SUFFIXES | FUNCTION_WORDS]
        if len(words) >= 2 and f" {' '.join(words)} " in f" {suffix} ":
            return True
    return False


def _recover_core_anchors(
    records: list[PartyRecord], proposals_by_record: dict[str, list[MatchProposal]],
    enhanced: dict[str, FinalDecision], catalog: Any, retriever: Any, scorer: Any,
    config: dict[str, Any], party_job: dict[str, dict], default_job: dict,
) -> tuple[int, int, int]:
    """Supplement, but never replace, matches accepted by the existing rules."""
    from .matching import CrossEncoderReranker, decide_records

    pending: list[tuple[PartyRecord, list[MatchProposal]]] = []
    blocked = 0
    for record in records:
        previous = enhanced[record.adm_party_id]
        if previous.decision != "NO_MATCH" or not previous.verified_party_id:
            continue
        mentions = parse_mentions(record)
        if len(mentions) != 1 or mentions[0].parse_warning or (previous.margin or 0.0) < 0.08:
            continue
        original = proposals_by_record.get(record.adm_party_id, [])
        owner = sorted(
            (item for item in original if item.root_party_id == previous.verified_party_id
             and item.candidate_type == "official" and item.candidate_collision_count == 1),
            key=lambda item: (-len(normalize_name(item.matched_name)), -item.feature_score),
        )[:2]
        clones = []
        for index, item in enumerate(owner):
            anchored = _anchored_official_prefix(record.raw_name, item.matched_name)
            if anchored is None:
                continue
            prefix, remainder = anchored
            if _unsafe_anchor_remainder(remainder, previous.verified_party_id, original, scorer.idf):
                blocked += 1
                continue
            clone = copy.copy(item)
            clone.mention_id = f"{record.adm_party_id}:rules_core:{index}"
            clone.mention_text = prefix
            clone.retrieval_sources = [*item.retrieval_sources, "rules_view:core_anchor"]
            clone.cross_encoder_score = None
            clone.embedding_score = None
            clone.rrf_score = 0.0
            clones.append(clone)
        if clones:
            pending.append((record, clones))

    scored = accepted = 0
    reranker = CrossEncoderReranker("unused-rules-core-model", "off")
    for start in range(0, len(pending), 250):
        batch = pending[start:start + 250]
        clones = [clone for _, items in batch for clone in items]
        scored += len(clones)
        chars = _pair_cosines(clones, retriever.char_vectorizer)
        words = _pair_cosines(clones, retriever.word_vectorizer)
        for item, char_score, word_score in zip(clones, chars, words):
            item.char_tfidf_score = float(char_score)
            item.word_tfidf_score = float(word_score)
            item.lexical_score = max(float(char_score), float(word_score))
            item.exact = float(normalize_name(item.mention_text) == normalize_name(item.matched_name))
        scorer.score_proposals(clones)
        augmented = {record.adm_party_id: [*proposals_by_record[record.adm_party_id], *items]
                     for record, items in batch}
        decisions = decide_records([record for record, _ in batch], augmented, catalog, scorer, reranker, config)
        for decision in decisions:
            previous = enhanced[decision.adm_party_id]
            if decision.decision != "MATCH" or decision.verified_party_id != previous.verified_party_id:
                continue
            if not any(item.adm_party_id == decision.adm_party_id
                       and item.mention_text == decision.matched_mention
                       and item.matched_name == decision.matched_candidate_name for item in clones):
                continue
            job = party_job.get(str(decision.matched_member_id or decision.verified_party_id), default_job)
            if decision.confidence < max(scorer.plain_threshold, float(job.get("confidenceCutoff", 0.0))):
                continue
            decision.decision_tier = "RULES_SAFE_VIEW_CORE_ANCHOR"
            decision.match_method = "rules:safe_view_core_anchor"
            enhanced[decision.adm_party_id] = decision
            accepted += 1
    return scored, accepted, blocked


def enhance_rules_decisions(
    records: list[PartyRecord], proposals_by_record: dict[str, list[MatchProposal]],
    baseline_decisions: list[FinalDecision], catalog: Any, retriever: Any,
    scorer: Any, config: dict[str, Any], party_job: dict[str, dict], default_job: dict,
) -> tuple[list[FinalDecision], dict[str, Any]]:
    """Recover guarded plain-name matches; preserve every baseline MATCH and connector."""
    from .matching import CrossEncoderReranker, decide_records

    baseline = {item.adm_party_id: item for item in baseline_decisions}
    enhanced = dict(baseline)
    root_tokens = _root_token_index(catalog)
    reviewed = view_candidates = short_matches = view_matches = unsafe_views = 0
    pending: list[tuple[PartyRecord, list[tuple[str, str, str]]]] = []
    for record in records:
        original = baseline[record.adm_party_id]
        if original.decision != "NO_MATCH" or not original.verified_party_id:
            continue
        mentions = parse_mentions(record)
        if len(mentions) != 1 or mentions[0].parse_warning or not proposals_by_record.get(record.adm_party_id):
            continue
        reviewed += 1
        job = party_job.get(str(original.matched_member_id or original.verified_party_id), default_job)
        short = _unique_short_name(record, original, proposals_by_record[record.adm_party_id],
                                   root_tokens, float(job.get("confidenceCutoff", 0.0)))
        if short is not None:
            enhanced[record.adm_party_id] = short
            short_matches += 1
            continue
        views = name_views(record.raw_name)
        if views:
            pending.append((record, views))

    # Fixed-size batches avoid another 200K-row reverse index and keep memory
    # bounded. Only the baseline winner's already-retrieved root is rescored.
    reranker = CrossEncoderReranker("unused-rules-view-model", "off")
    for start in range(0, len(pending), 250):
        batch = pending[start:start + 250]
        augmented: dict[str, list[MatchProposal]] = {}
        cloned: list[MatchProposal] = []
        view_meta: dict[str, str] = {}
        for record, views in batch:
            adm_id = record.adm_party_id
            current = baseline[adm_id]
            original_proposals = proposals_by_record[adm_id]
            owner = sorted((item for item in original_proposals if item.root_party_id == current.verified_party_id),
                           key=lambda item: (-item.feature_score, item.matched_name))[:2]
            candidates = list(original_proposals)
            for index, (kind, view, removed) in enumerate(views):
                if kind == "last_segment":
                    continue  # Plain-name last segments are too often another organization.
                if _unsafe_removed(removed, current.verified_party_id, original_proposals, scorer.idf):
                    unsafe_views += 1
                    continue
                for item in owner:
                    clone = copy.copy(item)
                    clone.mention_id = f"{adm_id}:rules_view:{index}"
                    clone.mention_text = view
                    clone.retrieval_sources = [*item.retrieval_sources, f"rules_view:{kind}"]
                    clone.cross_encoder_score = None
                    clone.embedding_score = None
                    clone.rrf_score = 0.0
                    cloned.append(clone)
                    candidates.append(clone)
                    view_meta[clone.mention_id] = kind
            augmented[adm_id] = candidates
        view_candidates += len(cloned)
        if not cloned:
            continue
        for offset in range(0, len(cloned), 5_000):
            scoring = cloned[offset:offset + 5_000]
            chars = _pair_cosines(scoring, retriever.char_vectorizer)
            words = _pair_cosines(scoring, retriever.word_vectorizer)
            for item, char_score, word_score in zip(scoring, chars, words):
                item.char_tfidf_score = float(char_score)
                item.word_tfidf_score = float(word_score)
                item.lexical_score = max(float(char_score), float(word_score))
                item.exact = float(normalize_name(item.mention_text) == normalize_name(item.matched_name))
            scorer.score_proposals(scoring)
        decisions = decide_records([item[0] for item in batch], augmented, catalog, scorer, reranker, config)
        for decision in decisions:
            if decision.decision != "MATCH" or decision.matched_mention == decision.raw_name:
                continue
            if decision.verified_party_id != baseline[decision.adm_party_id].verified_party_id:
                continue
            matching_view = next((item for item in cloned if item.adm_party_id == decision.adm_party_id
                                  and item.mention_text == decision.matched_mention
                                  and item.root_party_id == decision.verified_party_id), None)
            if matching_view is None:
                continue
            kind = view_meta[matching_view.mention_id]
            view_coverage = soft_token_coverage(decision.matched_mention, decision.matched_candidate_name or "", scorer.idf)
            # This view passed the removed-text guard before scoring; don't
            # repeat that O(candidate-count) scan for the selected result.
            if min(view_coverage) < 0.75:
                continue
            job = party_job.get(str(decision.matched_member_id or decision.verified_party_id), default_job)
            if decision.confidence < max(scorer.plain_threshold, float(job.get("confidenceCutoff", 0.0))):
                continue
            decision.decision_tier = f"RULES_SAFE_VIEW_{kind.upper()}"
            decision.match_method = f"rules:safe_view_{kind}"
            enhanced[decision.adm_party_id] = decision
            view_matches += 1
    core_scored, core_matches, core_blocked = _recover_core_anchors(
        records, proposals_by_record, enhanced, catalog, retriever, scorer, config, party_job, default_job,
    )
    result = [enhanced[item.adm_party_id] for item in records]
    return result, {"reviewed_plain_no_matches": reviewed, "view_candidates_scored": view_candidates,
                    "unsafe_views_skipped": unsafe_views, "short_name_matches": short_matches,
                    "safe_view_matches": view_matches, "core_anchor_candidates_scored": core_scored,
                    "core_anchor_matches": core_matches, "unsafe_core_anchors_skipped": core_blocked}
