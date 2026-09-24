"""Conservative rules recovery, independent of the trained ML path.

Only previously rejected plain records are reconsidered by the view stage.
The official-name uniqueness helpers also serve connector decisions. No labels
are read, and a view may only strengthen a root retrieved for the full name.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any

import numpy as np

from .domain import LEGAL_SUFFIXES, FinalDecision, MatchProposal, PartyRecord, base_name, compact_name, normalize_name, parse_mentions


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


def _root_token_index(graph: Any) -> dict[str, set[str]]:
    roots: dict[str, set[str]] = defaultdict(set)
    for party_id, node in graph.nodes.items():
        root = graph.root_id(party_id)
        for token in normalize_name(node.party_name).split():
            if len(token) >= 5 and token not in LEGAL_SUFFIXES | FUNCTION_WORDS:
                roots[token].add(root)
    return roots


def _root_prefix_index(graph: Any, names: set[str]) -> dict[str, set[str]]:
    """Index only queried official prefixes, not every possible graph n-gram."""
    wanted = {normalized for name in names if len((normalized := normalize_name(name)).split()) >= 2}
    roots: dict[str, set[str]] = defaultdict(set)
    if not wanted:
        return roots
    for party_id, node in graph.nodes.items():
        tokens = normalize_name(node.party_name).split()
        root_id = graph.root_id(party_id)
        for length in range(2, len(tokens) + 1):
            prefix = " ".join(tokens[:length])
            if prefix in wanted:
                roots[prefix].add(root_id)
    return roots


def _unique_official_short_name(
    mention_text: str, proposal: MatchProposal, root_tokens: dict[str, set[str]],
    root_prefixes: dict[str, set[str]], minimum_lexical: float, allow_multiword: bool = True,
) -> bool:
    """A short official-name prefix must identify one verified root, not an alias."""
    tokens = normalize_name(mention_text).split()
    if not tokens or proposal.candidate_type != "official":
        return False
    official = normalize_name(proposal.matched_name).split()
    if len(tokens) == 1:
        if (len(tokens[0]) < 5 or root_tokens.get(tokens[0]) != {proposal.root_party_id}
                or tokens[0] not in official):
            return False
    else:
        if not allow_multiword:
            return False
        if proposal.candidate_collision_count != 1 or official[:len(tokens)] != tokens:
            return False
        # A complete multiword stem may omit only a short legal/descriptive
        # ending. All verified nodes sharing the stem must resolve to one root.
        if len("".join(tokens)) < 8 or len(official) - len(tokens) > 2:
            return False
        if root_prefixes.get(" ".join(tokens)) != {proposal.root_party_id}:
            return False
    return (
        max(proposal.char_tfidf_score, proposal.word_tfidf_score) >= minimum_lexical
    )


def _unique_short_name(
    record: PartyRecord, baseline: FinalDecision, proposals: list[MatchProposal],
    root_tokens: dict[str, set[str]], root_prefixes: dict[str, set[str]], request_cutoff: float,
    allow_multiword: bool = True, multiword_floor: float = 0.0,
) -> FinalDecision | None:
    multiword = len(normalize_name(record.raw_name).split()) > 1
    if baseline.verified_party_id is None or (multiword and baseline.reason != "INSUFFICIENT_SUPPORT"):
        return None
    if baseline.margin is None or baseline.margin < 0.08 or baseline.confidence < request_cutoff:
        return None
    if multiword and baseline.confidence < multiword_floor:
        return None
    matching = [item for item in proposals if item.root_party_id == baseline.verified_party_id
                and (not multiword or (not item.digit_conflict and not item.distinctive_token_conflict))
                and _unique_official_short_name(
                    record.raw_name, item, root_tokens, root_prefixes, 0.50, allow_multiword,
                )]
    if not matching:
        return None
    winner = copy.copy(baseline)
    winner.decision = winner.reason = "MATCH"
    winner.decision_tier = "RULES_ROOT_UNIQUE_OFFICIAL_PREFIX" if multiword else "RULES_ROOT_UNIQUE_SHORT_NAME"
    winner.match_method = "rules:root_unique_official_prefix" if multiword else "rules:root_unique_short_name"
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
    enhanced: dict[str, FinalDecision], graph: Any, retriever: Any, scorer: Any,
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
        decisions = decide_records([record for record, _ in batch], augmented, graph, scorer, reranker, config)
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


def _root_base_index(graph: Any) -> dict[str, set[str]]:
    """Keep legal-suffix equivalence distinct from verified-root identity."""
    roots: dict[str, set[str]] = defaultdict(set)
    for party_id, node in graph.nodes.items():
        base = base_name(node.party_name)
        if base:
            roots[base].add(graph.root_id(party_id))
    return roots


def _minor_spelling_variant(raw: str, official: str, idf: dict[str, float]) -> bool:
    """One small token edit in an otherwise complete name; never a missing word."""
    raw_tokens, official_tokens = base_name(raw).split(), base_name(official).split()
    if len(raw_tokens) != len(official_tokens) or not raw_tokens:
        return False
    if len(raw_tokens) == 1 and min(len(raw_tokens[0]), len(official_tokens[0])) < 8:
        return False
    differences = [(left, right) for left, right in zip(raw_tokens, official_tokens) if left != right]
    if len(differences) != 1:
        return False
    left, right = differences[0]
    if min(len(left), len(right)) < 5 or _token_similarity(left, right) < 0.80:
        return False
    coverage = soft_token_coverage(raw, official, idf)
    return min(coverage) >= 0.88


def _second_pass_evidence(
    record: PartyRecord, item: MatchProposal, proposals: list[MatchProposal],
    root_tokens: dict[str, set[str]], root_prefixes: dict[str, set[str]],
    root_bases: dict[str, set[str]], idf: dict[str, float],
) -> tuple[int, str] | None:
    """Structural evidence for one already-retrieved verified root, strongest first."""
    if (item.candidate_type != "official" or item.candidate_collision_count != 1
            or item.digit_conflict):
        return None
    raw = record.raw_name
    anchored = _anchored_official_prefix(raw, item.matched_name)
    if anchored is not None and item.rules_score >= 0.25:
        _, remainder = anchored
        if not _unsafe_anchor_remainder(remainder, item.root_party_id, proposals, idf):
            return 4, "OFFICIAL_NAME_WITH_CONTEXT"
    raw_base = base_name(raw)
    official_base = base_name(item.matched_name)
    if (raw_base and raw_base == official_base and normalize_name(raw) != normalize_name(item.matched_name)
            and len(raw_base.replace(" ", "")) >= 8
            and root_bases.get(raw_base) == {item.root_party_id}
            and item.rules_score >= 0.55):
        return 3, "LEGAL_SUFFIX_VARIANT"
    if (not item.distinctive_token_conflict and item.rules_score >= 0.40
            and _unique_official_short_name(raw, item, root_tokens, root_prefixes, 0.50)):
        return 3, "DISTINCTIVE_SHORT_NAME"
    if item.rules_score >= 0.35 and _minor_spelling_variant(raw, item.matched_name, idf):
        return 2, "MINOR_SPELLING_VARIANT"
    return None


def _recover_second_pass(
    records: list[PartyRecord], proposals_by_record: dict[str, list[MatchProposal]],
    enhanced: dict[str, FinalDecision], graph: Any, scorer: Any,
    party_job: dict[str, dict], default_job: dict,
    root_tokens: dict[str, set[str]], root_prefixes: dict[str, set[str]],
) -> tuple[dict[str, FinalDecision], dict[str, int]]:
    """Revisit remaining plain NO_MATCH rows across retrieved roots only.

    This is deliberately a separate, reversible rules stage. Labels and model
    predictions are not read. Accepted matches retain the original rules score
    rather than receiving an invented high confidence.
    """
    previous: dict[str, FinalDecision] = {}
    counts: dict[str, int] = defaultdict(int)
    root_bases = _root_base_index(graph)
    for record in records:
        old = enhanced[record.adm_party_id]
        if old.decision != "NO_MATCH":
            continue
        mentions = parse_mentions(record)
        if len(mentions) != 1 or mentions[0].parse_warning:
            continue
        proposals = proposals_by_record.get(record.adm_party_id, [])
        if not proposals:
            continue
        counts["reviewed"] += 1
        evidence: dict[str, tuple[int, MatchProposal, str]] = {}
        for item in proposals:
            found = _second_pass_evidence(
                record, item, proposals, root_tokens, root_prefixes, root_bases, scorer.idf,
            )
            if found is None:
                continue
            strength, kind = found
            current = evidence.get(item.root_party_id)
            if current is None or (strength, item.rules_score) > (current[0], current[1].rules_score):
                evidence[item.root_party_id] = (strength, item, kind)
        if not evidence:
            continue
        ranked = sorted(evidence.values(), key=lambda entry: (-entry[0], -entry[1].rules_score, entry[1].root_party_id))
        strength, winner, kind = ranked[0]
        # A second plausible organization must not be silently displaced by a
        # strong substring or typo. This includes an exact competing root.
        if len(ranked) > 1 and ranked[1][0] >= strength - 1:
            counts["ambiguous"] += 1
            continue
        if any(item.root_party_id != winner.root_party_id and item.exact for item in proposals):
            counts["competing_exact"] += 1
            continue
        job = party_job.get(str(winner.owner_party_id), default_job)
        if winner.rules_score < float(job.get("confidenceCutoff", 0.0)):
            counts["request_cutoff"] += 1
            continue
        previous[record.adm_party_id] = old
        enhanced[record.adm_party_id] = FinalDecision(
            account_id=record.account_id, adm_party_id=record.adm_party_id, raw_name=record.raw_name,
            decision="MATCH", confidence=winner.rules_score, reason="MATCH",
            verified_party_id=winner.root_party_id, verified_party_name=winner.root_party_name,
            matched_member_id=winner.owner_party_id, matched_member_name=winner.owner_party_name,
            matched_candidate_name=winner.matched_name, matched_candidate_type=winner.candidate_type,
            candidate_expansion_confidence=winner.candidate_confidence, matched_mention=winner.mention_text,
            match_method=f"rules:second_pass_{kind.lower()}", decision_tier=f"RULES_SECOND_PASS_{kind}",
            retrieval_sources=winner.retrieval_sources, identity_score=winner.rules_score,
            char_tfidf_score=winner.char_tfidf_score, word_tfidf_score=winner.word_tfidf_score,
            rrf_score=winner.rrf_score, char_similarity=winner.char_similarity,
            jaro_winkler=winner.jaro_winkler, levenshtein=winner.levenshtein,
            token_jaccard=winner.token_jaccard, raw_coverage=winner.raw_coverage,
            candidate_coverage=winner.candidate_coverage,
            distinctive_token_conflict=winner.distinctive_token_conflict, digit_conflict=winner.digit_conflict,
            graph_path=graph.path_to_root(winner.owner_party_id), retrieved_root_ids=old.retrieved_root_ids,
            runner_up_party_id=ranked[1][1].root_party_id if len(ranked) > 1 else None,
            runner_up_score=ranked[1][1].rules_score if len(ranked) > 1 else None,
            margin=winner.rules_score - ranked[1][1].rules_score if len(ranked) > 1 else old.margin,
        )
        counts["matches"] += 1
        counts[kind.lower()] += 1
    return previous, dict(counts)


def enhance_rules_decisions(
    records: list[PartyRecord], proposals_by_record: dict[str, list[MatchProposal]],
    baseline_decisions: list[FinalDecision], graph: Any, retriever: Any,
    scorer: Any, config: dict[str, Any], party_job: dict[str, dict], default_job: dict,
) -> tuple[list[FinalDecision], dict[str, Any]]:
    """Recover guarded plain-name matches; preserve every baseline MATCH and connector."""
    from .matching import CrossEncoderReranker, decide_records

    baseline = {item.adm_party_id: item for item in baseline_decisions}
    enhanced = dict(baseline)
    root_tokens = _root_token_index(graph)
    allow_multiword = bool(config.get("decision", {}).get("rules_multiword_prefix_enabled", False))
    root_prefixes = _root_prefix_index(
        graph, {record.raw_name for record in records
                if baseline[record.adm_party_id].decision == "NO_MATCH"},
    ) if allow_multiword else {}
    reviewed = view_candidates = short_matches = multiword_short_matches = view_matches = unsafe_views = 0
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
                                   root_tokens, root_prefixes, float(job.get("confidenceCutoff", 0.0)),
                                   allow_multiword, float(config.get("decision", {}).get(
                                       "rules_containment_min_confidence", 0.40,
                                   )))
        if short is not None:
            enhanced[record.adm_party_id] = short
            short_matches += 1
            multiword_short_matches += int(short.decision_tier == "RULES_ROOT_UNIQUE_OFFICIAL_PREFIX")
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
        decisions = decide_records([item[0] for item in batch], augmented, graph, scorer, reranker, config)
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
        records, proposals_by_record, enhanced, graph, retriever, scorer, config, party_job, default_job,
    )
    second_previous: dict[str, FinalDecision] = {}
    second_stats: dict[str, int] = {}
    if bool(config.get("decision", {}).get("rules_second_pass_enabled", False)):
        second_previous, second_stats = _recover_second_pass(
            records, proposals_by_record, enhanced, graph, scorer, party_job, default_job,
            root_tokens, root_prefixes,
        )
    result = [enhanced[item.adm_party_id] for item in records]
    return result, {"reviewed_plain_no_matches": reviewed, "view_candidates_scored": view_candidates,
                    "unsafe_views_skipped": unsafe_views, "short_name_matches": short_matches,
                    "multiword_short_name_matches": multiword_short_matches,
                    "safe_view_matches": view_matches, "core_anchor_candidates_scored": core_scored,
                    "core_anchor_matches": core_matches, "unsafe_core_anchors_skipped": core_blocked,
                    "second_pass": second_stats, "_second_pass_previous": second_previous}
