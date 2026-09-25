"""Reversible rules-only recovery from a unique verified-name anchor.

This stage only revisits plain-name NO_MATCH decisions. It searches official
verified names directly, so a correct anchor is not lost merely because a
different candidate won the original whole-name retrieval. It does not infer
ownership: separate verified roots with the same anchor remain ambiguous.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from .domain import (
    LEGAL_SUFFIXES, FinalDecision, MatchProposal, PartyRecord, normalize_name,
    parse_mentions,
)
from .rules_enhancement import _pair_cosines


WORDS = re.compile(r"[^\W_]+", re.UNICODE)
ARTICLES = {"the", "a", "an"}


def _official_base(name: str) -> str:
    """Strip only a trailing legal designator, including bank-specific N.A."""
    tokens = normalize_name(name).split()
    while tokens and (tokens[-1] in LEGAL_SUFFIXES
                      or (tokens[-1] == "na" and "bank" in tokens[:-1])):
        tokens.pop()
    while tokens and tokens[0] in ARTICLES:
        tokens.pop(0)
    return " ".join(tokens)


def _raw_prefixes(name: str) -> list[tuple[str, str, str]]:
    """Return (written prefix, normalized prefix, remainder) at word boundaries."""
    result = []
    for token in list(WORDS.finditer(name))[:25]:
        prefix = name[:token.end()].strip()
        normalized = normalize_name(prefix)
        if normalized and (not result or result[-1][1] != normalized):
            result.append((prefix, normalized, name[token.end():].strip(" \t.,:;-/–—|()")))
    return result


def _context_allowed(anchor: str, remainder: str, exact_base: bool) -> bool:
    if exact_base:
        return len(normalize_name(anchor).replace(" ", "")) >= 3
    tokens = normalize_name(remainder).split()
    if not 1 <= len(tokens) <= 8 or any(token.isdigit() for token in tokens):
        return False
    core = normalize_name(anchor).replace(" ", "")
    # A short context-bearing anchor must look like an acronym. Bare short
    # names (for example Ripe -> Ripe Limited) are handled separately.
    if len(core) <= 4 and not (len(core) >= 3 and anchor.isupper()):
        return False
    return True


def recover_verified_name_anchors(
    records: list[PartyRecord], decisions: list[FinalDecision],
    graph: Any, proposals_by_record: dict[str, list[MatchProposal]],
    retriever: Any, scorer: Any, party_job: dict[str, dict], default_job: dict,
) -> tuple[list[FinalDecision], dict[str, int]]:
    """Score generated name views, then accept one unambiguous verified root."""
    full_index: dict[str, dict[str, str]] = defaultdict(dict)
    base_index: dict[str, dict[str, str]] = defaultdict(dict)
    for party_id, node in graph.nodes.items():
        root = graph.root_id(party_id)
        full_index[normalize_name(node.party_name)].setdefault(root, party_id)
        base_index[_official_base(node.party_name)].setdefault(root, party_id)

    by_id = {item.adm_party_id: item for item in decisions}
    pending: dict[str, list[tuple[int, str, str, MatchProposal]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        old = by_id[record.adm_party_id]
        if old.decision != "NO_MATCH":
            continue
        mentions = parse_mentions(record)
        if len(mentions) != 1 or mentions[0].parse_warning:
            continue
        counts["reviewed"] += 1
        raw_normalized = normalize_name(record.raw_name)
        for written, normalized, remainder in _raw_prefixes(record.raw_name):
            # A complete official-name prefix is stronger than a base-only
            # prefix; neither may span more than one verified root.
            for tier, kind, index in (
                (3, "OFFICIAL_PREFIX", full_index),
                (2 if not remainder else 1, "LEGAL_BASE" if not remainder else "BASE_WITH_CONTEXT", base_index),
            ):
                owners = index.get(normalized, {})
                if len(owners) != 1:
                    continue
                root, party_id = next(iter(owners.items()))
                if kind == "OFFICIAL_PREFIX" and not remainder:
                    continue  # Existing exact-name path owns full-name cases.
                if kind != "OFFICIAL_PREFIX" and not _context_allowed(written, remainder, not remainder):
                    continue
                if kind == "OFFICIAL_PREFIX" and not _context_allowed(written, remainder, False):
                    continue
                # Another, more specific official name spanning the raw name
                # is evidence against discarding the remainder.
                if any(
                    other_root != root and len(other_name) > len(normalized)
                    and (raw_normalized == other_name or raw_normalized.startswith(other_name + " "))
                    for _, other_name, _ in _raw_prefixes(record.raw_name)
                    for other_root in full_index.get(other_name, {})
                ):
                    counts["more_specific_competitor"] += 1
                    continue
                # An independently verified exact name is also a competitor
                # for the legal-base route (Ripe versus Ripe Limited).
                if kind != "OFFICIAL_PREFIX" and any(
                    other_root != root for other_root in full_index.get(raw_normalized, {})
                ):
                    counts["exact_competitor"] += 1
                    continue
                owner = graph.nodes[party_id]
                root_node = graph.nodes[root]
                score_name = owner.party_name if kind == "OFFICIAL_PREFIX" else normalized
                proposal = MatchProposal(
                    adm_party_id=record.adm_party_id,
                    mention_id=f"{record.adm_party_id}:rules_anchor:{len(pending[record.adm_party_id])}",
                    mention_text=written, connector_before=None, connector_after=None,
                    owner_party_id=party_id, owner_party_name=owner.party_name,
                    root_party_id=root, root_party_name=root_node.party_name,
                    matched_name=score_name,
                    candidate_type="official" if kind == "OFFICIAL_PREFIX" else "anchor_view",
                    candidate_source="official" if kind == "OFFICIAL_PREFIX" else "derived_legal_base",
                    candidate_confidence=None if kind == "OFFICIAL_PREFIX" else 0.90,
                    retrieval_sources=["rules_anchor:verified_catalog"],
                    exact=float(normalize_name(written) == normalize_name(score_name)),
                )
                pending[record.adm_party_id].append((tier, kind, remainder, proposal))

    candidates = [entry[3] for entries in pending.values() for entry in entries]
    counts["candidate_views"] = len(candidates)
    for start in range(0, len(candidates), 5_000):
        batch = candidates[start:start + 5_000]
        chars = _pair_cosines(batch, retriever.char_vectorizer)
        words = _pair_cosines(batch, retriever.word_vectorizer)
        for item, char_score, word_score in zip(batch, chars, words):
            item.char_tfidf_score = float(char_score)
            item.word_tfidf_score = float(word_score)
            item.lexical_score = max(float(char_score), float(word_score))
        scorer.score_proposals(batch)

    for adm_id, entries in pending.items():
        old = by_id[adm_id]
        # Keep the strongest structural tier. Multiple roots at that tier are
        # a genuine ambiguity, never a numerical-score tie-break.
        top_tier = max(entry[0] for entry in entries)
        top = [entry for entry in entries if entry[0] == top_tier]
        roots = {entry[3].root_party_id for entry in top}
        if len(roots) != 1:
            counts["ambiguous_roots"] += 1
            continue
        _, kind, remainder, winner = max(
            top, key=lambda entry: (entry[3].rules_score, len(normalize_name(entry[3].mention_text)),
                                    entry[3].matched_name),
        )
        root = winner.root_party_id
        if winner.digit_conflict or winner.distinctive_token_conflict:
            counts["conflict_guard"] += 1
            continue
        # A complete legal name at the start can outrank a short context term,
        # but a different exact *whole-name* official candidate cannot.
        if any(item.root_party_id != root and item.exact and
               normalize_name(item.mention_text) == normalize_name(old.raw_name)
               for item in proposals_by_record.get(adm_id, [])):
            counts["exact_competitor"] += 1
            continue
        cap = 0.95 if kind == "OFFICIAL_PREFIX" else 0.90
        confidence = min(winner.rules_score, cap)
        job = party_job.get(winner.owner_party_id, default_job)
        if confidence < max(float(scorer.plain_threshold or 0.0),
                            float(job.get("confidenceCutoff", 0.0))):
            counts["below_cutoff"] += 1
            continue
        by_id[adm_id] = FinalDecision(
            account_id=old.account_id, adm_party_id=adm_id, raw_name=old.raw_name,
            decision="MATCH", confidence=confidence, reason="VERIFIED_NAME_ANCHOR",
            verified_party_id=root, verified_party_name=winner.root_party_name,
            matched_member_id=winner.owner_party_id, matched_member_name=winner.owner_party_name,
            matched_candidate_name=winner.owner_party_name, matched_candidate_type="official",
            matched_mention=winner.mention_text,
            match_method=f"rules:verified_name_anchor_{kind.lower()}",
            decision_tier=f"RULES_VERIFIED_NAME_ANCHOR_{kind}",
            retrieval_sources=winner.retrieval_sources,
            identity_score=winner.rules_score, char_tfidf_score=winner.char_tfidf_score,
            word_tfidf_score=winner.word_tfidf_score, char_similarity=winner.char_similarity,
            jaro_winkler=winner.jaro_winkler, levenshtein=winner.levenshtein,
            token_jaccard=winner.token_jaccard, raw_coverage=winner.raw_coverage,
            candidate_coverage=winner.candidate_coverage,
            distinctive_token_conflict=winner.distinctive_token_conflict,
            digit_conflict=winner.digit_conflict,
            graph_path=graph.path_to_root(winner.owner_party_id),
            retrieved_root_ids=list(dict.fromkeys([*old.retrieved_root_ids, root])),
        )
        counts["matches"] += 1
        counts[kind.lower()] += 1
    return [by_id[item.adm_party_id] for item in decisions], dict(counts)
