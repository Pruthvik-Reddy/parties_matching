from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from .domain import (FinalDecision, LEGAL_SUFFIXES, MatchProposal,
                     OrganizationMention, PartyRecord, base_name, compact_name,
                     normalize_name, parse_mentions)
from .graph import VerifiedGraph

SKLEARN_ERROR = None
IDENTITY_FEATURES = [
    "exact", "base_exact", "compact_exact", "char_ratio", "jaro_winkler",
    "levenshtein", "token_jaccard", "raw_coverage", "candidate_coverage",
    "weighted_token_overlap", "weighted_candidate_coverage", "digit_conflict",
    "distinctive_token_conflict", "length_ratio", "char_tfidf", "word_tfidf",
    "rrf_score", "embedding_score", "official", "abbreviation", "collision_penalty",
    "acronym_relation", "legal_suffix_only_difference",
]
FEATURE_INDEX = {name: index for index, name in enumerate(IDENTITY_FEATURES)}

class CrossEncoderReranker:
    """No-op adapter: this handoff never loads or runs a cross-encoder."""
    def __init__(self, *args: Any) -> None:
        pass
    def score(self, proposals: list[MatchProposal]) -> None:
        pass

class FeatureScorer:
    """Exact POC rules formula, with no model artifact or learned inference."""
    def __init__(self, idf: dict[str, float], system_threshold: float,
                 plain_threshold: float) -> None:
        self.idf = idf
        self.system_threshold = system_threshold
        self.plain_threshold = plain_threshold
        self.timing: dict[str, float] = defaultdict(float)

    def score_proposals(self, proposals: list[MatchProposal], batch_size: int = 50_000) -> None:
        for start in range(0, len(proposals), max(1, batch_size)):
            batch = proposals[start:start + max(1, batch_size)]
            vectors = np.asarray([
                pair_features(item.mention_text, item.matched_name,
                              item.char_tfidf_score, item.word_tfidf_score,
                              item.rrf_score, item.embedding_score, item.candidate_type,
                              item.candidate_collision_count, self.idf)
                for item in batch
            ], dtype=np.float32)
            for item, vector in zip(batch, vectors):
                value = rule_identity_score(vector)
                if item.candidate_type != "official" and item.candidate_confidence is not None:
                    value = min(value, item.candidate_confidence)
                if item.exact and item.candidate_collision_count == 1:
                    floor = 0.999 if item.candidate_type == "official" else float(item.candidate_confidence or 0.0)
                    value = max(value, floor)
                item.feature_score = item.rules_score = value
                item.char_similarity = float(vector[FEATURE_INDEX["char_ratio"]])
                item.jaro_winkler = float(vector[FEATURE_INDEX["jaro_winkler"]])
                item.levenshtein = float(vector[FEATURE_INDEX["levenshtein"]])
                item.token_jaccard = float(vector[FEATURE_INDEX["token_jaccard"]])
                item.raw_coverage = float(vector[FEATURE_INDEX["raw_coverage"]])
                item.candidate_coverage = float(vector[FEATURE_INDEX["candidate_coverage"]])
                item.digit_conflict = bool(vector[FEATURE_INDEX["digit_conflict"]])
                item.distinctive_token_conflict = bool(vector[FEATURE_INDEX["distinctive_token_conflict"]])

    def score_target_vectors(self, vectors: list[list[float]], batch_size: int = 50_000) -> np.ndarray:
        if not vectors:
            return np.empty(0, dtype=np.float64)
        matrix = np.asarray(vectors, dtype=np.float32)
        margins = np.maximum(0.0, matrix[:, 0] - matrix[:, 3])
        return np.clip(matrix[:, 0] * (0.92 + np.minimum(0.08, margins)), 0.0, 0.999)

class MentionRetriever:
    def __init__(self, records: list[PartyRecord], config: dict[str, Any], workers: int = 1):
        self.config = config
        self.workers = workers
        self.timing: dict[str, float] = defaultdict(float)
        self.records = {record.adm_party_id: record for record in records}
        self.mentions = self._parse(records)
        self.exact: dict[str, list[int]] = defaultdict(list)
        for index, mention in enumerate(self.mentions):
            if mention.normalized:
                self.exact[mention.normalized].append(index)
        self.char_vectorizer: TfidfVectorizer | None = None
        self.char_matrix = None
        self.word_vectorizer: TfidfVectorizer | None = None
        self.word_matrix = None
        texts = [mention.text for mention in self.mentions]
        if texts and SKLEARN_ERROR is None:
            if bool(config.get("char_tfidf_enabled", True)):
                self.char_vectorizer = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                    sublinear_tf=True, dtype=np.float32,
                )
                self.char_matrix = self.char_vectorizer.fit_transform(texts)
            if bool(config.get("word_tfidf_enabled", True)):
                self.word_vectorizer = TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=1,
                    sublinear_tf=True, dtype=np.float32,
                )
                self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.embedding_model = None
        self.embedding_matrix = None
        self.ann_index = None
        self.embedding_warning = None

    def _parse(self, records: list[PartyRecord]) -> list[OrganizationMention]:
        if self.workers <= 1 or len(records) < 2_000:
            nested = map(parse_mentions, records)
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                nested = list(pool.map(parse_mentions, records, chunksize=500))
        return [mention for mentions in nested for mention in mentions]

    def retrieve_top_candidates(self, queries: list[str], top_n: int) -> dict[int, dict[int, dict[str, Any]]]:
        """Return hybrid, bounded verified-query candidates for each ADM mention."""
        by_mention: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        query_norms = [normalize_name(query) for query in queries]

        def empty_hit() -> dict[str, Any]:
            return {
                "sources": set(), "exact": 0.0, "lexical": 0.0,
                "char": 0.0, "word": 0.0, "rrf": 0.0, "embedding": None,
            }

        for query_index, normalized in enumerate(query_norms):
            for mention_index in self.exact.get(normalized, []):
                hit = empty_hit()
                hit.update({
                    "sources": {"exact"}, "exact": 1.0,
                    "char": float(self.char_matrix is not None),
                    "word": float(self.word_matrix is not None),
                    "lexical": float(self.char_matrix is not None or self.word_matrix is not None),
                })
                by_mention[mention_index][query_index] = hit

        def add_sparse(source: str, matrix: Any, vectorizer: Any, minimum: float) -> None:
            if matrix is None or vectorizer is None or not queries:
                return
            stage_started = time.perf_counter()
            query_matrix = vectorizer.transform(queries)
            self.timing[f"{source}_transform_seconds"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            similarities = sp_matmul_topn(
                matrix,
                query_matrix.T.tocsr(),
                top_n=max(1, int(top_n)),
                threshold=minimum,
                sort=True,
                n_threads=max(1, self.workers),
            )
            self.timing[f"{source}_matrix_seconds"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            for mention_index in range(similarities.shape[0]):
                first, last = similarities.indptr[mention_index:mention_index + 2]
                for rank, (query_index, score_value) in enumerate(zip(
                    similarities.indices[first:last], similarities.data[first:last],
                ), start=1):
                    query_index = int(query_index)
                    score = max(0.0, min(1.0, float(score_value)))
                    hit = by_mention[mention_index].setdefault(query_index, empty_hit())
                    hit["sources"].add(source)
                    hit[source] = max(float(hit[source]), score)
                    hit["lexical"] = max(float(hit["lexical"]), score)
                    if bool(self.config.get("rrf_enabled", True)):
                        hit["rrf"] += 1.0 / (float(self.config.get("rrf_k", 60)) + rank)
                    if self.mentions[mention_index].normalized == query_norms[query_index]:
                        hit["sources"].add("exact")
                        hit["exact"] = 1.0
            self.timing[f"{source}_hit_materialization_seconds"] += time.perf_counter() - stage_started

        if queries and (self.char_matrix is not None or self.word_matrix is not None):
            add_sparse(
                "char", self.char_matrix, self.char_vectorizer,
                float(self.config.get("char_min_score", self.config.get("lexical_min_score", 0.28))),
            )
            add_sparse(
                "word", self.word_matrix, self.word_vectorizer,
                float(self.config.get("word_min_score", 0.18)),
            )
        for mention_index, hits in tuple(by_mention.items()):
            exact = {key: value for key, value in hits.items() if value["exact"]}
            ranked = sorted(
                ((key, value) for key, value in hits.items() if key not in exact),
                key=lambda item: (
                    -float(item[1]["rrf"]),
                    -max(float(item[1]["lexical"]), float(item[1]["embedding"] or 0.0)),
                    item[0],
                ),
            )[:max(1, int(top_n))]
            by_mention[mention_index] = {**dict(ranked), **exact}
        return dict(by_mention)


def collect_proposals(
    graph: VerifiedGraph,
    retriever: MentionRetriever,
    scorer: FeatureScorer,
) -> tuple[dict[str, list[MatchProposal]], dict[str, Any]]:
    queries: list[tuple[str, str, str, str, float | None]] = []
    for party_id in sorted(graph.nodes):
        for name, candidate_type, source, candidate_confidence in graph.variants(party_id):
            if normalize_name(name):
                queries.append((party_id, name, candidate_type, source, candidate_confidence))
    started = time.perf_counter()
    root_limit = max(1, int(retriever.config.get("max_roots_per_mention", 50)))
    variants_per_root = max(1, int(retriever.config.get("max_variants_per_root", 2)))
    # mention index -> root -> (party/query variant -> lightweight retrieval tuple)
    shortlists: dict[int, dict[str, dict[tuple[str, str], tuple[Any, ...]]]] = defaultdict(dict)
    indexed = retriever.retrieve_top_candidates(
        [item[1] for item in queries],
        top_n=root_limit * variants_per_root,
    )
    query_seconds = time.perf_counter() - started
    query_rss_mb = _rss_mb()
    raw_hits = sum(len(hits) for hits in indexed.values())
    root_counts_before_cap: list[int] = []
    root_cap_hits = 0
    variant_cap_hits = 0
    for mention_index, hits in indexed.items():
        by_root = shortlists[mention_index]
        for query_index, evidence in hits.items():
            party_id, query_name, candidate_type, source, candidate_confidence = queries[query_index]
            root_id = graph.root_id(party_id)
            exact = float(evidence["exact"])
            lexical = float(evidence["lexical"])
            char_score = float(evidence["char"])
            word_score = float(evidence["word"])
            rrf_score = float(evidence["rrf"])
            embedding = float(evidence["embedding"] or 0.0)
            priority = 2.0 if exact else max(lexical, embedding) + min(0.05, rrf_score)
            variants = by_root.setdefault(root_id, {})
            variant_key = (party_id, normalize_name(query_name))
            pending = (
                priority, party_id, query_name, candidate_type, source,
                candidate_confidence, sorted(evidence["sources"]), exact,
                lexical, char_score, word_score, rrf_score,
                (embedding if evidence["embedding"] is not None else None),
            )
            current = variants.get(variant_key)
            if current is None or (priority, query_name) > (current[0], current[2]):
                variants[variant_key] = pending
        for root_id, variants in tuple(by_root.items()):
            if len(variants) > variants_per_root:
                variant_cap_hits += 1
                retained = sorted(
                    variants.items(), key=lambda item: (-item[1][0], item[0]),
                )[:variants_per_root]
                by_root[root_id] = dict(retained)
        root_counts_before_cap.append(len(by_root))
        if len(by_root) > root_limit:
            root_cap_hits += 1
            ranked = sorted(
                by_root.items(),
                key=lambda item: (-max(value[0] for value in item[1].values()), item[0]),
            )[:root_limit]
            shortlists[mention_index] = dict(ranked)
    shortlist_seconds = time.perf_counter() - started - query_seconds
    shortlist_rss_mb = _rss_mb()
    proposals: list[MatchProposal] = []
    for mention_index, by_root in shortlists.items():
        mention = retriever.mentions[mention_index]
        for root_id, variants in by_root.items():
            root = graph.nodes[root_id]
            for pending in variants.values():
                (
                    _, party_id, query_name, candidate_type, source,
                    candidate_confidence, retrieval_sources, exact,
                    lexical, char_score, word_score, rrf_score, embedding,
                ) = pending
                owner = graph.nodes[party_id]
                proposals.append(MatchProposal(
                    adm_party_id=mention.adm_party_id,
                    mention_id=mention.mention_id,
                    mention_text=mention.text,
                    connector_before=mention.connector_before,
                    connector_after=mention.connector_after,
                    owner_party_id=party_id,
                    owner_party_name=owner.party_name,
                    root_party_id=root_id,
                    root_party_name=root.party_name,
                    matched_name=query_name,
                    candidate_type=candidate_type,
                    candidate_source=source,
                    candidate_confidence=candidate_confidence,
                    candidate_collision_count=graph.candidate_collision_count(query_name),
                    retrieval_sources=retrieval_sources,
                    exact=exact,
                    lexical_score=lexical,
                    char_tfidf_score=char_score,
                    word_tfidf_score=word_score,
                    rrf_score=rrf_score,
                    embedding_score=embedding,
                ))
    proposal_build_seconds = time.perf_counter() - started - query_seconds - shortlist_seconds
    proposals_rss_mb = _rss_mb()
    scorer.score_proposals(
        proposals,
        batch_size=int(retriever.config.get("identity_batch_size", 50_000)),
    )
    by_record: dict[str, list[MatchProposal]] = defaultdict(list)
    for proposal in proposals:
        by_record[proposal.adm_party_id].append(proposal)
    count_values = np.asarray(root_counts_before_cap or [0], dtype=np.float32)
    return dict(by_record), {
        "verified_queries": len(queries),
        "raw_retrieval_hits": raw_hits,
        "scored_proposals": len(proposals),
        "retrieved_proposals": len(proposals),
        "mentions_with_candidates": len(indexed),
        "root_cap_hits": root_cap_hits,
        "variant_cap_hits": variant_cap_hits,
        "candidate_roots_p50": float(np.percentile(count_values, 50)),
        "candidate_roots_p95": float(np.percentile(count_values, 95)),
        "candidate_roots_max": int(count_values.max()),
        "retrieval_seconds": time.perf_counter() - started,
        "retrieval_query_seconds": query_seconds,
        "shortlist_seconds": shortlist_seconds,
        "proposal_build_seconds": proposal_build_seconds,
        "retrieval_rss_mb": query_rss_mb,
        "shortlist_rss_mb": shortlist_rss_mb,
        "proposal_rss_mb": proposals_rss_mb,
        **retriever.timing,
        **scorer.timing,
        "embedding_warning": retriever.embedding_warning,
    }


def _decide_records_legacy(
    records: list[PartyRecord],
    proposals_by_record: dict[str, list[MatchProposal]],
    graph: VerifiedGraph,
    scorer: FeatureScorer,
    reranker: CrossEncoderReranker,
    config: dict[str, Any],
) -> list[FinalDecision]:
    matching_cfg = config.get("matching", {})
    minimum_margin = float(matching_cfg.get("minimum_root_margin", 0.04))
    rerank_top_k = int(matching_cfg.get("rerank_top_k", 5))
    mode = str(matching_cfg.get("cross_encoder_mode", "ambiguous"))
    completed: dict[str, FinalDecision] = {}
    contexts: list[dict[str, Any]] = []
    target_vectors: list[list[float]] = []
    target_destinations: list[tuple[int, MatchProposal]] = []
    for record in records:
        proposals = proposals_by_record.get(record.adm_party_id, [])
        mentions = parse_mentions(record)
        parse_warning = next((m.parse_warning for m in mentions if m.parse_warning), None)
        if not record.raw_name.strip():
            completed[record.adm_party_id] = _no_match(record, "EMPTY_NAME", parse_warning=parse_warning)
            continue
        if not proposals:
            completed[record.adm_party_id] = _no_match(record, "NO_CANDIDATES", parse_warning=parse_warning)
            continue
        root_best = _best_per_root(proposals)
        roots_by_feature = sorted(root_best.values(), key=lambda item: (-item.feature_score, item.root_party_id))
        feature_margin = roots_by_feature[0].feature_score - (roots_by_feature[1].feature_score if len(roots_by_feature) > 1 else 0.0)
        should_rerank = mode == "all_shortlisted" or (
            mode == "ambiguous" and (
                float(matching_cfg.get("ambiguous_low", 0.55)) <= roots_by_feature[0].feature_score <= float(matching_cfg.get("ambiguous_high", 0.98))
                or feature_margin < float(matching_cfg.get("ambiguous_margin", 0.08))
            )
        )
        if should_rerank:
            reranker.score(roots_by_feature[:rerank_top_k])
        root_best = _best_per_root(proposals)
        root_support = Counter(root_id for root_id, _ in {
            (proposal.root_party_id, proposal.mention_id) for proposal in proposals
        })
        identity_order = sorted(root_best.values(), key=lambda item: (-_combined_identity(item), item.root_party_id))
        context_index = len(contexts)
        contexts.append({
            "record": record,
            "parse_warning": parse_warning,
            "plain": len(mentions) == 1 and parse_warning != "MALFORMED_CONNECTOR",
            "retrieved_roots": [proposal.root_party_id for proposal in identity_order],
            "scored_roots": [],
        })
        best_identity = _combined_identity(identity_order[0])
        second_identity = _combined_identity(identity_order[1]) if len(identity_order) > 1 else 0.0
        for position, proposal in enumerate(identity_order):
            other_identity = second_identity if position == 0 else best_identity
            depth = len(graph.path_to_root(proposal.owner_party_id))
            target_vectors.append(target_features(
                proposal, other_identity, root_support[proposal.root_party_id], depth,
            ))
            target_destinations.append((context_index, proposal))

    target_scores = scorer.score_target_vectors(
        target_vectors,
        batch_size=int(matching_cfg.get("confidence_batch_size", 50_000)),
    )
    for score, (context_index, proposal) in zip(target_scores, target_destinations):
        contexts[context_index]["scored_roots"].append((float(score), proposal))

    for context in contexts:
        record = context["record"]
        parse_warning = context["parse_warning"]
        retrieved_roots = context["retrieved_roots"]
        scored_roots = context["scored_roots"]
        scored_roots.sort(key=lambda item: (-item[0], item[1].root_party_id))
        eligible_roots = [
            item for item in scored_roots
            if not _guard_reason(item[1], matching_cfg)
        ]
        if eligible_roots:
            confidence, winner = eligible_roots[0]
            runner_score, runner = eligible_roots[1] if len(eligible_roots) > 1 else (0.0, None)
        else:
            confidence, winner = scored_roots[0]
            runner_score, runner = scored_roots[1] if len(scored_roots) > 1 else (0.0, None)
        margin = confidence - runner_score
        decision_tier = _decision_tier(winner, matching_cfg)
        if not eligible_roots:
            completed[record.adm_party_id] = _no_match(
                record, _guard_reason(winner, matching_cfg) or "INSUFFICIENT_SUPPORT",
                confidence=confidence, proposal=winner,
                runner=runner, runner_score=runner_score, margin=margin,
                retrieved_roots=retrieved_roots, parse_warning=parse_warning,
                decision_tier=decision_tier,
            )
            continue
        if runner and margin < minimum_margin:
            reason = "CANDIDATE_COLLISION" if winner.candidate_collision_count > 1 else "AMBIGUOUS_FINAL_TARGETS"
            completed[record.adm_party_id] = _no_match(
                record, reason, confidence=confidence, proposal=winner,
                runner=runner, runner_score=runner_score, margin=margin,
                retrieved_roots=retrieved_roots, parse_warning=parse_warning,
                decision_tier=decision_tier,
            )
            continue
        threshold = scorer.plain_threshold if context["plain"] and scorer.plain_threshold is not None else scorer.system_threshold
        containment = (
            context["plain"] and scorer.plain_threshold is not None
            and bool(config.get("decision", {}).get("rules_containment_enabled", False))
            and confidence < threshold
            and _rules_containment_evidence(winner, scorer.idf, len(graph.nodes), margin, config)
        )
        if containment:
            threshold = min(threshold, float(config["decision"].get("rules_containment_min_confidence", 0.40)))
        if confidence < threshold:
            completed[record.adm_party_id] = _no_match(
                record, "INSUFFICIENT_SUPPORT", confidence=confidence, proposal=winner,
                runner=runner, runner_score=runner_score, margin=margin,
                retrieved_roots=retrieved_roots, parse_warning=parse_warning,
                decision_tier=decision_tier,
            )
            continue
        path = graph.path_to_root(winner.owner_party_id)
        if containment:
            decision_tier = "RULES_UNIQUE_CONTAINMENT"
        completed[record.adm_party_id] = FinalDecision(
            account_id=record.account_id,
            adm_party_id=record.adm_party_id,
            raw_name=record.raw_name,
            decision="MATCH",
            confidence=confidence,
            reason="MATCH",
            verified_party_id=winner.root_party_id,
            verified_party_name=winner.root_party_name,
            matched_member_id=winner.owner_party_id,
            matched_member_name=winner.owner_party_name,
            matched_candidate_name=winner.matched_name,
            matched_candidate_type=winner.candidate_type,
            candidate_expansion_confidence=winner.candidate_confidence,
            matched_mention=winner.mention_text,
            connector=winner.connector_before or winner.connector_after,
            match_method="rules:unique_containment" if containment else _match_method(winner),
            decision_tier=decision_tier,
            retrieval_sources=winner.retrieval_sources,
            identity_score=winner.feature_score,
            char_tfidf_score=winner.char_tfidf_score,
            word_tfidf_score=winner.word_tfidf_score,
            rrf_score=winner.rrf_score,
            char_similarity=winner.char_similarity,
            jaro_winkler=winner.jaro_winkler,
            levenshtein=winner.levenshtein,
            token_jaccard=winner.token_jaccard,
            raw_coverage=winner.raw_coverage,
            candidate_coverage=winner.candidate_coverage,
            distinctive_token_conflict=winner.distinctive_token_conflict,
            digit_conflict=winner.digit_conflict,
            cross_encoder_score=winner.cross_encoder_score,
            runner_up_party_id=runner.root_party_id if runner else None,
            runner_up_score=runner_score if runner else None,
            margin=margin,
            graph_path=path,
            retrieved_root_ids=retrieved_roots,
            parse_warning=parse_warning,
        )
    return [completed[record.adm_party_id] for record in records]


def _decide_connectors(
    records: list[PartyRecord],
    proposals_by_record: dict[str, list[MatchProposal]],
    graph: VerifiedGraph,
    scorer: FeatureScorer,
    reranker: CrossEncoderReranker,
    config: dict[str, Any],
) -> list[FinalDecision]:
    matching_cfg = config.get("matching", {})
    minimum_margin = float(matching_cfg.get("minimum_root_margin", 0.04))
    decision_cfg = config.get("decision", {})
    # This is an enhanced-rules recovery only. The ML decisions and the
    # ordinary connector policy continue to use the existing cutoff.
    short_name_enabled = (
        scorer.plain_threshold is not None
        and bool(decision_cfg.get("rules_enhanced_enabled", False))
        and bool(decision_cfg.get("rules_connector_short_name_enabled", False))
    )
    if short_name_enabled:
        from .rules_enhancement import _root_token_index, _unique_official_short_token
        root_tokens = _root_token_index(graph)
        short_floor = float(decision_cfg.get("rules_containment_min_confidence", 0.40))
        short_margin = float(decision_cfg.get("rules_containment_min_margin", 0.08))
        short_lexical = float(decision_cfg.get("rules_containment_min_lexical", 0.50))
    contexts: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    destinations: list[tuple[int, int, MatchProposal]] = []
    for record in records:
        mentions = parse_mentions(record)
        connectors = {mention.connector_before for mention in mentions if mention.connector_before}
        context = {"record": record, "mentions": [], "connector": next(iter(connectors)) if len(connectors) == 1 else None}
        if any(mention.parse_warning == "MALFORMED_CONNECTOR" for mention in mentions):
            context["forced_reason"] = "MALFORMED_CONNECTOR"
        elif len(connectors) != 1:
            context["forced_reason"] = "MIXED_CONNECTORS_UNSUPPORTED"
        contexts.append(context)
        if context.get("forced_reason"):
            continue
        grouped: dict[str, list[MatchProposal]] = defaultdict(list)
        for proposal in proposals_by_record.get(record.adm_party_id, []):
            grouped[proposal.mention_id].append(proposal)
        for mention in mentions:
            proposals = grouped.get(mention.mention_id, [])
            by_root = _best_per_root(proposals)
            by_feature = sorted(by_root.values(), key=lambda item: (-item.feature_score, item.root_party_id))
            if by_feature:
                margin = by_feature[0].feature_score - (by_feature[1].feature_score if len(by_feature) > 1 else 0.0)
                mode = str(matching_cfg.get("cross_encoder_mode", "ambiguous"))
                if mode == "all_shortlisted" or (
                    mode == "ambiguous" and (
                        float(matching_cfg.get("ambiguous_low", 0.55)) <= by_feature[0].feature_score <= float(matching_cfg.get("ambiguous_high", 0.98))
                        or margin < float(matching_cfg.get("ambiguous_margin", 0.08))
                    )
                ):
                    reranker.score(by_feature[:int(matching_cfg.get("rerank_top_k", 5))])
            ranked = sorted(_best_per_root(proposals).values(), key=lambda item: (-_combined_identity(item), item.root_party_id))
            mention_context = {"mention": mention, "ranked": ranked, "scored": []}
            context["mentions"].append(mention_context)
            first = _combined_identity(ranked[0]) if ranked else 0.0
            second = _combined_identity(ranked[1]) if len(ranked) > 1 else 0.0
            for position, proposal in enumerate(ranked):
                vectors.append(target_features(
                    proposal, second if position == 0 else first,
                    1, len(graph.path_to_root(proposal.owner_party_id)),
                ))
                destinations.append((len(contexts) - 1, len(context["mentions"]) - 1, proposal))

    scores = scorer.score_target_vectors(vectors, batch_size=int(matching_cfg.get("confidence_batch_size", 50_000)))
    for score, (record_index, mention_index, proposal) in zip(scores, destinations):
        contexts[record_index]["mentions"][mention_index]["scored"].append((float(score), proposal))

    decisions: list[FinalDecision] = []
    for context in contexts:
        record = context["record"]
        if context.get("forced_reason"):
            decision = _no_match(record, context["forced_reason"], parse_warning=context["forced_reason"])
            decision.connector_resolution = context["forced_reason"]
            decisions.append(decision)
            continue
        retrieved = {}
        valid: list[dict[str, Any]] = []
        mention_results = []
        best_rejected: tuple[float, MatchProposal] | None = None
        for item in context["mentions"]:
            mention = item["mention"]
            for proposal in item["ranked"]:
                retrieved[proposal.root_party_id] = max(retrieved.get(proposal.root_party_id, 0.0), _combined_identity(proposal))
            scored = sorted(item["scored"], key=lambda value: (-value[0], value[1].root_party_id))
            eligible = [value for value in scored if not _guard_reason(value[1], matching_cfg)]
            top = eligible[0] if eligible else (scored[0] if scored else None)
            runner = next((value for value in eligible if top and value[1].root_party_id != top[1].root_party_id), None)
            item["top"] = top
            confidence, proposal = top if top else (0.0, None)
            margin = confidence - (runner[0] if runner else 0.0)
            short_name_match = bool(
                short_name_enabled and proposal is not None and eligible
                and confidence < scorer.system_threshold
                and confidence >= short_floor and margin >= short_margin
                and _unique_official_short_token(mention.text, proposal, root_tokens, short_lexical)
            )
            if proposal is None:
                reason = "NO_CANDIDATES"
            elif not eligible:
                reason = _guard_reason(proposal, matching_cfg) or "INSUFFICIENT_SUPPORT"
            elif runner and margin < minimum_margin:
                reason = "AMBIGUOUS_FINAL_TARGETS"
            elif confidence < scorer.system_threshold and not short_name_match:
                reason = "INSUFFICIENT_SUPPORT"
            else:
                reason = "MATCH"
            result = {
                "position": mention.position, "text": mention.text,
                "decision": "MATCH" if reason == "MATCH" else "NO_MATCH", "reason": reason,
                "confidence": confidence, "root_id": proposal.root_party_id if proposal else None,
                "root_name": proposal.root_party_name if proposal else None,
                "matched_candidate": proposal.matched_name if proposal else None,
                "exact": bool(proposal.exact) if proposal else False,
                "runner_up_root_id": runner[1].root_party_id if runner else None,
                "runner_up_score": runner[0] if runner else None,
                "margin": margin if proposal else None,
                "decision_tier": "RULES_ROOT_UNIQUE_CONNECTOR_SHORT_NAME" if short_name_match and reason == "MATCH" else None,
            }
            mention_results.append(result)
            if reason == "MATCH":
                valid.append({"position": mention.position, "confidence": confidence, "proposal": proposal,
                              "runner": runner[1] if runner else None, "runner_score": runner[0] if runner else None,
                              "margin": margin, "decision_tier": result["decision_tier"]})
            elif proposal and (best_rejected is None or confidence > best_rejected[0]):
                best_rejected = (confidence, proposal)
        retrieved_roots = [root for root, _ in sorted(retrieved.items(), key=lambda item: (-item[1], item[0]))]
        connector = context["connector"]
        # Both connector policies currently prefer the leftmost mention when
        # distinct verified roots are valid. Keep the near-cutoff veto tied to
        # that same preferred mention rather than falling through to the right.
        preferred_index = 0
        preferred = mention_results[preferred_index]
        band = max(0.0, float(matching_cfg.get("preferred_near_cutoff_band", 0.03)))
        if (
            valid and not any(item["position"] == preferred["position"] for item in valid)
            and preferred["root_id"] and preferred["reason"] in {"INSUFFICIENT_SUPPORT", "AMBIGUOUS_FINAL_TARGETS"}
            and preferred["confidence"] >= scorer.system_threshold - band
        ):
            proposal = context["mentions"][preferred_index]["top"][1]
            decision = _no_match(
                record, "PREFERRED_SEGMENT_NEAR_CUTOFF", preferred["confidence"],
                proposal=proposal, retrieved_roots=retrieved_roots,
            )
            decision.connector_resolution = "PREFERRED_NEAR_CUTOFF"
            decision.mention_results = mention_results
            decisions.append(decision)
            continue
        if not valid:
            reasons = {item["reason"] for item in mention_results}
            reason = "AMBIGUOUS_FINAL_TARGETS" if "AMBIGUOUS_FINAL_TARGETS" in reasons else (
                "INSUFFICIENT_SUPPORT" if "INSUFFICIENT_SUPPORT" in reasons else "NO_MENTION_MATCH"
            )
            score, proposal = best_rejected if best_rejected else (0.0, None)
            decision = _no_match(record, reason, score, proposal=proposal, retrieved_roots=retrieved_roots)
            decision.connector_resolution = "NONE"
        else:
            roots = {item["proposal"].root_party_id for item in valid}
            if len(roots) == 1:
                chosen = max(valid, key=lambda item: (item["confidence"], -item["position"]))
                resolution = "ONLY_MATCH" if len(valid) == 1 else "SAME_ROOT"
            else:
                chosen = min(valid, key=lambda item: item["position"])
                resolution = f"{connector}_LEFT"
            confidence = chosen["confidence"]
            proposal = chosen["proposal"]
            runner = chosen["runner"]
            decision = _no_match(
                record, "MATCH", confidence, proposal=proposal, runner=runner,
                runner_score=chosen["runner_score"], margin=chosen["margin"],
                retrieved_roots=retrieved_roots,
                decision_tier=chosen["decision_tier"] or _decision_tier(proposal, matching_cfg),
            )
            if chosen["decision_tier"]:
                decision.match_method = "rules:root_unique_connector_short_name"
            decision.provisional_party_id = proposal.root_party_id
            decision.provisional_party_name = proposal.root_party_name
            decision.selected_mention_position = chosen["position"]
            decision.connector_resolution = resolution
            decision.decision = "MATCH"
            decision.graph_path = graph.path_to_root(proposal.owner_party_id)
        decision.mention_results = mention_results
        decisions.append(decision)
    return decisions


def decide_records(
    records: list[PartyRecord], proposals_by_record: dict[str, list[MatchProposal]],
    graph: VerifiedGraph, scorer: FeatureScorer, reranker: CrossEncoderReranker,
    config: dict[str, Any],
) -> list[FinalDecision]:
    if str(config.get("matching", {}).get("connector_policy", "positional")) == "legacy":
        return _decide_records_legacy(records, proposals_by_record, graph, scorer, reranker, config)
    plain, connector = [], []
    for record in records:
        mentions = parse_mentions(record)
        (connector if len(mentions) > 1 or mentions[0].parse_warning == "MALFORMED_CONNECTOR" else plain).append(record)
    legacy = _decide_records_legacy(plain, proposals_by_record, graph, scorer, reranker, config)
    resolved = _decide_connectors(connector, proposals_by_record, graph, scorer, reranker, config)
    by_id = {decision.adm_party_id: decision for decision in [*legacy, *resolved]}
    return [by_id[record.adm_party_id] for record in records]


def pair_features(
    left: str,
    right: str,
    char_tfidf_score: float,
    word_tfidf_score: float,
    rrf_score: float,
    embedding_score: float | None,
    candidate_type: str,
    collision_count: int,
    idf: dict[str, float] | None = None,
) -> list[float]:
    left_norm, right_norm = normalize_name(left), normalize_name(right)
    left_tokens, right_tokens = set(left_norm.split()), set(right_norm.split())
    intersection, union = left_tokens & right_tokens, left_tokens | right_tokens
    weights = idf or {}
    shared_weight = sum(weights.get(token, 1.0) for token in intersection)
    total_weight = sum(weights.get(token, 1.0) for token in union) or 1.0
    candidate_weight = sum(weights.get(token, 1.0) for token in right_tokens) or 1.0
    digits_left = {token for token in left_tokens if any(char.isdigit() for char in token)}
    digits_right = {token for token in right_tokens if any(char.isdigit() for char in token)}
    return [
        float(bool(left_norm) and left_norm == right_norm),
        float(bool(base_name(left)) and base_name(left) == base_name(right)),
        float(bool(compact_name(left)) and compact_name(left) == compact_name(right)),
        char_similarity(left_norm, right_norm),
        jaro_winkler_similarity(left_norm, right_norm),
        levenshtein_similarity(left_norm, right_norm),
        len(intersection) / max(1, len(union)),
        len(intersection) / max(1, len(left_tokens)),
        len(intersection) / max(1, len(right_tokens)),
        shared_weight / total_weight,
        shared_weight / candidate_weight,
        float(bool(digits_left and digits_right and digits_left != digits_right)),
        float(distinctive_token_conflict(left_tokens, right_tokens, weights)),
        min(len(left_norm), len(right_norm)) / max(1, max(len(left_norm), len(right_norm))),
        float(char_tfidf_score),
        float(word_tfidf_score),
        min(1.0, float(rrf_score) * 30.0),
        float(embedding_score or 0.0),
        float(candidate_type == "official"),
        float(candidate_type == "abbreviation"),
        1.0 / max(1, collision_count),
        float(acronym_relation(left_norm, right_norm)),
        float(bool(left_norm != right_norm and base_name(left) and base_name(left) == base_name(right))),
    ]


def target_features(
    proposal: MatchProposal,
    runner_up_identity: float,
    supporting_mentions: int,
    graph_depth: int,
) -> list[float]:
    identity = _combined_identity(proposal)
    return [
        identity,
        float(proposal.cross_encoder_score if proposal.cross_encoder_score is not None else identity),
        float(proposal.cross_encoder_score is not None),
        float(runner_up_identity),
        float(identity - runner_up_identity),
        float(proposal.exact),
        1.0 / max(1, proposal.candidate_collision_count),
        float(supporting_mentions),
        float(graph_depth),
    ]


def rule_identity_score(vector: Iterable[float]) -> float:
    values = list(vector)
    if values[FEATURE_INDEX["exact"]]:
        return 0.999
    if values[FEATURE_INDEX["base_exact"]]:
        return 0.985
    if values[FEATURE_INDEX["compact_exact"]]:
        return 0.975
    score = (
        0.14 * values[FEATURE_INDEX["char_ratio"]]
        + 0.08 * values[FEATURE_INDEX["jaro_winkler"]]
        + 0.07 * values[FEATURE_INDEX["levenshtein"]]
        + 0.08 * values[FEATURE_INDEX["token_jaccard"]]
        + 0.12 * values[FEATURE_INDEX["candidate_coverage"]]
        + 0.08 * values[FEATURE_INDEX["weighted_token_overlap"]]
        + 0.12 * values[FEATURE_INDEX["char_tfidf"]]
        + 0.13 * values[FEATURE_INDEX["word_tfidf"]]
        + 0.06 * values[FEATURE_INDEX["rrf_score"]]
        + 0.07 * values[FEATURE_INDEX["embedding_score"]]
        + 0.03 * values[FEATURE_INDEX["official"]]
        + 0.02 * values[FEATURE_INDEX["acronym_relation"]]
        - 0.45 * values[FEATURE_INDEX["digit_conflict"]]
        - 0.30 * values[FEATURE_INDEX["distinctive_token_conflict"]]
    )
    return max(0.001, min(0.97, score))


def char_similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.fuzz import ratio
        return float(ratio(left, right)) / 100.0
    except ImportError:
        return SequenceMatcher(None, left, right).ratio()


def jaro_winkler_similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.distance import JaroWinkler
        return float(JaroWinkler.normalized_similarity(left, right))
    except ImportError:
        return char_similarity(left, right)


def levenshtein_similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.distance import Levenshtein
        return float(Levenshtein.normalized_similarity(left, right))
    except ImportError:
        return char_similarity(left, right)


def acronym_relation(left: str, right: str) -> bool:
    def meaningful(value: str) -> list[str]:
        return [token for token in value.split() if token not in LEGAL_SUFFIXES]
    left_tokens, right_tokens = meaningful(left), meaningful(right)
    left_compact, right_compact = "".join(left_tokens), "".join(right_tokens)
    left_initials = "".join(token[0] for token in left_tokens) if len(left_tokens) > 1 else ""
    right_initials = "".join(token[0] for token in right_tokens) if len(right_tokens) > 1 else ""
    return bool(
        (len(left_tokens) == 1 and left_compact == right_initials)
        or (len(right_tokens) == 1 and right_compact == left_initials)
    )


def distinctive_token_conflict(
    left_tokens: set[str], right_tokens: set[str], weights: dict[str, float],
) -> bool:
    exempt = set(LEGAL_SUFFIXES) | {"a", "an", "the", "of", "and"}
    left_only = left_tokens - right_tokens - exempt
    right_only = right_tokens - left_tokens - exempt
    shared = (left_tokens & right_tokens) - exempt
    if not shared or not left_only or not right_only:
        return False
    left_distinctive = {token for token in left_only if len(token) <= 4 or weights.get(token, 1.0) >= 1.5}
    right_distinctive = {token for token in right_only if len(token) <= 4 or weights.get(token, 1.0) >= 1.5}
    return bool(left_distinctive and right_distinctive)


def _combined_identity(proposal: MatchProposal) -> float:
    return (proposal.feature_score if proposal.cross_encoder_score is None else
            0.55 * proposal.feature_score + 0.45 * proposal.cross_encoder_score)


def _best_per_root(proposals: list[MatchProposal]) -> dict[str, MatchProposal]:
    best = {}
    for proposal in proposals:
        current = best.get(proposal.root_party_id)
        if current is None or (_combined_identity(proposal), proposal.matched_name) > (_combined_identity(current), current.matched_name):
            best[proposal.root_party_id] = proposal
    return best


def _match_method(proposal: MatchProposal) -> str:
    if proposal.exact:
        return "exact:normalized"
    if proposal.cross_encoder_score is not None:
        return "selective_hybrid:cross_encoder"
    sources = set(proposal.retrieval_sources)
    lexical = sorted(sources & {"word", "char"})
    if "embedding" in sources and lexical:
        return "feature:" + "_".join([*lexical, "embedding"])
    if "embedding" in sources:
        return "feature:embedding"
    return "feature:" + ("_".join(lexical) if lexical else "retrieved")


def _rules_containment_evidence(
    proposal: MatchProposal, idf: dict[str, float], verified_count: int,
    margin: float, config: dict[str, Any],
) -> bool:
    """Conservative rules-only path for a distinctive short name inside an official name."""
    options = config.get("decision", {})
    if (proposal.candidate_type != "official" or proposal.candidate_collision_count != 1
            or proposal.digit_conflict or proposal.distinctive_token_conflict or not idf):
        return False
    if margin < float(options.get("rules_containment_min_margin", 0.08)):
        return False
    if max(proposal.char_tfidf_score, proposal.word_tfidf_score) < float(
        options.get("rules_containment_min_lexical", 0.50)
    ):
        return False
    ignore = LEGAL_SUFFIXES | {"a", "an", "the", "of", "and"}
    short = set(normalize_name(proposal.mention_text).split()) - ignore
    long = set(normalize_name(proposal.matched_name).split()) - ignore
    if not short or not short < long or len(long - short) > 2:
        return False
    max_df = int(options.get("rules_containment_max_token_df", 3))
    return any(
        len(token) >= 5 and token in idf
        and (verified_count + 1) / math.exp(idf[token] - 1) - 1 <= max_df + 1e-6
        for token in short
    )


def _decision_tier(proposal: MatchProposal, config: dict[str, Any]) -> str:
    if proposal.exact and proposal.candidate_type == "official":
        return "EXACT_OFFICIAL"
    if proposal.exact:
        return "EXACT_CANDIDATE"
    if (
        proposal.candidate_coverage >= float(config.get("directional_coverage_threshold", 0.90))
        and proposal.token_jaccard >= float(config.get("directional_jaccard_threshold", 0.45))
    ):
        return "DIRECTIONAL_CONTAINMENT"
    if proposal.cross_encoder_score is not None:
        return "CROSS_ENCODER"
    return "MULTI_SIGNAL"


def _guard_reason(proposal: MatchProposal, config: dict[str, Any]) -> str | None:
    if proposal.exact:
        return None
    if bool(config.get("digit_conflict_guard", True)) and proposal.digit_conflict:
        return "DIGIT_CONFLICT"
    if bool(config.get("distinctive_token_guard", True)) and proposal.distinctive_token_conflict:
        return "DISTINCTIVE_TOKEN_CONFLICT"
    return None


def _no_match(
    record: PartyRecord,
    reason: str,
    confidence: float = 0.0,
    proposal: MatchProposal | None = None,
    runner: MatchProposal | None = None,
    runner_score: float | None = None,
    margin: float | None = None,
    retrieved_roots: list[str] | None = None,
    parse_warning: str | None = None,
    decision_tier: str | None = None,
) -> FinalDecision:
    return FinalDecision(
        account_id=record.account_id, adm_party_id=record.adm_party_id,
        raw_name=record.raw_name, decision="NO_MATCH", confidence=confidence, reason=reason,
        verified_party_id=proposal.root_party_id if proposal else None,
        verified_party_name=proposal.root_party_name if proposal else None,
        matched_member_id=proposal.owner_party_id if proposal else None,
        matched_member_name=proposal.owner_party_name if proposal else None,
        matched_candidate_name=proposal.matched_name if proposal else None,
        matched_candidate_type=proposal.candidate_type if proposal else None,
        candidate_expansion_confidence=proposal.candidate_confidence if proposal else None,
        matched_mention=proposal.mention_text if proposal else None,
        connector=(proposal.connector_before or proposal.connector_after) if proposal else None,
        match_method=_match_method(proposal) if proposal else None,
        decision_tier=decision_tier,
        retrieval_sources=proposal.retrieval_sources if proposal else [],
        identity_score=proposal.feature_score if proposal else None,
        char_tfidf_score=proposal.char_tfidf_score if proposal else None,
        word_tfidf_score=proposal.word_tfidf_score if proposal else None,
        rrf_score=proposal.rrf_score if proposal else None,
        char_similarity=proposal.char_similarity if proposal else None,
        jaro_winkler=proposal.jaro_winkler if proposal else None,
        levenshtein=proposal.levenshtein if proposal else None,
        token_jaccard=proposal.token_jaccard if proposal else None,
        raw_coverage=proposal.raw_coverage if proposal else None,
        candidate_coverage=proposal.candidate_coverage if proposal else None,
        distinctive_token_conflict=proposal.distinctive_token_conflict if proposal else None,
        digit_conflict=proposal.digit_conflict if proposal else None,
        cross_encoder_score=proposal.cross_encoder_score if proposal else None,
        runner_up_party_id=runner.root_party_id if runner else None,
        runner_up_score=runner_score,
        margin=margin,
        retrieved_root_ids=retrieved_roots or [],
        parse_warning=parse_warning,
    )


def _token_idf(names: list[str]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for name in names:
        counts.update(set(normalize_name(name).split()))
    total = max(1, len(names))
    return {token: math.log((1 + total) / (1 + count)) + 1.0 for token, count in counts.items()}


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None
