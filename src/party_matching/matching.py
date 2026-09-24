from __future__ import annotations

import copy
import math
import os
import platform
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sparse_dot_topn import sp_matmul_topn
if os.getenv("PARTY_MATCHING_DISABLE_SKLEARN", "").casefold() in {"1", "true", "yes"}:
    HistGradientBoostingClassifier = TfidfVectorizer = IsotonicRegression = LogisticRegression = NearestNeighbors = None
    SKLEARN_ERROR = "disabled by PARTY_MATCHING_DISABLE_SKLEARN"
else:
  try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_ERROR = None
  except Exception as exc:  # permits exact/brute smoke runs on restricted machines
    HistGradientBoostingClassifier = TfidfVectorizer = IsotonicRegression = LogisticRegression = NearestNeighbors = None
    SKLEARN_ERROR = str(exc)

from .domain import (
    FinalDecision,
    LEGAL_SUFFIXES,
    MatchProposal,
    OrganizationMention,
    PartyRecord,
    base_name,
    compact_name,
    normalize_name,
    parse_mentions,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .expansion import ExpansionService
from .graph import VerifiedGraph


IDENTITY_FEATURES = [
    "exact", "base_exact", "compact_exact", "char_ratio", "jaro_winkler",
    "levenshtein", "token_jaccard", "raw_coverage", "candidate_coverage",
    "weighted_token_overlap", "weighted_candidate_coverage", "digit_conflict",
    "distinctive_token_conflict", "length_ratio", "char_tfidf", "word_tfidf",
    "rrf_score", "embedding_score", "official", "abbreviation", "collision_penalty",
    "acronym_relation", "legal_suffix_only_difference",
]
FEATURE_INDEX = {name: index for index, name in enumerate(IDENTITY_FEATURES)}
TARGET_FEATURES = [
    "identity", "cross_encoder", "has_cross_encoder", "runner_up_identity",
    "identity_margin", "exact", "collision_penalty", "supporting_mentions", "graph_depth",
]


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
        self.embedding_matrix: np.ndarray | None = None
        self.ann_index = None
        self.embedding_warning: str | None = None
        if bool(config.get("embedding_enabled", True)) and texts:
            self._build_embeddings(texts)

    def _parse(self, records: list[PartyRecord]) -> list[OrganizationMention]:
        if self.workers <= 1 or len(records) < 2_000:
            nested = map(parse_mentions, records)
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                nested = list(pool.map(parse_mentions, records, chunksize=500))
        return [mention for mentions in nested for mention in mentions]

    def _build_embeddings(self, texts: list[str]) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = str(self.config.get("embedding_model_path") or self.config.get("embedding_model"))
            self.embedding_model = SentenceTransformer(model_name, device="cpu")
            matrix = self.embedding_model.encode(
                texts,
                batch_size=256,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10_000,
            ).astype(np.float32)
            self.embedding_matrix = matrix
            if str(self.config.get("ann_backend", "usearch")) == "usearch":
                try:
                    from usearch.index import Index
                    index = Index(ndim=matrix.shape[1], metric="cos", dtype="f32")
                    index.add(np.arange(len(matrix), dtype=np.uint64), matrix)
                    self.ann_index = index
                except Exception as exc:  # optional native dependency
                    self.embedding_warning = f"USearch unavailable; using brute cosine: {exc}"
        except Exception as exc:  # optional model path
            self.embedding_warning = f"Embeddings disabled for this run: {exc}"
            self.embedding_model = None
            self.embedding_matrix = None

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
        if self.embedding_model is not None and self.embedding_matrix is not None and queries:
            count = min(max(1, int(self.config.get("embedding_top_k", 100))), len(self.mentions))
            minimum = float(self.config.get("embedding_min_score", 0.42))
            for start in range(0, len(queries), 128):
                vectors = self.embedding_model.encode(
                    queries[start:start + 128], batch_size=128, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                ).astype(np.float32)
                for local_index, vector in enumerate(vectors):
                    query_index = start + local_index
                    if self.ann_index is not None:
                        matches = self.ann_index.search(vector, count=count)
                        keys = np.atleast_1d(matches.keys)
                        scores = 1.0 - np.atleast_1d(matches.distances)
                    else:
                        all_scores = self.embedding_matrix @ vector
                        keys = np.argpartition(all_scores, -count)[-count:]
                        scores = all_scores[keys]
                    for mention_index, score_value in zip(keys, scores):
                        score = max(-1.0, min(1.0, float(score_value)))
                        if score < minimum:
                            continue
                        hit = by_mention[int(mention_index)].setdefault(query_index, empty_hit())
                        hit["sources"].add("embedding")
                        hit["embedding"] = max(hit["embedding"] or -1.0, score)

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


class FeatureScorer:
    def __init__(self, artifact_path: str | Path, expected_data_version: str | None = None):
        self.path = Path(artifact_path)
        self.load_warning = None
        try:
            self.artifact = joblib.load(self.path) if self.path.exists() else {}
        except Exception as exc:
            self.artifact = {}
            self.load_warning = f"Matcher artifact could not be loaded; using fallback rules: {exc}"
        artifact_version = self.artifact.get("data_version")
        if self.artifact and expected_data_version and artifact_version != expected_data_version:
            raise RuntimeError("Matcher artifact is stale for the prepared workbook. Run scripts/train.py again.")
        self.model = self.artifact.get("identity_model")
        self.target_model = self.artifact.get("target_model")
        self.calibrator = self.artifact.get("calibrator")
        self.timing: dict[str, float] = defaultdict(float)
        self.idf = self.artifact.get("idf", {})
        self.system_threshold = float(self.artifact.get("system_threshold", 0.90))
        self.plain_threshold: float | None = None
        artifact_features = self.artifact.get("identity_features")
        if artifact_features and list(artifact_features) != IDENTITY_FEATURES:
            raise RuntimeError("Matcher feature schema changed. Run scripts/train.py again.")

    def score_proposals(self, proposals: list[MatchProposal], batch_size: int = 50_000) -> None:
        if not proposals:
            return
        size = max(1, int(batch_size))
        for start in range(0, len(proposals), size):
            batch = proposals[start:start + size]
            stage_started = time.perf_counter()
            vectors = np.asarray([
                pair_features(
                    proposal.mention_text, proposal.matched_name,
                    proposal.char_tfidf_score, proposal.word_tfidf_score,
                    proposal.rrf_score, proposal.embedding_score,
                    proposal.candidate_type, proposal.candidate_collision_count, self.idf,
                ) for proposal in batch
            ], dtype=np.float32)
            self.timing["identity_feature_seconds"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            rule_scores = np.asarray([rule_identity_score(vector) for vector in vectors])
            if self.model is not None:
                scores = self.model.predict_proba(vectors)[:, 1]
            else:
                scores = rule_scores
            self.timing["identity_model_seconds"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            for proposal, score, rule_score in zip(batch, scores, rule_scores):
                value = float(score)
                rule_value = float(rule_score)
                if proposal.candidate_type != "official" and proposal.candidate_confidence is not None:
                    value = min(value, proposal.candidate_confidence)
                    rule_value = min(rule_value, proposal.candidate_confidence)
                if proposal.exact and proposal.candidate_collision_count == 1:
                    exact_floor = 0.999 if proposal.candidate_type == "official" else float(proposal.candidate_confidence or 0.0)
                    value = max(value, exact_floor)
                    rule_value = max(rule_value, exact_floor)
                proposal.feature_score = value
                proposal.rules_score = rule_value
            for proposal, vector in zip(batch, vectors):
                proposal.char_similarity = float(vector[FEATURE_INDEX["char_ratio"]])
                proposal.jaro_winkler = float(vector[FEATURE_INDEX["jaro_winkler"]])
                proposal.levenshtein = float(vector[FEATURE_INDEX["levenshtein"]])
                proposal.token_jaccard = float(vector[FEATURE_INDEX["token_jaccard"]])
                proposal.raw_coverage = float(vector[FEATURE_INDEX["raw_coverage"]])
                proposal.candidate_coverage = float(vector[FEATURE_INDEX["candidate_coverage"]])
                proposal.digit_conflict = bool(vector[FEATURE_INDEX["digit_conflict"]])
                proposal.distinctive_token_conflict = bool(vector[FEATURE_INDEX["distinctive_token_conflict"]])
            self.timing["identity_assignment_seconds"] += time.perf_counter() - stage_started

    def target_score(
        self,
        proposal: MatchProposal,
        runner_up_identity: float,
        supporting_mentions: int,
        graph_depth: int,
    ) -> float:
        identity = _combined_identity(proposal)
        vector = target_features(proposal, runner_up_identity, supporting_mentions, graph_depth)
        if self.target_model is not None:
            value = float(self.target_model.predict_proba(np.asarray([vector], dtype=np.float32))[0, 1])
        else:
            margin = max(0.0, identity - runner_up_identity)
            value = float(max(0.0, min(0.999, identity * (0.92 + min(0.08, margin)))))
        return float(self.calibrator.predict([value])[0]) if self.calibrator is not None else value

    def score_target_vectors(self, vectors: list[list[float]], batch_size: int = 50_000) -> np.ndarray:
        """Score shortlisted roots in large batches instead of one sklearn call per root."""
        if not vectors:
            return np.empty(0, dtype=np.float64)
        matrix = np.asarray(vectors, dtype=np.float32)
        if self.target_model is None:
            margins = np.maximum(0.0, matrix[:, 0] - matrix[:, 3])
            values = np.clip(matrix[:, 0] * (0.92 + np.minimum(0.08, margins)), 0.0, 0.999)
        else:
            size = max(1, int(batch_size))
            values = np.concatenate([
                self.target_model.predict_proba(matrix[start:start + size])[:, 1]
                for start in range(0, len(matrix), size)
            ])
        if self.calibrator is not None:
            values = np.asarray(self.calibrator.predict(values), dtype=np.float64)
        return values


class CrossEncoderReranker:
    def __init__(self, model_dir: str | Path, mode: str):
        self.mode = mode
        self.model = None
        path = Path(model_dir)
        if mode != "off" and path.exists():
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(str(path), device="cpu")
            except Exception:
                self.model = None

    def score(self, proposals: list[MatchProposal]) -> None:
        if self.model is None or not proposals:
            return
        values = np.asarray(self.model.predict(
            [(proposal.mention_text, proposal.matched_name) for proposal in proposals],
            batch_size=64,
            show_progress_bar=False,
        )).reshape(-1)
        if np.any((values < 0) | (values > 1)):
            values = 1.0 / (1.0 + np.exp(-values))
        for proposal, value in zip(proposals, values):
            proposal.cross_encoder_score = float(value)


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
            if proposal is None:
                reason = "NO_CANDIDATES"
            elif not eligible:
                reason = _guard_reason(proposal, matching_cfg) or "INSUFFICIENT_SUPPORT"
            elif runner and margin < minimum_margin:
                reason = "AMBIGUOUS_FINAL_TARGETS"
            elif confidence < scorer.system_threshold:
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
            }
            mention_results.append(result)
            if reason == "MATCH":
                valid.append({"position": mention.position, "confidence": confidence, "proposal": proposal,
                              "runner": runner[1] if runner else None, "runner_score": runner[0] if runner else None,
                              "margin": margin})
            elif proposal and (best_rejected is None or confidence > best_rejected[0]):
                best_rejected = (confidence, proposal)
        retrieved_roots = [root for root, _ in sorted(retrieved.items(), key=lambda item: (-item[1], item[0]))]
        connector = context["connector"]
        preferred_index = len(mention_results) - 1 if connector == "OBO" else 0
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
                chosen = max(valid, key=lambda item: item["position"]) if connector == "OBO" else min(valid, key=lambda item: item["position"])
                resolution = f"{connector}_{'RIGHT' if connector == 'OBO' else 'LEFT'}"
            confidence = chosen["confidence"]
            proposal = chosen["proposal"]
            runner = chosen["runner"]
            decision = _no_match(
                record, "MATCH", confidence, proposal=proposal, runner=runner,
                runner_score=chosen["runner_score"], margin=chosen["margin"],
                retrieved_roots=retrieved_roots, decision_tier=_decision_tier(proposal, matching_cfg),
            )
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


def run_matching(
    config: dict[str, Any], output_dir: str | Path, fresh_state: bool = False,
    diagnose_rules: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    paths = config.get("paths", {})
    prepared = Path(paths.get("prepared_dir", "data/prepared"))
    manifest = read_json(prepared / "manifest.json", {}) or {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = read_json(prepared / "jobs.json", []) or []
    if not jobs:
        raise ValueError("No prepared jobs found. Run scripts/prepare.py first.")
    account_id = str(jobs[0]["accountId"])
    persistent_state_root = Path(paths.get("state_dir", "state")) / account_id
    persistent_state_root.mkdir(parents=True, exist_ok=True)
    state_root = output / "state" if fresh_state else persistent_state_root
    state_root.mkdir(parents=True, exist_ok=True)
    graph_started = time.perf_counter()
    graph = VerifiedGraph(account_id, state_root / "graph.json", load_existing=not fresh_state)
    party_job: dict[str, dict[str, Any]] = {}
    all_parties = []
    for job in jobs:
        if job["accountId"] != account_id:
            raise ValueError("A run may contain only one account")
        all_parties.extend(job["verifiedParties"])
        for party in job["verifiedParties"]:
            party_job[party["partyId"]] = job
    graph.add_parties(all_parties)
    expansion_cfg = config.get("expansion", {})
    expansion_service = ExpansionService(
        expansion_cfg,
        paths.get("prompt", "prompts/party_expansion_v1.txt"),
        persistent_state_root / "expansions.json",
    )
    graph.apply_expansions(expansion_service.expand(all_parties))
    graph_updates = graph.reconcile_suggested_parents()
    graph.save()
    graph_seconds = time.perf_counter() - graph_started
    graph_rss_mb = _rss_mb()

    mapping_path = state_root / "mappings.jsonl"
    mapped = {row["admPartyId"] for row in read_jsonl(mapping_path)} if mapping_path.exists() else set()
    records = [PartyRecord(**row) for row in read_jsonl(prepared / "adm_records.jsonl")]
    limit = int(config.get("execution", {}).get("limit", 0) or 0)
    ids_file = str(config.get("execution", {}).get("ids_file", "") or "")
    if ids_file:
        selected = {line.strip() for line in Path(ids_file).read_text(encoding="utf-8").splitlines() if line.strip()}
        records = [record for record in records if record.adm_party_id in selected]
    if limit > 0:
        records = records[:limit]
    matchable = [record for record in records if record.eligible and record.adm_party_id not in mapped]
    execution = config.get("execution", {})
    workers = 1 if str(execution.get("mode", "parallel")) == "serial" else max(1, int(execution.get("workers", 10)))
    os.environ.setdefault("OMP_NUM_THREADS", str(execution.get("native_threads", 10)))
    index_started = time.perf_counter()
    retriever = MentionRetriever(matchable, config.get("retrieval", {}), workers=workers)
    index_seconds = time.perf_counter() - index_started
    index_rss_mb = _rss_mb()
    artifact_dir = Path(paths.get("artifacts_dir", "artifacts"))
    scorer = FeatureScorer(artifact_dir / "matcher.joblib", manifest.get("data_version"))
    strategy = str(config.get("matching", {}).get("strategy", "selective_hybrid"))
    rerank_mode = (
        str(config.get("matching", {}).get("cross_encoder_mode", "ambiguous"))
        if strategy != "feature" and scorer.artifact.get("cross_encoder_enabled", False)
        else "off"
    )
    effective_strategy = strategy if rerank_mode != "off" else "feature"
    reranker = CrossEncoderReranker(
        artifact_dir / "cross_encoder",
        rerank_mode,
    )
    proposals, retrieval_stats = collect_proposals(graph, retriever, scorer)
    decision_started = time.perf_counter()
    decisions = decide_records(matchable, proposals, graph, scorer, reranker, config)
    decision_seconds = time.perf_counter() - decision_started
    decision_rss_mb = _rss_mb()
    decision_by_id = {decision.adm_party_id: decision for decision in decisions}
    for record in records:
        if not record.eligible:
            decision_by_id[record.adm_party_id] = FinalDecision(
                account_id=account_id, adm_party_id=record.adm_party_id, raw_name=record.raw_name,
                decision="NO_MATCH", confidence=0.0, reason="INELIGIBLE_PARENT_VERIFIED",
            )
        elif record.adm_party_id in mapped:
            decision_by_id[record.adm_party_id] = FinalDecision(
                account_id=account_id, adm_party_id=record.adm_party_id, raw_name=record.raw_name,
                decision="NO_MATCH", confidence=0.0, reason="ALREADY_MAPPED",
            )
    decisions = [decision_by_id[record.adm_party_id] for record in records]

    request_cutoff = max(float(job.get("confidenceCutoff", 0.0)) for job in jobs)
    effective_cutoff = max(scorer.system_threshold, request_cutoff)
    default_job = jobs[0]
    for decision in decisions:
        event_job = party_job.get(str(decision.matched_member_id or decision.verified_party_id), default_job)
        decision_cutoff = max(scorer.system_threshold, float(event_job.get("confidenceCutoff", 0.0)))
        if decision.decision == "MATCH" and decision.confidence < decision_cutoff:
            decision.decision = "NO_MATCH"
            decision.reason = "BELOW_REQUEST_CUTOFF"

    rules_started = time.perf_counter()
    rules_scorer = copy.copy(scorer)
    rules_scorer.model = None
    rules_scorer.target_model = None
    rules_scorer.calibrator = None
    rules_scorer.system_threshold = float(scorer.artifact.get(
        "rules_system_threshold", config.get("decision", {}).get("fallback_system_threshold", 0.90),
    ))
    rules_scorer.plain_threshold = float(config.get("decision", {}).get(
        "rules_plain_threshold", rules_scorer.system_threshold,
    ))
    if not 0.0 <= rules_scorer.plain_threshold <= 1.0:
        raise ValueError("decision.rules_plain_threshold must be between 0 and 1")
    containment_floor = float(config.get("decision", {}).get("rules_containment_min_confidence", 0.40))
    if not 0.0 <= containment_floor <= 1.0:
        raise ValueError("decision.rules_containment_min_confidence must be between 0 and 1")
    for record_proposals in proposals.values():
        for proposal in record_proposals:
            proposal.feature_score = proposal.rules_score
            proposal.cross_encoder_score = None
    rules_by_id = {
        decision.adm_party_id: decision for decision in decide_records(
            matchable, proposals, graph, rules_scorer,
            CrossEncoderReranker(artifact_dir / "cross_encoder", "off"), config,
        )
    }
    for record in records:
        if record.adm_party_id not in rules_by_id:
            rules_by_id[record.adm_party_id] = FinalDecision(
                account_id=account_id, adm_party_id=record.adm_party_id, raw_name=record.raw_name,
                decision="NO_MATCH", confidence=0.0,
                reason="INELIGIBLE_PARENT_VERIFIED" if not record.eligible else "ALREADY_MAPPED",
            )
    rules_decisions = [rules_by_id[record.adm_party_id] for record in records]
    for decision in rules_decisions:
        event_job = party_job.get(str(decision.matched_member_id or decision.verified_party_id), default_job)
        if decision.decision_tier == "RULES_UNIQUE_CONTAINMENT":
            rules_cutoff = min(rules_scorer.plain_threshold, containment_floor)
        elif decision.connector is None and decision.connector_resolution is None:
            rules_cutoff = rules_scorer.plain_threshold
        else:
            rules_cutoff = rules_scorer.system_threshold
        decision_cutoff = max(rules_cutoff, float(event_job.get("confidenceCutoff", 0.0)))
        if decision.decision == "MATCH" and decision.confidence < decision_cutoff:
            decision.decision = "NO_MATCH"
            decision.reason = "BELOW_REQUEST_CUTOFF"
    rules_decision_seconds = time.perf_counter() - rules_started
    baseline_rules_decisions = rules_decisions
    # TEMPORARY DIAGNOSTIC HOOK: remove with diagnostics.py and --diagnose-rules
    # after rules behavior is settled. It only reads already-scored proposals and
    # writes a sidecar; no decision, graph, mapping, or event is changed here.
    rules_trace_stats = None
    if diagnose_rules:
        from .diagnostics import write_rules_candidate_trace
        trace_started = time.perf_counter()
        rules_trace_stats = write_rules_candidate_trace(
            output / "rules_candidate_trace.jsonl", matchable, proposals,
            {decision.adm_party_id: decision for decision in rules_decisions},
            prepared / "labels.jsonl", graph, rules_scorer, config,
        )
        rules_trace_stats["seconds"] = time.perf_counter() - trace_started
    rules_enhancement_stats = None
    if bool(config.get("decision", {}).get("rules_enhanced_enabled", False)):
        from .rules_enhancement import enhance_rules_decisions
        enhanced_started = time.perf_counter()
        rules_decisions, rules_enhancement_stats = enhance_rules_decisions(
            matchable, proposals,
            [rules_by_id[record.adm_party_id] for record in matchable],
            graph, retriever, rules_scorer, config, party_job, default_job,
        )
        enhanced_by_id = {decision.adm_party_id: decision for decision in rules_decisions}
        rules_decisions = [enhanced_by_id.get(record.adm_party_id, rules_by_id[record.adm_party_id])
                           for record in records]
        rules_enhancement_stats["seconds"] = time.perf_counter() - enhanced_started
    events: list[dict[str, Any]] = []
    for update in graph_updates:
        root_id = graph.root_id(update["child_id"])
        root = graph.nodes[root_id]
        job = party_job.get(update["child_id"], jobs[0])
        events.append({
            "accountId": account_id,
            "verifiedPartyId": root_id,
            "verifiedPartyName": root.party_name,
            "admPartyId": update["child_id"],
            "admPartyRawName": update["child_name"],
            "confidence": 1.0,
            "matchMethod": "graph:suggested_parent",
            "failedMethods": [],
            "jobId": job["jobId"],
            "shardId": job["shardId"],
        })
    new_mappings = []
    for decision in decisions:
        event_job = party_job.get(str(decision.matched_member_id or decision.verified_party_id), default_job)
        if decision.decision != "MATCH":
            continue
        decision.emitted = True
        events.append({
            "accountId": account_id,
            "verifiedPartyId": decision.verified_party_id,
            "verifiedPartyName": decision.verified_party_name,
            "admPartyId": decision.adm_party_id,
            "admPartyRawName": decision.raw_name,
            "confidence": round(decision.confidence, 6),
            "matchMethod": decision.match_method,
            "failedMethods": [],
            "jobId": event_job["jobId"],
            "shardId": event_job["shardId"],
        })
        new_mappings.append({
            "accountId": account_id,
            "admPartyId": decision.adm_party_id,
            "verifiedPartyId": decision.verified_party_id,
            "confidence": decision.confidence,
            "matchMethod": decision.match_method,
        })
    write_started = time.perf_counter()
    write_jsonl(output / "decisions.jsonl", (decision.to_dict() for decision in decisions))
    write_jsonl(output / "rules_decisions.jsonl", (decision.to_dict() for decision in rules_decisions))
    write_jsonl(output / "rules_baseline_decisions.jsonl", (decision.to_dict() for decision in baseline_rules_decisions))
    write_jsonl(output / "events.jsonl", events)
    if new_mappings and not fresh_state:
        existing = list(read_jsonl(mapping_path)) if mapping_path.exists() else []
        write_jsonl(mapping_path, [*existing, *new_mappings])
    write_seconds = time.perf_counter() - write_started
    stats = {
        "account_id": account_id,
        "records": len(records),
        "eligible_records": sum(record.eligible for record in records),
        "matchable_records": len(matchable),
        "verified_nodes": len(graph.nodes),
        "graph_path": str(graph.path.resolve()),
        "graph_updates": len(graph_updates),
        "events": len(events),
        "effective_cutoff": effective_cutoff,
        "system_threshold": scorer.system_threshold,
        "rules_system_threshold": rules_scorer.system_threshold,
        "rules_plain_threshold": rules_scorer.plain_threshold,
        "rules_containment_enabled": bool(config.get("decision", {}).get("rules_containment_enabled", False)),
        "rules_containment_min_confidence": containment_floor,
        "rules_containment_min_margin": float(config.get("decision", {}).get("rules_containment_min_margin", 0.08)),
        "rules_containment_min_lexical": float(config.get("decision", {}).get("rules_containment_min_lexical", 0.50)),
        "rules_containment_max_token_df": int(config.get("decision", {}).get("rules_containment_max_token_df", 3)),
        "rules_containment_matches": sum(
            decision.decision == "MATCH" and decision.decision_tier == "RULES_UNIQUE_CONTAINMENT"
            for decision in rules_decisions
        ),
        "strategy": effective_strategy,
        "requested_strategy": strategy,
        "cross_encoder_mode": rerank_mode,
        "identity_scorer": "trained_model" if scorer.model is not None else "fallback_rules",
        "target_scorer": "trained_model" if scorer.target_model is not None else "fallback_formula",
        "confidence_calibration": "isotonic" if scorer.calibrator is not None else "none",
        "model_version": str(scorer.artifact.get("model_version", "fallback-rules-v2")),
        "model_warning": scorer.load_warning,
        "expansion_version": str(config.get("expansion", {}).get("version", "v1")),
        "graph_schema_version": graph.schema_version,
        "run_id": output.name,
        "graph_seconds": graph_seconds,
        "graph_rss_mb": graph_rss_mb,
        "index_seconds": index_seconds,
        "index_rss_mb": index_rss_mb,
        "decision_seconds": decision_seconds,
        "rules_decision_seconds": rules_decision_seconds,
        "rules_enhanced_enabled": bool(config.get("decision", {}).get("rules_enhanced_enabled", False)),
        "rules_enhancement_seconds": (rules_enhancement_stats or {}).get("seconds"),
        **({"rules_enhancement": rules_enhancement_stats} if rules_enhancement_stats is not None else {}),
        **({"rules_diagnostic_trace": rules_trace_stats} if rules_trace_stats is not None else {}),
        "decision_rss_mb": decision_rss_mb,
        "write_seconds": write_seconds,
        "peak_rss_mb": max((value for value in (
            graph_rss_mb, index_rss_mb, decision_rss_mb,
            retrieval_stats.get("retrieval_rss_mb"), retrieval_stats.get("shortlist_rss_mb"),
            retrieval_stats.get("proposal_rss_mb"), _rss_mb(), _peak_rss_mb(),
        ) if value is not None), default=None),
        "total_seconds": time.perf_counter() - started,
        **retrieval_stats,
    }
    write_json(output / "run_stats.json", stats)
    return stats


def train_models(config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("paths", {})
    prepared = Path(paths.get("prepared_dir", "data/prepared"))
    artifact_dir = Path(paths.get("artifacts_dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(prepared / "manifest.json", {}) or {}
    data_version = str(manifest.get("data_version", ""))
    if not data_version:
        raise ValueError("Prepared data manifest is missing data_version. Run scripts/prepare.py again.")
    seed = int(config.get("execution", {}).get("seed", 42))
    random.seed(seed)
    records = {row["adm_party_id"]: PartyRecord(**row) for row in read_jsonl(prepared / "adm_records.jsonl")}
    labels = list(read_jsonl(prepared / "labels.jsonl"))
    parties = read_json(prepared / "verified_parties.json", []) or []
    party_names = [party["partyName"] for party in parties]
    party_ids = [party["partyId"] for party in parties]
    if len(party_names) < 2:
        raise ValueError("At least two verified parties are required for training")
    if SKLEARN_ERROR is not None:
        threshold = float(config.get("decision", {}).get("fallback_system_threshold", 0.90))
        artifact = {
            "identity_model": None,
            "target_model": None,
            "calibrator": None,
            "idf": _token_idf(party_names),
            "system_threshold": threshold,
            "rules_system_threshold": threshold,
            "identity_features": IDENTITY_FEATURES,
            "target_features": TARGET_FEATURES,
            "training_pairs": 0,
            "calibration_records": 0,
            "calibration_method": "none",
            "precision_target": float(config.get("decision", {}).get("precision_target", 0.98)),
            "fallback_reason": SKLEARN_ERROR,
            "model_version": "fallback-rules-v2",
            "data_version": data_version,
            "cross_encoder_enabled": False,
        }
        joblib.dump(artifact, artifact_dir / "matcher.joblib")
        report = {
            "training_pairs": 0,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "calibration_records": 0,
            "calibration_method": "none",
            "system_threshold": threshold,
            "rules_system_threshold": threshold,
            "cross_encoder": f"skipped: scikit-learn unavailable ({SKLEARN_ERROR})",
        }
        write_json(artifact_dir / "training_report.json", report)
        return report
    canonical_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, dtype=np.float32)
    canonical_matrix = canonical_vectorizer.fit_transform(party_names)
    canonical_index = NearestNeighbors(metric="cosine", algorithm="brute").fit(canonical_matrix)
    retrieval_cfg = config.get("retrieval", {})
    mention_texts = [
        mention.text for record in records.values() if record.eligible
        for mention in parse_mentions(record) if mention.normalized
    ]
    char_vectorizer = (
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, dtype=np.float32).fit(mention_texts)
        if mention_texts and bool(retrieval_cfg.get("char_tfidf_enabled", True)) else None
    )
    word_vectorizer = (
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32).fit(mention_texts)
        if mention_texts and bool(retrieval_cfg.get("word_tfidf_enabled", True)) else None
    )
    idf = _token_idf(party_names)
    training_rows = [label for label in labels if label.get("split") == "TRAIN" and label.get("scorable")]
    pair_rows = _build_pair_rows(
        training_rows, records, party_names, party_ids, canonical_vectorizer, canonical_index,
        negative_ratio=int(config.get("training", {}).get("negative_ratio", 3)),
        max_pairs=int(config.get("training", {}).get("max_identity_pairs", 100_000)),
    )
    if len(pair_rows) < 20 or len({row[2] for row in pair_rows}) < 2:
        pair_rows = _synthetic_pair_rows(party_names)
    x_train = _training_pair_matrix(
        [(left, right) for left, right, _ in pair_rows], idf,
        char_vectorizer, word_vectorizer, retrieval_cfg,
    )
    y_train = np.asarray([label for _, _, label in pair_rows], dtype=np.int8)
    identity_model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=160, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed,
    ).fit(x_train, y_train)

    calibration_rows = [label for label in labels if label.get("split") == "CALIBRATION" and label.get("scorable")]
    partitions = [[], [], []]
    for row in calibration_rows:
        bucket = int(row["adm_party_id"].replace("-", "")[:8], 16) % 3
        partitions[bucket].append(row)
    target_rows, isotonic_rows, threshold_rows = partitions
    if min(map(len, partitions)) < 5:
        target_rows, isotonic_rows, threshold_rows = [], [], calibration_rows
    target_examples, _ = _build_target_examples(
        target_rows, records, party_names, party_ids,
        canonical_vectorizer, canonical_index, identity_model, idf,
        char_vectorizer, word_vectorizer, retrieval_cfg,
    )
    _, isotonic_decisions = _build_target_examples(
        isotonic_rows, records, party_names, party_ids,
        canonical_vectorizer, canonical_index, identity_model, idf,
        char_vectorizer, word_vectorizer, retrieval_cfg,
    )
    _, threshold_decisions = _build_target_examples(
        threshold_rows, records, party_names, party_ids,
        canonical_vectorizer, canonical_index, identity_model, idf,
        char_vectorizer, word_vectorizer, retrieval_cfg,
    )
    _, rules_threshold_decisions = _build_target_examples(
        threshold_rows, records, party_names, party_ids,
        canonical_vectorizer, canonical_index, None, idf,
        char_vectorizer, word_vectorizer, retrieval_cfg,
    )
    target_model = None
    if target_examples and len({label for _, label in target_examples}) > 1:
        target_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        target_model.fit(
            np.asarray([features for features, _ in target_examples], dtype=np.float32),
            np.asarray([label for _, label in target_examples], dtype=np.int8),
        )
    isotonic_examples = _rescore_decision_examples(isotonic_decisions, target_model)
    decision_examples = _rescore_decision_examples(threshold_decisions, target_model)
    calibrator = None
    calibration_method = "none"
    requested_calibration = str(config.get("decision", {}).get("calibration_method", "isotonic"))
    if (
        requested_calibration == "isotonic"
        and IsotonicRegression is not None
        and len(isotonic_examples) >= 20
        and len({bool(correct) for _, correct in isotonic_examples}) > 1
        and len({round(float(score), 6) for score, _ in isotonic_examples}) >= 3
    ):
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        raw_scores = np.asarray([score for score, _ in isotonic_examples], dtype=np.float64)
        outcomes = np.asarray([correct for _, correct in isotonic_examples], dtype=np.int8)
        calibrator.fit(raw_scores, outcomes)
        calibrated_scores = calibrator.predict([score for score, _ in decision_examples])
        decision_examples = [
            (float(score), bool(correct))
            for score, (_, correct) in zip(calibrated_scores, decision_examples)
        ]
        calibration_method = "isotonic"
    precision_target = float(config.get("decision", {}).get("precision_target", 0.98))
    fallback = float(config.get("decision", {}).get("fallback_system_threshold", 0.90))
    requested_minimum = int(config.get("decision", {}).get("minimum_calibration_matches", 100))
    available_decisions = len(decision_examples)
    minimum_emitted = min(requested_minimum, max(5, available_decisions // 2))
    threshold = _select_threshold(decision_examples, precision_target, fallback, minimum_emitted)
    rules_examples = _rescore_decision_examples(rules_threshold_decisions, None)
    rules_threshold = _select_threshold(rules_examples, precision_target, fallback, minimum_emitted)
    cross_encoder_status = "disabled"
    if bool(config.get("training", {}).get("train_cross_encoder", False)):
        cross_encoder_status = _train_cross_encoder(config, pair_rows, artifact_dir / "cross_encoder")
    artifact = {
        "identity_model": identity_model,
        "target_model": target_model,
        "calibrator": calibrator,
        "idf": idf,
        "system_threshold": threshold,
        "rules_system_threshold": rules_threshold,
        "identity_features": IDENTITY_FEATURES,
        "target_features": TARGET_FEATURES,
        "training_pairs": len(pair_rows),
        "calibration_records": len(calibration_rows),
        "precision_target": precision_target,
        "model_version": f"feature-v4-{data_version}-{len(pair_rows)}-{seed}",
        "data_version": data_version,
        "cross_encoder_enabled": cross_encoder_status == "trained",
    }
    joblib.dump(artifact, artifact_dir / "matcher.joblib")
    report = {
        "training_pairs": len(pair_rows),
        "positive_pairs": int(y_train.sum()),
        "negative_pairs": int(len(y_train) - y_train.sum()),
        "calibration_records": len(calibration_rows),
        "target_training_records": len(target_rows),
        "isotonic_records": len(isotonic_rows),
        "threshold_records": len(threshold_rows),
        "calibration_method": calibration_method,
        "system_threshold": threshold,
        "rules_system_threshold": rules_threshold,
        "cross_encoder": cross_encoder_status,
    }
    write_json(artifact_dir / "training_report.json", report)
    return report


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


def _training_pair_matrix(
    pairs: list[tuple[str, str]],
    idf: dict[str, float],
    char_vectorizer: TfidfVectorizer | None,
    word_vectorizer: TfidfVectorizer | None,
    retrieval_config: dict[str, Any],
) -> np.ndarray:
    """Use the same sparse score definitions as inference, in bounded batches."""
    batches = []
    for start in range(0, len(pairs), 10_000):
        batch = pairs[start:start + 10_000]
        lefts, rights = zip(*batch)

        def cosine(vectorizer: TfidfVectorizer | None) -> np.ndarray:
            if vectorizer is None:
                return np.zeros(len(batch), dtype=np.float32)
            left_matrix = vectorizer.transform(lefts)
            right_matrix = vectorizer.transform(rights)
            return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()

        char_scores = cosine(char_vectorizer)
        word_scores = cosine(word_vectorizer)
        char_min = float(retrieval_config.get("char_min_score", retrieval_config.get("lexical_min_score", 0.28)))
        word_min = float(retrieval_config.get("word_min_score", 0.18))
        rrf_k = float(retrieval_config.get("rrf_k", 60))
        vectors = []
        for (left, right), char_score, word_score in zip(batch, char_scores, word_scores):
            if normalize_name(left) and normalize_name(left) == normalize_name(right):
                char_score = 1.0 if char_vectorizer is not None else 0.0
                word_score = 1.0 if word_vectorizer is not None else 0.0
            rrf_score = 0.0
            if bool(retrieval_config.get("rrf_enabled", True)):
                rrf_score = (
                    (1.0 / (rrf_k + 1.0) if char_vectorizer is not None and char_score >= char_min else 0.0)
                    + (1.0 / (rrf_k + 1.0) if word_vectorizer is not None and word_score >= word_min else 0.0)
                )
            vectors.append(pair_features(
                left, right, float(char_score), float(word_score), rrf_score,
                None, "official", 1, idf,
            ))
        batches.append(np.asarray(vectors, dtype=np.float32))
    return np.concatenate(batches) if batches else np.empty((0, len(IDENTITY_FEATURES)), dtype=np.float32)


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


def _build_pair_rows(
    label_rows: list[dict[str, Any]],
    records: dict[str, PartyRecord],
    party_names: list[str],
    party_ids: list[str],
    vectorizer: TfidfVectorizer,
    index: NearestNeighbors,
    negative_ratio: int,
    max_pairs: int,
) -> list[tuple[str, str, int]]:
    party_name_by_id = dict(zip(party_ids, party_names))
    usable = []
    for label in label_rows:
        record = records.get(label["adm_party_id"])
        expected = party_name_by_id.get(label.get("expected_party_id"))
        if not record or not expected or len(parse_mentions(record)) != 1:
            continue
        usable.append((record.raw_name, expected, label.get("expected_party_id")))
    if not usable:
        return []
    count = min(len(party_names), max(negative_ratio + 3, 10))
    distances, indices = index.kneighbors(vectorizer.transform([row[0] for row in usable]), n_neighbors=count)
    rows: list[tuple[str, str, int]] = []
    for (raw, expected, expected_id), candidate_indices in zip(usable, indices):
        rows.append((raw, expected, 1))
        added = 0
        for candidate_index in candidate_indices:
            if party_ids[int(candidate_index)] == expected_id:
                continue
            if base_name(party_names[int(candidate_index)]) == base_name(expected):
                continue
            rows.append((raw, party_names[int(candidate_index)], 0))
            added += 1
            if added >= negative_ratio:
                break
        if len(rows) >= max_pairs:
            break
    return rows[:max_pairs]


def _synthetic_pair_rows(party_names: list[str]) -> list[tuple[str, str, int]]:
    rows = []
    for index, name in enumerate(party_names):
        rows.extend([
            (name, name, 1),
            (normalize_name(name), name, 1),
            (name.replace(" ", ""), name, 1),
            (name, party_names[(index + 1) % len(party_names)], 0),
            (name, party_names[(index + 2) % len(party_names)], 0),
        ])
    return rows


def _build_target_examples(
    label_rows: list[dict[str, Any]],
    records: dict[str, PartyRecord],
    party_names: list[str],
    party_ids: list[str],
    vectorizer: TfidfVectorizer,
    index: NearestNeighbors,
    identity_model: Any | None,
    idf: dict[str, float],
    char_vectorizer: TfidfVectorizer | None,
    word_vectorizer: TfidfVectorizer | None,
    retrieval_config: dict[str, Any],
) -> tuple[list[tuple[list[float], int]], list[dict[str, Any]]]:
    expected_by_id = dict(zip(party_ids, party_names))
    usable = [(row, records.get(row["adm_party_id"])) for row in label_rows]
    usable = [
        (row, record) for row, record in usable
        if record and row.get("expected_party_id") in expected_by_id and len(parse_mentions(record)) == 1
    ]
    if not usable:
        return [], []
    count = min(10, len(party_names))
    _, indices = index.kneighbors(vectorizer.transform([record.raw_name for _, record in usable]), n_neighbors=count)
    candidate_groups = []
    pairs = []
    for (label, record), candidate_indices in zip(usable, indices):
        candidate_ids = [party_ids[int(index_value)] for index_value in candidate_indices]
        if label["expected_party_id"] not in candidate_ids:
            candidate_ids.append(label["expected_party_id"])
        candidate_groups.append((label, candidate_ids))
        pairs.extend((record.raw_name, expected_by_id[candidate_id]) for candidate_id in candidate_ids)
    vectors = _training_pair_matrix(pairs, idf, char_vectorizer, word_vectorizer, retrieval_config)
    identity_scores = (
        identity_model.predict_proba(vectors)[:, 1] if identity_model is not None
        else np.asarray([rule_identity_score(vector) for vector in vectors])
    )
    examples, decisions = [], []
    offset = 0
    for label, candidate_ids in candidate_groups:
        candidates = [
            (candidate_id, expected_by_id[candidate_id], float(identity_scores[offset + index_value]),
             float(vectors[offset + index_value, 0]))
            for index_value, candidate_id in enumerate(candidate_ids)
        ]
        offset += len(candidate_ids)
        candidates.sort(key=lambda item: (-item[2], item[0]))
        for candidate_id, name, identity, exact in candidates:
            runner = max((item[2] for item in candidates if item[0] != candidate_id), default=0.0)
            target_vector = [identity, identity, 0.0, runner, identity - runner, exact, 1.0, 1.0, 1.0]
            examples.append((target_vector, int(candidate_id == label["expected_party_id"])))
        decisions.append({
            "expected_id": label["expected_party_id"],
            "candidates": candidates,
        })
    return examples, decisions


def _rescore_decision_examples(decisions: list[dict[str, Any]], model: Any | None) -> list[tuple[float, bool]]:
    output = []
    for decision in decisions:
        candidates = decision["candidates"]
        scored = []
        for candidate_id, _, identity, exact in candidates:
            runner = max((item[2] for item in candidates if item[0] != candidate_id), default=0.0)
            vector = [identity, identity, 0.0, runner, identity - runner, exact, 1.0, 1.0, 1.0]
            if model is None:
                score = identity * (0.92 + min(0.08, max(0.0, identity - runner)))
            else:
                score = float(model.predict_proba(np.asarray([vector], dtype=np.float32))[0, 1])
            scored.append((float(score), candidate_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        output.append((scored[0][0], scored[0][1] == decision["expected_id"]))
    return output


def _select_threshold(
    examples: list[tuple[float, bool]] | list[dict[str, Any]],
    target: float,
    fallback: float,
    minimum_emitted: int,
) -> float:
    if not examples or isinstance(examples[0], dict):
        return fallback
    best = None
    for threshold in sorted({round(float(score), 6) for score, _ in examples}, reverse=True):
        emitted = [correct for score, correct in examples if score >= threshold]
        if len(emitted) < minimum_emitted:
            continue
        precision = sum(emitted) / len(emitted)
        recall_count = sum(emitted)
        if precision >= target and (best is None or recall_count > best[0] or (recall_count == best[0] and threshold < best[1])):
            best = (recall_count, threshold)
    return float(best[1] if best else fallback)


def _train_cross_encoder(config: dict[str, Any], rows: list[tuple[str, str, int]], output_dir: Path) -> str:
    if len(rows) < 100:
        return "skipped: insufficient reliable pairs"
    try:
        from torch.utils.data import DataLoader
        from sentence_transformers import CrossEncoder, InputExample
        model_name = str(config.get("matching", {}).get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L6-v2"))
        model = CrossEncoder(model_name, num_labels=1, device="cpu")
        examples = [InputExample(texts=[left, right], label=float(label)) for left, right, label in rows]
        loader = DataLoader(examples, shuffle=True, batch_size=int(config.get("training", {}).get("cross_encoder_batch_size", 32)))
        model.fit(
            train_dataloader=loader,
            epochs=int(config.get("training", {}).get("cross_encoder_epochs", 1)),
            warmup_steps=max(1, len(loader) // 10),
            output_path=str(output_dir),
            show_progress_bar=True,
        )
        return "trained"
    except Exception as exc:
        return f"skipped: {exc}"


def _peak_rss_mb() -> float | None:
    try:
        import psutil
        memory = psutil.Process().memory_info()
        if hasattr(memory, "peak_wset"):
            return memory.peak_wset / (1024 * 1024)
    except ImportError:
        pass
    try:
        import resource
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024
    except Exception:
        return None


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None
