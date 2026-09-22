from __future__ import annotations

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
    HistGradientBoostingClassifier = TfidfVectorizer = LogisticRegression = NearestNeighbors = None
    SKLEARN_ERROR = "disabled by PARTY_MATCHING_DISABLE_SKLEARN"
else:
  try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_ERROR = None
  except Exception as exc:  # permits exact/brute smoke runs on restricted machines
    HistGradientBoostingClassifier = TfidfVectorizer = LogisticRegression = NearestNeighbors = None
    SKLEARN_ERROR = str(exc)

from .domain import (
    FinalDecision,
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
    "exact", "base_exact", "compact_exact", "char_ratio", "token_jaccard",
    "weighted_token_overlap", "token_containment", "digit_conflict", "length_ratio",
    "lexical_score", "embedding_score", "official", "abbreviation", "collision_penalty",
]
TARGET_FEATURES = [
    "identity", "cross_encoder", "has_cross_encoder", "runner_up_identity",
    "identity_margin", "exact", "collision_penalty", "supporting_mentions", "graph_depth",
]


class MentionRetriever:
    def __init__(self, records: list[PartyRecord], config: dict[str, Any], workers: int = 1):
        self.config = config
        self.workers = workers
        self.records = {record.adm_party_id: record for record in records}
        self.mentions = self._parse(records)
        self.exact: dict[str, list[int]] = defaultdict(list)
        for index, mention in enumerate(self.mentions):
            if mention.normalized:
                self.exact[mention.normalized].append(index)
        self.vectorizer: TfidfVectorizer | None = None
        self.lexical_matrix = None
        texts = [mention.text for mention in self.mentions]
        if texts and SKLEARN_ERROR is None:
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, dtype=np.float32)
            self.lexical_matrix = self.vectorizer.fit_transform(texts)
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
        """Return bounded verified-query candidates for each indexed ADM mention."""
        by_mention: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        query_norms = [normalize_name(query) for query in queries]
        for query_index, normalized in enumerate(query_norms):
            for mention_index in self.exact.get(normalized, []):
                by_mention[mention_index][query_index] = {
                    "sources": {"exact", "lexical"},
                    "exact": 1.0,
                    "lexical": 1.0,
                    "embedding": None,
                }
        if self.lexical_matrix is not None and queries:
            query_matrix = self.vectorizer.transform(queries)
            similarities = sp_matmul_topn(
                self.lexical_matrix,
                query_matrix.T.tocsr(),
                top_n=max(1, int(top_n)),
                threshold=float(self.config.get("lexical_min_score", 0.28)),
                sort=True,
                n_threads=max(1, self.workers),
            )
            for mention_index in range(similarities.shape[0]):
                first, last = similarities.indptr[mention_index:mention_index + 2]
                for query_index, score_value in zip(
                    similarities.indices[first:last], similarities.data[first:last],
                ):
                    query_index = int(query_index)
                    hit = by_mention[mention_index].setdefault(query_index, {
                        "sources": set(), "exact": 0.0, "lexical": 0.0, "embedding": None,
                    })
                    hit["sources"].add("lexical")
                    hit["lexical"] = max(hit["lexical"], float(score_value))
                    if self.mentions[mention_index].normalized == query_norms[query_index]:
                        hit["sources"].add("exact")
                        hit["exact"] = 1.0
        elif queries and len(self.mentions) > 5_000:
            raise RuntimeError(f"scikit-learn is required for more than 5,000 mentions: {SKLEARN_ERROR}")
        if self.embedding_model is not None and self.embedding_matrix is not None and queries:
            count = min(max(1, int(self.config.get("embedding_top_k", 100))), len(self.mentions))
            minimum = float(self.config.get("embedding_min_score", 0.42))
            for start in range(0, len(queries), 128):
                vectors = self.embedding_model.encode(
                    queries[start:start + 128], batch_size=128, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                ).astype(np.float32)
                touched: set[int] = set()
                for local_index, vector in enumerate(vectors):
                    query_index = start + local_index
                    if self.ann_index is not None:
                        matches = self.ann_index.search(vector, count=count)
                        keys = np.atleast_1d(matches.keys)
                        similarities = 1.0 - np.atleast_1d(matches.distances)
                    else:
                        all_scores = self.embedding_matrix @ vector
                        keys = np.argpartition(all_scores, -count)[-count:]
                        similarities = all_scores[keys]
                    for mention_index, score_value in zip(keys, similarities):
                        score = float(score_value)
                        if score < minimum:
                            continue
                        mention_index = int(mention_index)
                        hit = by_mention[mention_index].setdefault(query_index, {
                            "sources": set(), "exact": 0.0, "lexical": 0.0, "embedding": None,
                        })
                        hit["sources"].add("embedding")
                        hit["embedding"] = max(hit["embedding"] or -1.0, score)
                        touched.add(mention_index)
                for mention_index in touched:
                    hits = by_mention[mention_index]
                    exact = {key: value for key, value in hits.items() if value["exact"]}
                    ranked = sorted(
                        ((key, value) for key, value in hits.items() if key not in exact),
                        key=lambda item: -max(item[1]["lexical"], item[1]["embedding"] or 0.0),
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
        self.idf = self.artifact.get("idf", {})
        self.system_threshold = float(self.artifact.get("system_threshold", 0.90))

    def score_proposals(self, proposals: list[MatchProposal], batch_size: int = 50_000) -> None:
        if not proposals:
            return
        size = max(1, int(batch_size))
        for start in range(0, len(proposals), size):
            batch = proposals[start:start + size]
            vectors = np.asarray([
                pair_features(
                    proposal.mention_text, proposal.matched_name,
                    proposal.lexical_score, proposal.embedding_score,
                    proposal.candidate_type, proposal.candidate_collision_count, self.idf,
                ) for proposal in batch
            ], dtype=np.float32)
            if self.model is not None:
                scores = self.model.predict_proba(vectors)[:, 1]
            else:
                scores = np.asarray([rule_identity_score(vector) for vector in vectors])
            for proposal, score in zip(batch, scores):
                value = float(score)
                if proposal.candidate_type != "official" and proposal.candidate_confidence is not None:
                    value = min(value, proposal.candidate_confidence)
                if proposal.exact and proposal.candidate_collision_count == 1:
                    exact_floor = 0.999 if proposal.candidate_type == "official" else float(proposal.candidate_confidence or 0.0)
                    value = max(value, exact_floor)
                proposal.feature_score = value

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
            return float(self.target_model.predict_proba(np.asarray([vector], dtype=np.float32))[0, 1])
        margin = max(0.0, identity - runner_up_identity)
        return float(max(0.0, min(0.999, identity * (0.92 + min(0.08, margin)))))

    def score_target_vectors(self, vectors: list[list[float]], batch_size: int = 50_000) -> np.ndarray:
        """Score shortlisted roots in large batches instead of one sklearn call per root."""
        if not vectors:
            return np.empty(0, dtype=np.float64)
        matrix = np.asarray(vectors, dtype=np.float64)
        if self.target_model is None:
            margins = np.maximum(0.0, matrix[:, 0] - matrix[:, 3])
            return np.clip(matrix[:, 0] * (0.92 + np.minimum(0.08, margins)), 0.0, 0.999)
        size = max(1, int(batch_size))
        return np.concatenate([
            self.target_model.predict_proba(matrix[start:start + size])[:, 1]
            for start in range(0, len(matrix), size)
        ])


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
    raw_hits = sum(len(hits) for hits in indexed.values())
    for mention_index, hits in indexed.items():
        by_root = shortlists[mention_index]
        for query_index, evidence in hits.items():
            party_id, query_name, candidate_type, source, candidate_confidence = queries[query_index]
            root_id = graph.root_id(party_id)
            exact = float(evidence["exact"])
            lexical = float(evidence["lexical"])
            embedding = float(evidence["embedding"] or 0.0)
            priority = 2.0 if exact else max(lexical, embedding)
            variants = by_root.setdefault(root_id, {})
            variant_key = (party_id, normalize_name(query_name))
            pending = (
                priority, party_id, query_name, candidate_type, source,
                candidate_confidence, sorted(evidence["sources"]), exact,
                lexical, (embedding if evidence["embedding"] is not None else None),
            )
            current = variants.get(variant_key)
            if current is None or (priority, query_name) > (current[0], current[2]):
                variants[variant_key] = pending
        for root_id, variants in tuple(by_root.items()):
            if len(variants) > variants_per_root:
                retained = sorted(
                    variants.items(), key=lambda item: (-item[1][0], item[0]),
                )[:variants_per_root]
                by_root[root_id] = dict(retained)
        if len(by_root) > root_limit:
            ranked = sorted(
                by_root.items(),
                key=lambda item: (-max(value[0] for value in item[1].values()), item[0]),
            )[:root_limit]
            shortlists[mention_index] = dict(ranked)
    proposals: list[MatchProposal] = []
    for mention_index, by_root in shortlists.items():
        mention = retriever.mentions[mention_index]
        for root_id, variants in by_root.items():
            root = graph.nodes[root_id]
            for pending in variants.values():
                (
                    _, party_id, query_name, candidate_type, source,
                    candidate_confidence, retrieval_sources, exact,
                    lexical, embedding,
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
                    embedding_score=embedding,
                ))
    scorer.score_proposals(
        proposals,
        batch_size=int(retriever.config.get("identity_batch_size", 50_000)),
    )
    by_record: dict[str, list[MatchProposal]] = defaultdict(list)
    for proposal in proposals:
        by_record[proposal.adm_party_id].append(proposal)
    return dict(by_record), {
        "verified_queries": len(queries),
        "raw_retrieval_hits": raw_hits,
        "scored_proposals": len(proposals),
        "retrieved_proposals": len(proposals),
        "retrieval_seconds": time.perf_counter() - started,
        "embedding_warning": retriever.embedding_warning,
    }


def decide_records(
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
            "retrieved_roots": sorted(root_best),
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
        confidence, winner = scored_roots[0]
        runner_score, runner = scored_roots[1] if len(scored_roots) > 1 else (0.0, None)
        margin = confidence - runner_score
        if runner and margin < minimum_margin:
            reason = "CANDIDATE_COLLISION" if winner.candidate_collision_count > 1 else "AMBIGUOUS_FINAL_TARGETS"
            completed[record.adm_party_id] = _no_match(
                record, reason, confidence=confidence, proposal=winner,
                runner=runner, runner_score=runner_score, margin=margin,
                retrieved_roots=retrieved_roots, parse_warning=parse_warning,
            )
            continue
        if confidence < scorer.system_threshold:
            completed[record.adm_party_id] = _no_match(
                record, "INSUFFICIENT_SUPPORT", confidence=confidence, proposal=winner,
                runner=runner, runner_score=runner_score, margin=margin,
                retrieved_roots=retrieved_roots, parse_warning=parse_warning,
            )
            continue
        path = graph.path_to_root(winner.owner_party_id)
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
            match_method=_match_method(winner),
            retrieval_sources=winner.retrieval_sources,
            identity_score=winner.feature_score,
            cross_encoder_score=winner.cross_encoder_score,
            runner_up_party_id=runner.root_party_id if runner else None,
            runner_up_score=runner_score if runner else None,
            margin=margin,
            graph_path=path,
            retrieved_root_ids=retrieved_roots,
            parse_warning=parse_warning,
        )
    return [completed[record.adm_party_id] for record in records]


def run_matching(config: dict[str, Any], output_dir: str | Path, fresh_state: bool = False) -> dict[str, Any]:
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
        "graph_updates": len(graph_updates),
        "events": len(events),
        "effective_cutoff": effective_cutoff,
        "system_threshold": scorer.system_threshold,
        "strategy": effective_strategy,
        "requested_strategy": strategy,
        "cross_encoder_mode": rerank_mode,
        "model_version": str(scorer.artifact.get("model_version", "fallback-rules-v1")),
        "model_warning": scorer.load_warning,
        "expansion_version": str(config.get("expansion", {}).get("version", "v1")),
        "graph_schema_version": graph.schema_version,
        "run_id": output.name,
        "graph_seconds": graph_seconds,
        "index_seconds": index_seconds,
        "decision_seconds": decision_seconds,
        "write_seconds": write_seconds,
        "peak_rss_mb": _peak_rss_mb(),
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
            "idf": _token_idf(party_names),
            "system_threshold": threshold,
            "identity_features": IDENTITY_FEATURES,
            "target_features": TARGET_FEATURES,
            "training_pairs": 0,
            "calibration_records": 0,
            "precision_target": float(config.get("decision", {}).get("precision_target", 0.98)),
            "fallback_reason": SKLEARN_ERROR,
            "model_version": "fallback-rules-v1",
            "data_version": data_version,
            "cross_encoder_enabled": False,
        }
        joblib.dump(artifact, artifact_dir / "matcher.joblib")
        report = {
            "training_pairs": 0,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "calibration_records": 0,
            "system_threshold": threshold,
            "cross_encoder": f"skipped: scikit-learn unavailable ({SKLEARN_ERROR})",
        }
        write_json(artifact_dir / "training_report.json", report)
        return report
    canonical_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, dtype=np.float32)
    canonical_matrix = canonical_vectorizer.fit_transform(party_names)
    canonical_index = NearestNeighbors(metric="cosine", algorithm="brute").fit(canonical_matrix)
    idf = _token_idf(party_names)
    training_rows = [label for label in labels if label.get("split") == "TRAIN" and label.get("scorable")]
    pair_rows = _build_pair_rows(
        training_rows, records, party_names, party_ids, canonical_vectorizer, canonical_index,
        negative_ratio=int(config.get("training", {}).get("negative_ratio", 3)),
        max_pairs=int(config.get("training", {}).get("max_identity_pairs", 100_000)),
    )
    if len(pair_rows) < 20 or len({row[2] for row in pair_rows}) < 2:
        pair_rows = _synthetic_pair_rows(party_names)
    x_train = np.asarray([pair_features(a, b, char_similarity(a, b), None, "official", 1, idf) for a, b, _ in pair_rows], dtype=np.float32)
    y_train = np.asarray([label for _, _, label in pair_rows], dtype=np.int8)
    identity_model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=160, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed,
    ).fit(x_train, y_train)

    calibration_rows = [label for label in labels if label.get("split") == "CALIBRATION" and label.get("scorable")]
    target_rows = [row for row in calibration_rows if sum(map(ord, row["adm_party_id"])) % 2 == 0]
    threshold_rows = [row for row in calibration_rows if sum(map(ord, row["adm_party_id"])) % 2 == 1]
    if len(target_rows) < 5 or len(threshold_rows) < 5:
        target_rows, threshold_rows = [], calibration_rows
    target_examples, _ = _build_target_examples(
        target_rows, records, party_names, party_ids,
        canonical_vectorizer, canonical_index, identity_model, idf,
    )
    _, decision_examples = _build_target_examples(
        threshold_rows, records, party_names, party_ids,
        canonical_vectorizer, canonical_index, identity_model, idf,
    )
    target_model = None
    if target_examples and len({label for _, label in target_examples}) > 1:
        target_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        target_model.fit(
            np.asarray([features for features, _ in target_examples], dtype=np.float32),
            np.asarray([label for _, label in target_examples], dtype=np.int8),
        )
        decision_examples = _rescore_decision_examples(decision_examples, target_model)
    precision_target = float(config.get("decision", {}).get("precision_target", 0.98))
    fallback = float(config.get("decision", {}).get("fallback_system_threshold", 0.90))
    requested_minimum = int(config.get("decision", {}).get("minimum_calibration_matches", 100))
    available_decisions = len(decision_examples)
    minimum_emitted = min(requested_minimum, max(5, available_decisions // 2))
    threshold = _select_threshold(decision_examples, precision_target, fallback, minimum_emitted)
    cross_encoder_status = "disabled"
    if bool(config.get("training", {}).get("train_cross_encoder", False)):
        cross_encoder_status = _train_cross_encoder(config, pair_rows, artifact_dir / "cross_encoder")
    artifact = {
        "identity_model": identity_model,
        "target_model": target_model,
        "idf": idf,
        "system_threshold": threshold,
        "identity_features": IDENTITY_FEATURES,
        "target_features": TARGET_FEATURES,
        "training_pairs": len(pair_rows),
        "calibration_records": len(calibration_rows),
        "precision_target": precision_target,
        "model_version": f"feature-v1-{data_version}-{len(pair_rows)}-{seed}",
        "data_version": data_version,
        "cross_encoder_enabled": cross_encoder_status == "trained",
    }
    joblib.dump(artifact, artifact_dir / "matcher.joblib")
    report = {
        "training_pairs": len(pair_rows),
        "positive_pairs": int(y_train.sum()),
        "negative_pairs": int(len(y_train) - y_train.sum()),
        "calibration_records": len(calibration_rows),
        "system_threshold": threshold,
        "cross_encoder": cross_encoder_status,
    }
    write_json(artifact_dir / "training_report.json", report)
    return report


def pair_features(
    left: str,
    right: str,
    lexical_score: float,
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
    digits_left = {token for token in left_tokens if any(char.isdigit() for char in token)}
    digits_right = {token for token in right_tokens if any(char.isdigit() for char in token)}
    return [
        float(bool(left_norm) and left_norm == right_norm),
        float(bool(base_name(left)) and base_name(left) == base_name(right)),
        float(bool(compact_name(left)) and compact_name(left) == compact_name(right)),
        char_similarity(left_norm, right_norm),
        len(intersection) / max(1, len(union)),
        shared_weight / total_weight,
        len(intersection) / max(1, min(len(left_tokens), len(right_tokens))),
        float(bool(digits_left and digits_right and digits_left != digits_right)),
        min(len(left_norm), len(right_norm)) / max(1, max(len(left_norm), len(right_norm))),
        float(lexical_score),
        float(embedding_score or 0.0),
        float(candidate_type == "official"),
        float(candidate_type == "abbreviation"),
        1.0 / max(1, collision_count),
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
    if values[0]:
        return 0.999
    if values[1]:
        return 0.985
    if values[2]:
        return 0.975
    score = 0.38 * values[3] + 0.24 * values[5] + 0.14 * values[6] + 0.12 * values[9] + 0.10 * values[10]
    score += 0.02 * values[11] - 0.35 * values[7]
    return max(0.001, min(0.97, score))


def char_similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.fuzz import ratio
        return float(ratio(left, right)) / 100.0
    except ImportError:
        return SequenceMatcher(None, left, right).ratio()


def _combined_identity(proposal: MatchProposal) -> float:
    if proposal.cross_encoder_score is None:
        return proposal.feature_score
    return 0.55 * proposal.feature_score + 0.45 * proposal.cross_encoder_score


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
    if "embedding" in sources and "lexical" in sources:
        return "feature:lexical_embedding"
    if "embedding" in sources:
        return "feature:embedding"
    return "feature:lexical"


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
        retrieval_sources=proposal.retrieval_sources if proposal else [],
        identity_score=proposal.feature_score if proposal else None,
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
        similarity = char_similarity(record.raw_name, expected)
        reliable_identity = (
            normalize_name(record.raw_name) == normalize_name(expected)
            or base_name(record.raw_name) == base_name(expected)
            or similarity >= 0.82
        )
        if not reliable_identity:
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
    identity_model: Any,
    idf: dict[str, float],
) -> tuple[list[tuple[list[float], int]], list[dict[str, Any]]]:
    expected_by_id = dict(zip(party_ids, party_names))
    usable = [(row, records.get(row["adm_party_id"])) for row in label_rows]
    usable = [(row, record) for row, record in usable if record and row.get("expected_party_id") in expected_by_id]
    if not usable:
        return [], []
    count = min(10, len(party_names))
    _, indices = index.kneighbors(vectorizer.transform([record.raw_name for _, record in usable]), n_neighbors=count)
    examples, decisions = [], []
    for (label, record), candidate_indices in zip(usable, indices):
        candidate_ids = [party_ids[int(index_value)] for index_value in candidate_indices]
        if label["expected_party_id"] not in candidate_ids:
            candidate_ids.append(label["expected_party_id"])
        candidates = []
        for candidate_id in candidate_ids:
            name = expected_by_id[candidate_id]
            vector = pair_features(record.raw_name, name, char_similarity(record.raw_name, name), None, "official", 1, idf)
            identity = float(identity_model.predict_proba(np.asarray([vector], dtype=np.float32))[0, 1])
            candidates.append((candidate_id, name, identity, vector[0]))
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


def _rescore_decision_examples(decisions: list[dict[str, Any]], model: Any) -> list[tuple[float, bool]]:
    output = []
    for decision in decisions:
        candidates = decision["candidates"]
        scored = []
        for candidate_id, _, identity, exact in candidates:
            runner = max((item[2] for item in candidates if item[0] != candidate_id), default=0.0)
            vector = [identity, identity, 0.0, runner, identity - runner, exact, 1.0, 1.0, 1.0]
            scored.append((float(model.predict_proba(np.asarray([vector], dtype=np.float32))[0, 1]), candidate_id))
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
        import resource
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024
    except Exception:
        return None
