"""TEMPORARY SHADOW EXPERIMENT: compare candidate views without emitting matches.

Remove this module, its CLI flag, and the run_matching hook after choosing a
single production design. No entity-specific aliases or held-out labels are
used to generate candidates or make decisions; labels are read only for eval.
"""

from __future__ import annotations

import copy
import csv
import gc
import re
import time
from collections import ChainMap, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .domain import PartyRecord, base_name, compact_name, normalize_name, parse_mentions, read_jsonl, write_json
from .graph import VerifiedGraph


SEPARATOR = re.compile(r"-{2,}|\s+[-–—|/]\s*|\s*[-–—|/]\s+")
TRAILING_PARENS = re.compile(r"^(.+?)\s*\([^()]+\)\s*\.?$")


def name_views(raw_name: str) -> list[tuple[str, str]]:
    """Small, entity-agnostic set of alternative readings of one plain name."""
    views: list[tuple[str, str]] = []
    parts = [part.strip() for part in SEPARATOR.split(raw_name) if part.strip()]
    if len(parts) > 1:
        views.extend((("first_segment", parts[0]), ("last_segment", parts[-1])))
    parenthetical = TRAILING_PARENS.match(raw_name.strip())
    if parenthetical:
        views.append(("without_parenthetical", parenthetical.group(1).strip()))
    tokens = normalize_name(raw_name).split()
    if len(tokens) == 2 and all(3 <= len(token) <= 12 for token in tokens):
        views.append(("compact_spacing", compact_name(raw_name)))
    if any(len(token) >= 6 and token.endswith("ies") for token in tokens):
        views.append(("plural_ies", " ".join(token[:-3] + "y" if len(token) >= 6 and token.endswith("ies") else token for token in tokens)))
    original = normalize_name(raw_name)
    seen = {original}
    result = []
    for kind, text in views:
        normalized = normalize_name(text)
        if len(normalized) < 3 or normalized in seen:
            continue
        seen.add(normalized)
        result.append((kind, text))
        if len(result) == 3:  # Bounded per-row experimental cost.
            break
    return result


def _pair_cosines(proposals: list[Any], vectorizer: Any) -> np.ndarray:
    if vectorizer is None or not proposals:
        return np.zeros(len(proposals), dtype=np.float32)
    left = vectorizer.transform([item.mention_text for item in proposals])
    right = vectorizer.transform([item.matched_name for item in proposals])
    return np.asarray(left.multiply(right).sum(axis=1)).ravel().astype(np.float32)


def _rescore_against_full_name(proposals: list[Any], scorer: Any, char_vectorizer: Any, word_vectorizer: Any) -> None:
    """R1/S0 isolates retrieval: a new root is scored against the original full name."""
    for start in range(0, len(proposals), 5_000):
        batch = proposals[start:start + 5_000]
        char_scores = _pair_cosines(batch, char_vectorizer)
        word_scores = _pair_cosines(batch, word_vectorizer)
        for item, char_score, word_score in zip(batch, char_scores, word_scores):
            item.char_tfidf_score = float(char_score)
            item.word_tfidf_score = float(word_score)
            item.lexical_score = max(float(char_score), float(word_score))
            item.rrf_score = 0.0  # Variant rank must not masquerade as full-name evidence.
            item.embedding_score = None
            item.exact = float(normalize_name(item.mention_text) == normalize_name(item.matched_name))
        scorer.score_proposals(batch)


def _apply_rules_cutoff(decisions: list[Any], scorer: Any, config: dict[str, Any], party_job: dict, default_job: dict) -> None:
    floor = float(config.get("decision", {}).get("rules_containment_min_confidence", 0.40))
    for decision in decisions:
        job = party_job.get(str(decision.matched_member_id or decision.verified_party_id), default_job)
        if decision.decision_tier == "RULES_UNIQUE_CONTAINMENT":
            cutoff = min(scorer.plain_threshold, floor)
        elif decision.connector is None and decision.connector_resolution is None:
            cutoff = scorer.plain_threshold
        else:
            cutoff = scorer.system_threshold
        cutoff = max(cutoff, float(job.get("confidenceCutoff", 0.0)))
        if decision.decision == "MATCH" and decision.confidence < cutoff:
            decision.decision = "NO_MATCH"
            decision.reason = "BELOW_REQUEST_CUTOFF"


def _competing_view_guard(decisions: list[Any], views_by_id: dict[str, list[Any]]) -> int:
    """A distinct strong organization found in another view prevents partial-name acceptance."""
    vetoes = 0
    for decision in decisions:
        if decision.decision != "MATCH" or decision.matched_mention == decision.raw_name:
            continue
        competing = [
            item for item in views_by_id.get(decision.adm_party_id, [])
            if item.root_party_id != decision.verified_party_id
            and item.feature_score >= 0.80
            and item.mention_text != decision.matched_mention
        ]
        if competing:
            decision.decision = "NO_MATCH"
            decision.reason = "SHADOW_COMPETING_VIEW"
            vetoes += 1
    return vetoes


def _metrics(decisions: dict[str, Any], labels: dict[str, dict], graph: VerifiedGraph, ids: list[str]) -> dict[str, Any]:
    matches = correct = 0
    for adm_id in ids:
        decision = decisions.get(adm_id)
        if decision is None or decision.decision != "MATCH":
            continue
        matches += 1
        expected = str(labels[adm_id]["expected_party_id"])
        root = graph.root_id(expected) if expected in graph.nodes else expected
        correct += decision.verified_party_id == root
    precision = correct / matches if matches else None
    recall = correct / len(ids) if ids else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {"rows": len(ids), "matches": matches, "correct": correct,
            "wrong": matches - correct, "precision": precision, "recall": recall, "f1": f1}


def run_shadow_name_views(
    output: Path, labels_path: Path, matchable: list[PartyRecord],
    baseline_proposals: dict[str, list[Any]], baseline_decisions: dict[str, Any],
    graph: VerifiedGraph, retriever: Any, scorer: Any, config: dict[str, Any],
    party_job: dict[str, dict], jobs: list[dict],
) -> dict[str, Any]:
    """Run a bounded 2x2 rules-only experiment; write no normal output files."""
    from .matching import CrossEncoderReranker, MentionRetriever, collect_proposals, decide_records

    labels = {str(row["adm_party_id"]): row for row in read_jsonl(labels_path)
              if row.get("scorable") and row.get("split") in {"CALIBRATION", "TEST_KNOWN", "TEST_UNSEEN"}}
    records = [record for record in matchable if record.adm_party_id in labels
               and len(parse_mentions(record)) == 1 and not parse_mentions(record)[0].parse_warning]
    views_by_id: dict[str, list[tuple[str, str]]] = {}
    view_records: list[PartyRecord] = []
    view_owner: dict[str, tuple[str, str]] = {}
    for record in records:
        views = name_views(record.raw_name)
        views_by_id[record.adm_party_id] = views
        for index, (kind, text) in enumerate(views):
            view_id = f"{record.adm_party_id}:shadow_view:{index}"
            view_owner[view_id] = (record.adm_party_id, kind)
            view_records.append(PartyRecord(record.account_id, view_id, text, record.source_row))

    # Normal decisions are already complete. Retain only evaluated proposals
    # and release the large full-run index before building the shadow index.
    original = {record.adm_party_id: baseline_proposals.get(record.adm_party_id, []) for record in records}
    baseline_proposals.clear()
    char_vectorizer, word_vectorizer = retriever.char_vectorizer, retriever.word_vectorizer
    retriever.char_matrix = retriever.word_matrix = None
    retriever.embedding_matrix = retriever.ann_index = retriever.embedding_model = None
    retriever.mentions.clear()
    retriever.records.clear()
    retriever.exact.clear()
    gc.collect()

    variant_by_id: dict[str, list[Any]] = defaultdict(list)
    retrieval_stats: dict[str, Any] = {"view_records": len(view_records), "view_rows": sum(bool(v) for v in views_by_id.values())}
    if view_records:
        # Reuse the full-run TF-IDF vocabulary/IDF. Fitting on held-out names
        # would leak their distribution into this experiment and change scores.
        shadow_cfg = {**config.get("retrieval", {}), "char_tfidf_enabled": False,
                      "word_tfidf_enabled": False, "embedding_enabled": False,
                      "max_roots_per_mention": 20, "max_variants_per_root": 1}
        view_retriever = MentionRetriever(view_records, shadow_cfg, workers=1)
        view_retriever.char_vectorizer = char_vectorizer
        view_retriever.word_vectorizer = word_vectorizer
        texts = [mention.text for mention in view_retriever.mentions]
        if char_vectorizer is not None:
            view_retriever.char_matrix = char_vectorizer.transform(texts)
        if word_vectorizer is not None:
            view_retriever.word_matrix = word_vectorizer.transform(texts)
        candidate_started = time.perf_counter()
        by_view, raw_stats = collect_proposals(graph, view_retriever, scorer)
        retrieval_stats.update({"view_retrieval_seconds": time.perf_counter() - candidate_started,
                                "view_scored_proposals": raw_stats["scored_proposals"],
                                "view_root_cap_hits": raw_stats["root_cap_hits"]})
        for view_id, proposals in by_view.items():
            owner_id, kind = view_owner[view_id]
            for proposal in proposals:
                proposal.adm_party_id = owner_id
                proposal.retrieval_sources.append(f"shadow_view:{kind}")
                variant_by_id[owner_id].append(proposal)
        del by_view, view_retriever
        gc.collect()

    r1s0: dict[str, list[Any]] = {}
    r0s1: dict[str, list[Any]] = {}
    r1s1: dict[str, list[Any]] = {}
    new_full: list[Any] = []
    new_roots_count = 0
    for record in records:
        adm_id = record.adm_party_id
        base = original[adm_id]
        base_roots = {item.root_party_id for item in base}
        best_views: dict[str, Any] = {}
        for item in variant_by_id.get(adm_id, []):
            previous = best_views.get(item.root_party_id)
            if previous is None or (item.feature_score, item.matched_name) > (previous.feature_score, previous.matched_name):
                best_views[item.root_party_id] = item
        old_root_views = [item for root, item in best_views.items() if root in base_roots]
        new_root_views = sorted(
            (item for root, item in best_views.items() if root not in base_roots),
            key=lambda item: (-item.feature_score, item.root_party_id),
        )[:10]
        new_roots_count += len(new_root_views)
        r0s1[adm_id] = [*base, *old_root_views]
        r1s1[adm_id] = [*base, *old_root_views, *new_root_views]
        full_candidates = []
        for item in new_root_views:
            full = copy.copy(item)
            full.mention_text = record.raw_name
            full.mention_id = f"{adm_id}:m0"
            full_candidates.append(full)
            new_full.append(full)
        r1s0[adm_id] = [*base, *full_candidates]
    del variant_by_id
    _rescore_against_full_name(new_full, scorer, char_vectorizer, word_vectorizer)

    reranker = CrossEncoderReranker(Path("unused-shadow-cross-encoder"), "off")
    alternative: dict[str, dict[str, Any]] = {}
    vetoes = {}
    for arm, proposal_map in (("retrieval_only", r1s0), ("decision_only", r0s1), ("combined", r1s1)):
        chosen = decide_records(records, proposal_map, graph, scorer, reranker, config)
        _apply_rules_cutoff(chosen, scorer, config, party_job, jobs[0])
        vetoes[arm] = _competing_view_guard(chosen, proposal_map) if arm != "retrieval_only" else 0
        alternative[arm] = {item.adm_party_id: item for item in chosen}

    arms = {"baseline": baseline_decisions, **alternative}
    split_ids = {
        "calibration_plain": [item.adm_party_id for item in records if labels[item.adm_party_id]["split"] == "CALIBRATION"],
        "test_known_plain": [item.adm_party_id for item in records if labels[item.adm_party_id]["split"] == "TEST_KNOWN"],
        "test_unseen_plain": [item.adm_party_id for item in records if labels[item.adm_party_id]["split"] == "TEST_UNSEEN"],
        "held_out_plain": [item.adm_party_id for item in records if labels[item.adm_party_id]["split"].startswith("TEST_")],
        "held_out_all": [item.adm_party_id for item in matchable
                         if item.adm_party_id in labels and labels[item.adm_party_id]["split"].startswith("TEST_")],
    }
    # Connector decisions are deliberately unchanged in the overall projection.
    metrics = {arm: {split: _metrics(ChainMap(decisions, baseline_decisions), labels, graph, ids)
                     for split, ids in split_ids.items()}
               for arm, decisions in arms.items()}
    comparisons: dict[str, Any] = {}
    held_ids = split_ids["held_out_plain"]
    for arm in ("retrieval_only", "decision_only", "combined"):
        gained = lost = newly_wrong = corrected_wrong = 0
        for adm_id in held_ids:
            expected = str(labels[adm_id]["expected_party_id"])
            root = graph.root_id(expected) if expected in graph.nodes else expected
            before, after = baseline_decisions[adm_id], alternative[arm][adm_id]
            old_ok = before.decision == "MATCH" and before.verified_party_id == root
            new_ok = after.decision == "MATCH" and after.verified_party_id == root
            gained += new_ok and not old_ok
            lost += old_ok and not new_ok
            newly_wrong += after.decision == "MATCH" and after.verified_party_id != root and before.decision != "MATCH"
            corrected_wrong += before.decision == "MATCH" and before.verified_party_id != root and after.decision != "MATCH"
        comparisons[arm] = {"new_correct": gained, "lost_correct": lost,
                            "new_wrong_from_abstain": newly_wrong, "wrong_to_abstain": corrected_wrong,
                            "competing_view_vetoes": vetoes[arm]}

    summary = {"experiment": "temporary_name_views_2x2", "scope": "plain names changed; connectors held fixed",
               "label_usage": "Labels select audited rows and measure outcomes; target labels never enter views, retrieval, scoring, or decisions.",
               "retrieval": retrieval_stats, "new_roots_considered": new_roots_count,
               "metrics": metrics, "comparisons": comparisons,
               "limitations": ["OBO/VIA and unknown rows are not compared.",
                               "Shadow view retrieval is lexical only, even if embeddings are enabled.",
                               "At most 20 roots per view and 10 new roots per row are retained.",
                               "Overall held-out metrics carry forward unchanged connector decisions."]}
    write_json(output / "shadow_name_views.json", summary)
    case_path = output / "shadow_name_views_changes.csv"
    with case_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "raw_name", "expected_name", "views", "baseline", "retrieval_only", "decision_only", "combined", "adm_party_id"])
        for record in records:
            adm_id = record.adm_party_id
            choices = [arms[arm][adm_id] for arm in ("baseline", "retrieval_only", "decision_only", "combined")]
            outcomes = [(item.decision, item.verified_party_id if item.decision == "MATCH" else None) for item in choices]
            if len(set(outcomes)) == 1:
                continue
            expected = str(labels[adm_id]["expected_party_id"])
            expected_root = graph.root_id(expected) if expected in graph.nodes else expected
            writer.writerow([labels[adm_id]["split"], record.raw_name, labels[adm_id].get("expected_canonical_name"),
                             " | ".join(f"{kind}:{text}" for kind, text in views_by_id[adm_id]),
                             *(f"{'CORRECT' if item.verified_party_id == expected_root else 'WRONG'}:{item.verified_party_name} ({item.confidence:.3f})"
                               if item.decision == "MATCH" else f"NO_MATCH:{item.reason} ({item.confidence:.3f})"
                               for item in choices), adm_id])
    report = ["# Temporary shadow name-view experiment", "",
              "All arms share the same verified graph and trained rules artifact. Only plain labeled rows are compared.",
              "Normal ML decisions, rules comparison, events, and graph updates are unchanged.",
              "Overall held-out metrics retain baseline OBO/VIA decisions; only plain names change.", "",
              "| Arm | Retrieval | Decision | Calibration plain F1 | Held-out plain precision | Held-out plain recall | Held-out plain F1 | Overall held-out recall | Correct plain | Wrong plain |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, retrieval, decision_label in (("baseline", "current", "current"),
                                            ("retrieval_only", "name views", "current full-name scoring"),
                                            ("decision_only", "current roots", "view-aware scoring + conflict guard"),
                                            ("combined", "name views", "view-aware scoring + conflict guard")):
        cal = metrics[arm]["calibration_plain"]
        test = metrics[arm]["held_out_plain"]
        fmt = lambda value: f"{value:.1%}" if value is not None else "n.a."
        overall = metrics[arm]["held_out_all"]
        report.append(f"| {arm} | {retrieval} | {decision_label} | {fmt(cal['f1'])} | {fmt(test['precision'])} | {fmt(test['recall'])} | {fmt(test['f1'])} | {fmt(overall['recall'])} | {test['correct']:,} | {test['wrong']:,} |")
    report += ["", "## Held-out changes versus baseline", "",
               "| Arm | Newly correct | Lost correct | New wrong from abstain | Wrong to abstain | Competing-view vetoes |",
               "|---|---:|---:|---:|---:|---:|"]
    for arm, values in comparisons.items():
        report.append(f"| {arm} | {values['new_correct']:,} | {values['lost_correct']:,} | {values['new_wrong_from_abstain']:,} | {values['wrong_to_abstain']:,} | {values['competing_view_vetoes']:,} |")
    report += ["", "Candidate views are hypotheses, not verified aliases. A suffix can name another organization;",
               "review the CSV before adopting any arm. Choose/tune on calibration, then treat held-out results as validation.", ""]
    (output / "shadow_name_views.md").write_text("\n".join(report), encoding="utf-8")
    return {"path": str(output / "shadow_name_views.md"), "view_records": len(view_records),
            "new_roots_considered": new_roots_count,
            "held_out_plain": {arm: values["held_out_plain"] for arm, values in metrics.items()},
            "held_out_all": {arm: values["held_out_all"] for arm, values in metrics.items()}}
