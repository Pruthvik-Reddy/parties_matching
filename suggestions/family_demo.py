"""Two read-only, full-input suggestion snapshots for verified parties.

Initial decisions are independent: each party is the only verified party in
its decision graph. Final decisions use the complete catalog. Both
search the same full eligible unverified population; neither uses the old
leading-token input filter, emits events, or saves graph changes. New audit
labels are read after decisions. The reused POC matcher retains its existing
TRAIN/CALIBRATION-based multipart prior when configured.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from party_matching.domain import load_config, read_json, read_jsonl
from party_matching.graph import VerifiedGraph
from party_matching.matching import FeatureScorer, MentionRetriever, _best_per_root, _guard_reason

from .engine import _GENERIC_ANCHORS, name_tokens
from .poc import _current_graph, _eligible_records, _record_case, _rules_decisions, _write_csv
from .simple_workbook import export_family_snapshot


# These are candidate-presentation floors, not changes to the POC's MATCH
# cutoffs. A shared brand is review evidence, never proof of legal identity.
REVIEW_RULE_SCORE = 0.45
REVIEW_LEXICAL_SCORE = 0.45
FAMILY_SIMILARITY_FLOOR = 0.42
FAMILY_ANCHOR_MIN_LENGTH = 5
NON_ANCHOR_LEXICAL_FLOOR = 0.65
NON_ANCHOR_NAME_SIMILARITY_FLOOR = 0.68

SUGGESTION_COLUMNS = (
    "verified_party", "unverified_party", "status", "reason", "rules_score",
    "lexical_score", "family_similarity", "retrieval_source", "matched_segment",
    "name_case", "matched_verified_party", "assigned_to", "source_row",
    "verified_party_id", "unverified_party_id",
)
MISS_COLUMNS = (
    "verified_party", "unverified_party", "expected_party", "split", "reason",
    "source_row", "verified_party_id", "unverified_party_id",
)


def _representatives(catalog: list[dict], entries: list[dict] | None) -> list[dict]:
    # The normal run uses the entire current verified catalog. The old short
    # list is retained only as an explicit, optional focus for quick reviews.
    if entries is None:
        if not catalog:
            raise ValueError("The prepared verified catalog is empty")
        return catalog
    if not entries:
        raise ValueError("The optional representatives list is empty")
    selected: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        matches = [party for party in catalog if
                   str(party.get("partyName", "")).strip().casefold() == name.casefold()]
        if len(matches) != 1:
            raise ValueError(f"Representative must uniquely name a verified party: {name!r}")
        party = matches[0]
        party_id = str(party["partyId"])
        if party_id in seen:
            raise ValueError(f"Duplicate representative: {name}")
        seen.add(party_id)
        selected.append(party)
    return selected


def _family_anchor(name: str) -> str | None:
    """One distinctive *leading* token; never a substring or generic word."""
    tokens = name_tokens(name)
    if not tokens:
        return None
    token = tokens[0]
    if len(token) < FAMILY_ANCHOR_MIN_LENGTH or token in _GENERIC_ANCHORS:
        return None
    return token


def _family_index(retriever: MentionRetriever) -> tuple[dict[str, set[str]], dict[str, str]]:
    """An inverted token index over the preferred parsed mention, built once.

    It is deliberately narrower than a fuzzy all-pairs search. In particular,
    a vendor merely mentioned in a nonpreferred OBO/VIA segment is not added.
    """
    preferred: dict[str, str] = {}
    warnings: set[str] = set()
    connector_types: dict[str, set[str]] = defaultdict(set)
    for mention in retriever.mentions:
        preferred.setdefault(mention.adm_party_id, mention.text)
        if mention.parse_warning:
            warnings.add(mention.adm_party_id)
        if mention.connector_before:
            connector_types[mention.adm_party_id].add(mention.connector_before)
    postings: dict[str, set[str]] = defaultdict(set)
    for adm_id, text in preferred.items():
        if adm_id in warnings or len(connector_types[adm_id]) > 1:
            continue
        tokens = name_tokens(text)
        if tokens:
            postings[tokens[0]].add(adm_id)
    return dict(postings), preferred


def _family_similarity(verified_name: str, mention: str) -> float:
    left, right = name_tokens(verified_name), name_tokens(mention)
    return SequenceMatcher(None, " ".join(left), " ".join(right)).ratio() if left and right else 0.0


def _independent_identity_evidence(verified_name: str, mention: str, best) -> bool:
    """A one-party run must not be strong merely because no rival was present.

    A complete verified core at the start of the relevant mention, or an exact
    verified alias, is independent name evidence. A shared brand with a
    different business word remains review-only even if the POC's one-root
    decision happened to accept it.
    """
    raw, verified = name_tokens(mention), name_tokens(verified_name)
    full_core_prefix = (bool(raw and verified) and raw[:len(verified)] == verified
                        and (len(verified) > 1 or _family_anchor(verified_name) is not None))
    return full_core_prefix or bool(best and best.exact)


def _qualified_proposal(item, preferred_mention: str) -> bool:
    """Exclude incidental fuzzy index hits from the reader-facing pair pool."""
    if item.mention_text != preferred_mention:
        return False
    if item.exact:
        return True
    lexical = max(item.char_tfidf_score, item.word_tfidf_score)
    similarity = _family_similarity(item.matched_name, preferred_mention)
    return (lexical >= NON_ANCHOR_LEXICAL_FLOOR
            and similarity >= NON_ANCHOR_NAME_SIMILARITY_FLOOR)


def _label_index(prepared: Path, eligible_ids: set[str]) -> dict[str, dict]:
    path = prepared / "labels.jsonl"
    if not path.exists():
        return {}
    return {str(label["adm_party_id"]): label for label in read_jsonl(path)
            if label.get("scorable") and str(label.get("adm_party_id")) in eligible_ids}


def _one_party_graph(full_graph: VerifiedGraph, party: dict) -> VerifiedGraph:
    """Keep this party's known aliases, but not its catalog competitors."""
    party_id = str(party["partyId"])
    graph = VerifiedGraph(full_graph.account_id, full_graph.path, load_existing=False)
    graph.add_parties([party])
    graph.nodes[party_id].candidates = list(full_graph.nodes[party_id].candidates)
    graph._rebuild_indexes()
    return graph


def _proposals_by_owner(proposals: dict[str, list], full_graph: VerifiedGraph,
                        representatives: list[dict]) -> dict[str, dict[str, list]]:
    """Partition one bounded full-catalog retrieval by the verified owner."""
    selected = {str(party["partyId"]) for party in representatives}
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for adm_id, items in proposals.items():
        for item in items:
            owner_id = item.owner_party_id
            if owner_id not in selected:
                continue
            isolated = copy.copy(item)
            isolated.root_party_id = owner_id
            isolated.root_party_name = full_graph.nodes[owner_id].party_name
            # A collision among *other* verified parties does not exist in an
            # independent initial run. Its final run still sees that collision.
            isolated.candidate_collision_count = 1
            grouped[owner_id][adm_id].append(isolated)
    return {party_id: dict(rows) for party_id, rows in grouped.items()}


def _run_stage(stage: str, records: list, prepared: Path, config: dict, jobs: list[dict],
               representatives: list[dict],
               retriever: MentionRetriever, scorer: FeatureScorer,
               postings: dict[str, set[str]], preferred: dict[str, str],
               labels: dict[str, dict], full_graph,
               global_result: tuple, initial_by_owner: dict[str, dict[str, list]]) -> dict:
    started = time.perf_counter()
    records_by_id = {record.adm_party_id: record for record in records}
    strong: list[dict] = []
    review: list[dict] = []
    dropped: list[dict] = []
    misses: list[dict] = []
    per_party: list[dict] = []
    stage_details: list[dict] = []
    stage_run_count = 0
    expected_by_root: dict[str, set[str]] = defaultdict(set)
    for adm_id, label in labels.items():
        expected = str(label.get("expected_party_id") or "")
        if expected in full_graph.nodes:
            expected_by_root[full_graph.root_id(expected)].add(adm_id)
    selected_by_id = {str(party["partyId"]): party for party in representatives}
    display_by_root: dict[str, str] = {}
    for party in representatives:
        party_id = str(party["partyId"])
        root_id = full_graph.root_id(party_id)
        display_by_root.setdefault(root_id, party_id)
        if root_id in selected_by_id:
            display_by_root[root_id] = root_id

    # Initial graphs contain *one* party each. The final graph is created once
    # with all verified parties, so final decisions can resolve competition.
    runs = ((party, _one_party_graph(full_graph, party)) for party in representatives) \
        if stage == "initial" else ((None, full_graph),)
    for run_party, graph in runs:
        family_ids_for_run = (postings.get(_family_anchor(str(run_party["partyName"])), set())
                              if run_party is not None else None)
        if run_party is None:
            decisions, proposals, _, stages = global_result
        else:
            party_id = str(run_party["partyId"])
            proposals_for_party = initial_by_owner.get(party_id, {})
            candidate_record_ids = set(proposals_for_party) | (family_ids_for_run or set())
            candidate_records = [records_by_id[adm_id] for adm_id in sorted(candidate_record_ids)]
            # A one-root catalog has no competing fragment roots, so the
            # multipart chooser cannot alter a result. Skip its repeated
            # TRAIN/CALIBRATION label scan across thousands of parties.
            single_config = {**config, "decision": {**config.get("decision", {}),
                                                    "rules_multipart_enabled": False}}
            if candidate_records:
                decisions, proposals, _, stages = _rules_decisions(
                    candidate_records, graph, single_config, prepared, jobs,
                    retriever=retriever, scorer=scorer,
                    proposals_override=proposals_for_party)
            else:
                decisions, proposals, stages = [], {}, {"baseline": 0, "retrieval": {"reused_global_proposals": True}}
        decision_by_id = {decision.adm_party_id: decision for decision in decisions}
        stage_run_count += 1
        if len(stage_details) < 20:
            stage_details.append(stages)
        targets = [run_party] if run_party is not None else representatives
        by_root: dict[str, set[str]] = defaultdict(set)
        accepted_by_root: dict[str, set[str]] = defaultdict(set)
        target_roots = {graph.root_id(str(party["partyId"])) for party in targets}
        for adm_id, items in proposals.items():
            for root in {item.root_party_id for item in items if item.root_party_id in target_roots}:
                by_root[root].add(adm_id)
        for adm_id, decision in decision_by_id.items():
            if (decision.decision == "MATCH" and decision.verified_party_id in target_roots
                    and (_record_case(records_by_id[adm_id]) == "PLAIN"
                         or decision.matched_mention == preferred[adm_id])):
                accepted_by_root[decision.verified_party_id].add(adm_id)
        for party in targets:
            party_id = str(party["partyId"])
            party_name = str(party["partyName"])
            root_id = graph.root_id(party_id)
            if stage == "final" and display_by_root[full_graph.root_id(party_id)] != party_id:
                # The final rules decision names a graph root. Show that root
                # once, while retaining a visible zero-count row for each
                # child verified party in the full catalog.
                per_party.append({
                    "verified_party": party_name, "verified_party_id": party_id,
                    "rolled_into": full_graph.nodes[full_graph.root_id(party_id)].party_name,
                    "eligible_unverified": len(records), "indexed_proposals": 0,
                    "qualified_index_proposals": 0, "leading_name_hits": 0,
                    "candidate_pairs": 0, "strong": 0, "review": 0, "dropped": 0,
                    "known_labels": 0, "known_label_misses": 0,
                    "known_label_index_misses": 0, "known_label_gate_exclusions": 0,
                    "held_out_labels": 0, "held_out_candidate_recall": None,
                    "held_out_strong_precision": None, "held_out_strong_recall": None,
                })
                continue
            anchor = _family_anchor(party_name)
            family_ids = postings.get(anchor, set()) if anchor else set()
            proposal_ids = by_root.get(root_id, set())
            qualified_proposal_ids = {
                adm_id for adm_id in proposal_ids
                if any(_qualified_proposal(item, preferred[adm_id]) for item in
                       proposals.get(adm_id, []) if item.root_party_id == root_id)
            }
            accepted_ids = accepted_by_root.get(root_id, set())
            candidate_ids = qualified_proposal_ids | family_ids | accepted_ids
            pair_status: dict[str, str] = {}
            labeled_strong_ids: list[str] = []
            for adm_id in sorted(candidate_ids):
                record = records_by_id[adm_id]
                decision = decision_by_id.get(adm_id)
                relevant = [item for item in proposals.get(adm_id, [])
                            if item.root_party_id == root_id
                            and item.mention_text == preferred[adm_id]]
                best = _best_per_root(relevant).get(root_id) if relevant else None
                guard = _guard_reason(best, config.get("matching", {})) if best else None
                lexical = max(best.char_tfidf_score, best.word_tfidf_score, best.exact) if best else 0.0
                family_similarity = _family_similarity(party_name, preferred[adm_id]) if adm_id in family_ids else 0.0
                assigned_elsewhere = (stage == "final" and decision is not None
                                      and decision.decision == "MATCH"
                                      and decision.verified_party_id != root_id)
                independent_support = _independent_identity_evidence(party_name, preferred[adm_id], best)
                if adm_id in accepted_ids and (stage == "final" or independent_support):
                    status, reason = "STRONG", decision.match_method or "MATCH"
                elif assigned_elsewhere:
                    status, reason = "DROPPED", "ASSIGNED_TO_OTHER_PARTY"
                elif guard:
                    status, reason = "DROPPED", guard
                elif adm_id in accepted_ids and not independent_support:
                    status, reason = "REVIEW", "SINGLE_PARTY_IDENTITY_UNCONFIRMED"
                elif best and best.rules_score >= REVIEW_RULE_SCORE and lexical >= REVIEW_LEXICAL_SCORE:
                    status, reason = "REVIEW", decision.reason if decision else "BELOW_STRONG_RULES"
                elif adm_id in family_ids and family_similarity >= FAMILY_SIMILARITY_FLOOR:
                    status, reason = "REVIEW", "SHARED_LEADING_NAME_ONLY"
                else:
                    status, reason = "DROPPED", "BELOW_REVIEW_EVIDENCE"
                row = {
                    "verified_party": party_name, "unverified_party": record.raw_name,
                    "status": status, "reason": reason,
                    "rules_score": round(best.rules_score, 6) if best else None,
                    "lexical_score": round(lexical, 6) if best else None,
                    "family_similarity": round(family_similarity, 6) if adm_id in family_ids else None,
                    "retrieval_source": ("POC_INDEX+LEADING_NAME_INDEX" if best and adm_id in family_ids
                                         else "POC_INDEX" if best else "LEADING_NAME_INDEX"),
                    "matched_segment": (best.mention_text if best else preferred[adm_id]),
                    "name_case": _record_case(record),
                    "matched_verified_party": (decision.verified_party_name if decision is not None
                                               and decision.decision == "MATCH" else ""),
                    "assigned_to": decision.verified_party_name if assigned_elsewhere else "",
                    "source_row": record.source_row,
                    "verified_party_id": party_id, "unverified_party_id": adm_id,
                }
                {"STRONG": strong, "REVIEW": review, "DROPPED": dropped}[status].append(row)
                pair_status[adm_id] = status
                if (status == "STRONG" and adm_id in labels
                        and labels[adm_id].get("split") in {"TEST_KNOWN", "TEST_UNSEEN"}):
                    labeled_strong_ids.append(adm_id)

            # This label audit runs *after* decisions and covers exact
            # party/root associations, not inferred brand families. The POC's
            # existing multipart prior is a separate TRAIN/CALIBRATION input.
            expected_ids = expected_by_root.get(full_graph.root_id(party_id), set())
            index_miss_ids = expected_ids - (proposal_ids | family_ids)
            gate_miss_ids = (expected_ids - candidate_ids) - index_miss_ids
            for adm_id in sorted(expected_ids - candidate_ids):
                label = labels[adm_id]
                record = records_by_id[adm_id]
                gap_reason = ("NOT_RETRIEVED_FOR_PARTY" if adm_id in index_miss_ids
                              else "INDEXED_BUT_NOT_FAMILY_QUALIFIED")
                misses.append({
                    "verified_party": party_name, "unverified_party": record.raw_name,
                    "expected_party": label.get("expected_canonical_name") or "",
                    "split": label.get("split") or "", "reason": gap_reason,
                    "source_row": record.source_row, "verified_party_id": party_id,
                    "unverified_party_id": adm_id,
                })
            held_out = {adm_id for adm_id in expected_ids if labels[adm_id].get("split") in
                        {"TEST_KNOWN", "TEST_UNSEEN"}}
            correct_strong = sum(adm_id in held_out for adm_id in labeled_strong_ids)
            per_party.append({
                "verified_party": party_name, "verified_party_id": party_id,
                "rolled_into": "",
                "eligible_unverified": len(records),
                "indexed_proposals": len(proposal_ids),
                "qualified_index_proposals": len(qualified_proposal_ids),
                "leading_name_hits": len(family_ids),
                "candidate_pairs": len(candidate_ids),
                "strong": sum(value == "STRONG" for value in pair_status.values()),
                "review": sum(value == "REVIEW" for value in pair_status.values()),
                "dropped": sum(value == "DROPPED" for value in pair_status.values()),
                "known_labels": len(expected_ids),
                "known_label_misses": len(expected_ids - candidate_ids),
                "known_label_index_misses": len(index_miss_ids),
                "known_label_gate_exclusions": len(gate_miss_ids),
                "held_out_labels": len(held_out),
                "held_out_candidate_recall": len(held_out & candidate_ids) / len(held_out) if held_out else None,
                "held_out_strong_precision": correct_strong / len(labeled_strong_ids) if labeled_strong_ids else None,
                "held_out_strong_recall": correct_strong / len(held_out) if held_out else None,
            })

    ordering = lambda row: (row["verified_party"].casefold(), row["unverified_party"].casefold(),
                            row["unverified_party_id"])
    for rows in (strong, review, dropped, misses):
        rows.sort(key=ordering)
    return {
        "stage": stage, "strong": strong, "review": review, "dropped": dropped,
        "misses": misses, "parties": per_party,
        "metrics": {
            "scope": "all eligible unverified rows; no leading-token input filter",
            "label_scope": "exact verified party or known graph root; other brand relatives not inferred",
            "selected_verified_parties": len(representatives),
            "eligible_unverified_rows": len(records),
            "candidate_pairs": sum(item["candidate_pairs"] for item in per_party),
            "strong_pairs": len(strong), "review_pairs": len(review),
            "dropped_pairs": len(dropped), "known_label_misses": len(misses),
            "known_label_index_misses": sum(item["known_label_index_misses"] for item in per_party),
            "known_label_gate_exclusions": sum(item["known_label_gate_exclusions"] for item in per_party),
            "drop_reasons": dict(Counter(row["reason"] for row in dropped)),
            "per_verified_party": per_party,
            "rules_stages": stage_details,
            "rules_stage_runs": stage_run_count,
            "rules_stages_sampled": stage_run_count > len(stage_details),
            "stage_seconds": time.perf_counter() - started,
        },
    }


def build_family_demo(prepared: Path, config: dict,
                      representative_entries: list[dict] | None = None) -> tuple[dict, dict, dict]:
    started = time.perf_counter()
    catalog = read_json(prepared / "verified_parties.json", []) or []
    representatives = _representatives(catalog, representative_entries)
    jobs = read_json(prepared / "jobs.json", []) or []
    if not jobs:
        raise ValueError("Prepared jobs.json is missing; run scripts/prepare.py first")
    records, excluded = _eligible_records(prepared)
    if not records:
        raise ValueError("No eligible unverified rows in prepared data")
    account_id = str(jobs[0]["accountId"])
    if any(str(job["accountId"]) != account_id for job in jobs):
        raise ValueError("The demo requires one account")
    index_started = time.perf_counter()
    execution = config.get("execution", {})
    workers = 1 if execution.get("mode") == "serial" else max(1, int(execution.get("workers", 10)))
    retriever = MentionRetriever(records, config.get("retrieval", {}), workers=workers)
    postings, preferred = _family_index(retriever)
    index_seconds = time.perf_counter() - index_started
    artifact = Path(config.get("paths", {}).get("artifacts_dir", "artifacts")) / "matcher.joblib"
    scorer = FeatureScorer(artifact)
    full_graph = _current_graph(prepared, config, account_id, [], catalog)
    labels = _label_index(prepared, {record.adm_party_id for record in records})
    # Retrieve and score the full catalog once. Reusing its bounded proposals
    # makes an all-verified independent snapshot feasible; an initial party
    # still has its own one-root graph and cannot lose to another party at the
    # decision stage. The leading-name index catches additional review leads.
    global_result = _rules_decisions(records, full_graph, config, prepared, jobs,
                                     retriever=retriever, scorer=scorer)
    initial_by_owner = _proposals_by_owner(global_result[1], full_graph, representatives)
    initial = _run_stage("initial", records, prepared, config, jobs, representatives,
                         retriever, scorer, postings, preferred, labels, full_graph,
                         global_result, initial_by_owner)
    final = _run_stage("final", records, prepared, config, jobs, representatives,
                       retriever, scorer, postings, preferred, labels, full_graph,
                       global_result, initial_by_owner)
    common = {
        "index_seconds": index_seconds, "total_seconds_before_export": time.perf_counter() - started,
        "indexed_mentions": len(retriever.mentions), "eligible_unverified_rows": len(records),
        "full_catalog_retrieval": global_result[3].get("retrieval", {}),
        "excluded_rows": dict(excluded), "verified_party_count": len(representatives),
        "verified_selection": "all_prepared" if representative_entries is None else "named_subset",
        "representatives": ([p["partyName"] for p in representatives]
                            if representative_entries is not None else None),
        "note": "Initial decisions isolate each party using one full-catalog indexed retrieval; "
                "pair counts may overlap. Final uses the full verified catalog.",
    }
    return initial, final, common


def _save_stage(folder: Path, result: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for key, filename, columns in (
        ("strong", "strong_suggestions.csv", SUGGESTION_COLUMNS),
        ("review", "review_candidates.csv", SUGGESTION_COLUMNS),
        ("dropped", "dropped_candidates.csv", SUGGESTION_COLUMNS),
        ("misses", "known_label_gaps.csv", MISS_COLUMNS),
    ):
        _write_csv(folder / filename, result[key], columns)
    export_started = time.perf_counter()
    export_family_snapshot(folder, result)
    result["metrics"]["workbook_export_seconds"] = time.perf_counter() - export_started
    (folder / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--representatives-file", type=Path,
                        help="Optional named subset for smaller output; omitted means every prepared verified party")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty; existing results are preserved")
    config = load_config(args.config)
    prepared = args.prepared_dir or Path(config.get("paths", {}).get("prepared_dir", "data/prepared"))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, str(config.get("execution", {}).get("native_threads", 10)))
    try:
        entries = None
        if args.representatives_file is not None:
            entries = json.loads(args.representatives_file.read_text(encoding="utf-8"))
            if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
                raise ValueError("--representatives-file must contain a JSON list of {name} objects")
        initial, final, common = build_family_demo(prepared, config, entries)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_stage(args.output_dir / "initial", initial)
    _save_stage(args.output_dir / "final", final)
    common["total_seconds_including_export"] = time.perf_counter() - run_started
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(common, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"All {common['verified_party_count']:,} selected verified parties and "
          f"{common['eligible_unverified_rows']:,} eligible unverified rows; no quick-demo filter")
    print(f"Initial: {len(initial['strong']):,} strong, {len(initial['review']):,} review")
    print(f"Final: {len(final['strong']):,} strong, {len(final['review']):,} review")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
