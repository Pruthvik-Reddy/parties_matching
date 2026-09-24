import json
import tempfile
import unittest
from pathlib import Path

from party_matching.diagnostics import write_rules_candidate_trace
from party_matching.domain import FinalDecision, MatchProposal, PartyRecord, read_jsonl
from party_matching.graph import VerifiedGraph


class IdentityScorer:
    plain_threshold = 0.80

    def score_target_vectors(self, vectors):
        return [vector[0] for vector in vectors]


class RulesDiagnosticTests(unittest.TestCase):
    def test_failure_buckets_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            graph = VerifiedGraph("account", folder / "graph.json", load_existing=False)
            graph.add_parties([
                {"partyId": "expected", "partyName": "Expected Party"},
                {"partyId": "other", "partyName": "Other Party"},
            ])
            cases = [
                ("missing", "RETRIEVAL_MISS", [("other", 0.9, False)]),
                ("guarded", "EXPECTED_GUARDED", [("expected", 0.9, True)]),
                ("ranked", "RANKING_LOSS", [("expected", 0.7, False), ("other", 0.9, False)]),
                ("support", "SUPPORT_REJECTION", [("expected", 0.6, False)]),
            ]
            records = [PartyRecord("account", case_id, case_id, index)
                       for index, (case_id, _, _) in enumerate(cases, 1)]
            labels_path = folder / "labels.jsonl"
            labels_path.write_text("".join(
                json.dumps({"adm_party_id": case_id, "expected_party_id": "expected",
                            "expected_canonical_name": "Expected Party", "split": "TEST_KNOWN", "scorable": True}) + "\n"
                for case_id, _, _ in cases
            ), encoding="utf-8")
            proposals = {}
            decisions = {}
            for case_id, _, roots in cases:
                proposals[case_id] = [MatchProposal(
                    adm_party_id=case_id, mention_id=f"{case_id}:m0", mention_text=case_id,
                    connector_before=None, connector_after=None,
                    owner_party_id=root_id, owner_party_name=root_id,
                    root_party_id=root_id, root_party_name=root_id, matched_name=root_id,
                    candidate_type="official", candidate_source="official",
                    candidate_confidence=None, feature_score=score,
                    distinctive_token_conflict=guard,
                ) for root_id, score, guard in roots]
                decisions[case_id] = FinalDecision(
                    "account", case_id, case_id, "NO_MATCH", 0.6, "INSUFFICIENT_SUPPORT",
                    verified_party_id=roots[0][0],
                )
            trace_path = folder / "trace.jsonl"
            result = write_rules_candidate_trace(
                trace_path, records, proposals, decisions, labels_path, graph,
                IdentityScorer(), {"matching": {"distinctive_token_guard": True}},
            )
            traced = {row["adm_party_id"]: row for row in read_jsonl(trace_path)}
            self.assertEqual(result["traced_misses"], 4)
            for case_id, expected_bucket, roots in cases:
                self.assertEqual(traced[case_id]["bucket"], expected_bucket)
                self.assertEqual([item.feature_score for item in proposals[case_id]],
                                 [score for _, score, _ in roots])


if __name__ == "__main__":
    unittest.main()
