import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from party_matching.domain import FinalDecision, MatchProposal, PartyRecord, write_jsonl
from party_matching.graph import VerifiedGraph
from party_matching.rules_multipart import choose_multipart_rules, multipart_fragments


class MultipartRulesTests(unittest.TestCase):
    def test_fragment_extraction_preserves_brand_hyphens(self):
        self.assertEqual(multipart_fragments("Snap-on | Challenger"),
                         [("pipe", 0, "Snap-on"), ("pipe", 1, "Challenger")])
        self.assertEqual(multipart_fragments("Inhabit IQ (ResMan)"),
                         [("parentheses", 0, "Inhabit IQ"), ("parentheses", 1, "ResMan")])
        self.assertEqual(multipart_fragments("Snap-on"), [])

    def test_training_position_prior_chooses_one_root_without_test_label_leakage(self):
        with TemporaryDirectory() as directory:
            graph = VerifiedGraph("a", Path(directory) / "graph.json", load_existing=False)
            graph.add_parties([{"partyId": "left", "partyName": "Snap-on Incorporated"},
                               {"partyId": "right", "partyName": "Challenger Lifts, Inc."}])
            rows = [PartyRecord("a", f"r{i}", "Snap-on | Challenger", i) for i in range(21)]
            old = [FinalDecision("a", row.adm_party_id, row.raw_name, "NO_MATCH", 0.0,
                                 "AMBIGUOUS_FINAL_TARGETS") for row in rows]
            labels_path = Path(directory) / "labels.jsonl"
            write_jsonl(labels_path, [
                {"adm_party_id": row.adm_party_id, "scorable": True,
                 "split": "TRAIN" if i < 20 else "TEST_UNSEEN",
                 "expected_party_id": "right" if i < 20 else "left"}
                for i, row in enumerate(rows)
            ])

            def proposal(fragment, root, name):
                return MatchProposal(
                    adm_party_id=fragment.adm_party_id, mention_id=f"{fragment.adm_party_id}:m0",
                    mention_text=fragment.raw_name, connector_before=None, connector_after=None,
                    owner_party_id=root, owner_party_name=name, root_party_id=root,
                    root_party_name=name, matched_name=name, candidate_type="official",
                    candidate_source="official", candidate_confidence=None,
                    rules_score=0.99, exact=1.0, char_tfidf_score=0.99,
                )

            def fake_collect(_graph, retriever, _scorer):
                result = {}
                for fragment in retriever.records:
                    root = "left" if fragment.adm_party_id.endswith(":0") else "right"
                    result[fragment.adm_party_id] = [proposal(fragment, root, graph.nodes[root].party_name)]
                return result, {"scored_proposals": len(result), "retrieval_seconds": 0.0}

            class FakeRetriever:
                def __init__(self, records, _config, workers=1):
                    self.records = records

            scorer = type("Scorer", (), {"plain_threshold": 0.8})()
            with patch("party_matching.matching.MentionRetriever", FakeRetriever), \
                 patch("party_matching.matching.collect_proposals", fake_collect):
                result, stats = choose_multipart_rules(
                    rows, old, graph, scorer, {}, 1, labels_path, {}, {"confidenceCutoff": 0.0},
                )
            self.assertEqual(result[-1].verified_party_id, "right")
            self.assertEqual(result[-1].matched_mention, "Challenger")
            self.assertEqual(stats["learned_position_counts"]["pipe"], {1: 20})
            self.assertEqual(stats["changed"], 21)


if __name__ == "__main__":
    unittest.main()
