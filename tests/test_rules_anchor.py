import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from party_matching.domain import FinalDecision, PartyRecord
from party_matching.graph import VerifiedGraph
from party_matching.matching import FeatureScorer, MentionRetriever
from party_matching.rules_anchor import _official_base, recover_verified_name_anchors


class VerifiedNameAnchorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.graph = VerifiedGraph("account", Path(self.temp.name) / "graph.json", load_existing=False)
        self.graph.add_parties([
            {"partyId": "pepsi", "partyName": "PepsiCo, Inc."},
            {"partyId": "ripe", "partyName": "Ripe Limited"},
            {"partyId": "home", "partyName": "Home Instead, Inc."},
            {"partyId": "adp", "partyName": "ADP Inc."},
            {"partyId": "ge", "partyName": "GE Precision Healthcare LLC"},
            {"partyId": "citizens", "partyName": "Citizens Bank, N.A."},
        ])
        self.scorer = FeatureScorer("nonexistent-test-artifact.joblib")
        self.scorer.model = None
        self.scorer.plain_threshold = 0.8

    def evaluate(self, names, decisions=None):
        records = [PartyRecord("account", str(i), name, i) for i, name in enumerate(names)]
        old = decisions or [FinalDecision("account", str(i), name, "NO_MATCH", 0.0,
                                          "INSUFFICIENT_SUPPORT") for i, name in enumerate(names)]
        retriever = MentionRetriever(records, {"embedding_enabled": False}, workers=1)
        result, stats = recover_verified_name_anchors(
            records, old, self.graph, {}, retriever, self.scorer,
            {}, {"confidenceCutoff": 0.0},
        )
        return result, stats

    def test_complete_official_and_legal_base_anchors(self):
        names = [
            "PepsiCo, Inc. Procurement", "PepsiCo, inc. Purchasing", "Ripe",
            "Home Instead Seattle", "Home Instead Senior Care Australia",
            "ADP Treasury", "GE Precision Healthcare,LLC-Mytech",
            "Citizens Bank", "Citizens Bank Treasury Solutions - CLM",
        ]
        result, stats = self.evaluate(names)
        self.assertEqual([item.verified_party_id for item in result],
                         ["pepsi", "pepsi", "ripe", "home", "home", "adp", "ge",
                          "citizens", "citizens"])
        self.assertEqual(stats["matches"], len(names))
        self.assertEqual(_official_base("Citizens Bank, N.A."), "citizens bank")

    def test_more_specific_root_and_unsupported_brand_variation_abstain(self):
        self.graph.add_parties([
            {"partyId": "senior", "partyName": "Home Instead Senior Care"},
            {"partyId": "pepsi-procurement", "partyName": "PepsiCo, Inc. Procurement LLC"},
        ])
        result, _ = self.evaluate([
            "Home Instead Senior Care", "PepsiCo, Inc. Procurement LLC",
            "Citizens Commercial Banking", "Ripe OBO Other Party",
        ])
        self.assertNotEqual(result[0].verified_party_id, "home")
        self.assertNotEqual(result[1].verified_party_id, "pepsi")
        self.assertEqual(result[2].decision, "NO_MATCH")
        self.assertEqual(result[3].decision, "NO_MATCH")

    def test_previous_matches_untouched_and_same_base_collision_abstains(self):
        self.graph.add_parties([{"partyId": "ripe-other", "partyName": "Ripe Inc."}])
        prior = [
            FinalDecision("account", "0", "PepsiCo, Inc - Foodservice", "MATCH", 0.92,
                          "MATCH", verified_party_id="pepsi"),
            FinalDecision("account", "1", "Ripe", "NO_MATCH", 0.0,
                          "AMBIGUOUS_FINAL_TARGETS"),
        ]
        result, _ = self.evaluate([item.raw_name for item in prior], prior)
        self.assertIs(result[0], prior[0])
        self.assertEqual(result[1].decision, "NO_MATCH")


if __name__ == "__main__":
    unittest.main()
