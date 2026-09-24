import math
import unittest

from party_matching.domain import MatchProposal, PartyRecord
from party_matching.graph import VerifiedGraph
from party_matching.matching import _rules_containment_evidence, decide_records


class StubScorer:
    def __init__(self, idf, plain_threshold):
        self.idf = idf
        self.plain_threshold = plain_threshold
        self.system_threshold = 0.80

    def score_target_vectors(self, vectors, batch_size=50_000):
        return [0.45] * len(vectors)


class StubReranker:
    def score(self, proposals):
        return None


class ContainmentTests(unittest.TestCase):
    def setUp(self):
        self.name = "Carahsoft Technology Corp."
        self.idf = {"carahsoft": math.log(101 / 2) + 1}
        self.proposal = MatchProposal(
            adm_party_id="raw-1", mention_id="raw-1:m0", mention_text="Carahsoft",
            connector_before=None, connector_after=None,
            owner_party_id="verified-1", owner_party_name=self.name,
            root_party_id="verified-1", root_party_name=self.name,
            matched_name=self.name, candidate_type="official", candidate_source="official",
            candidate_confidence=None, feature_score=0.45, rules_score=0.45,
            char_tfidf_score=0.60, word_tfidf_score=0.30,
        )
        self.config = {
            "matching": {"cross_encoder_mode": "off", "minimum_root_margin": 0.04},
            "decision": {"rules_containment_enabled": True,
                         "rules_containment_min_confidence": 0.40,
                         "rules_containment_min_margin": 0.08,
                         "rules_containment_min_lexical": 0.50,
                         "rules_containment_max_token_df": 3},
        }

    def test_evidence_requires_rarity_and_margin(self):
        self.assertTrue(_rules_containment_evidence(self.proposal, self.idf, 100, 0.45, self.config))
        self.assertFalse(_rules_containment_evidence(self.proposal, self.idf, 100, 0.05, self.config))
        common = {"carahsoft": math.log(101 / 21) + 1}
        self.assertFalse(_rules_containment_evidence(self.proposal, common, 100, 0.45, self.config))
        self.proposal.digit_conflict = True
        self.assertFalse(_rules_containment_evidence(self.proposal, self.idf, 100, 0.45, self.config))

    def test_only_enabled_rules_path_accepts(self):
        graph = VerifiedGraph("account", "unused-graph.json", load_existing=False)
        graph.add_parties([{"partyId": "verified-1", "partyName": self.name}])
        record = PartyRecord("account", "raw-1", "Carahsoft", 1)
        proposals = {"raw-1": [self.proposal]}
        reranker = StubReranker()

        enabled = decide_records([record], proposals, graph, StubScorer(self.idf, 0.80), reranker, self.config)[0]
        self.assertEqual((enabled.decision, enabled.decision_tier), ("MATCH", "RULES_UNIQUE_CONTAINMENT"))
        self.assertAlmostEqual(enabled.confidence, 0.45)

        self.config["decision"]["rules_containment_enabled"] = False
        disabled = decide_records([record], proposals, graph, StubScorer(self.idf, 0.80), reranker, self.config)[0]
        self.assertEqual((disabled.decision, disabled.reason), ("NO_MATCH", "INSUFFICIENT_SUPPORT"))

        self.config["decision"]["rules_containment_enabled"] = True
        primary = decide_records([record], proposals, graph, StubScorer(self.idf, None), reranker, self.config)[0]
        self.assertEqual((primary.decision, primary.reason), ("NO_MATCH", "INSUFFICIENT_SUPPORT"))


if __name__ == "__main__":
    unittest.main()
