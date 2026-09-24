import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from party_matching.domain import FinalDecision, MatchProposal, PartyRecord
from party_matching.graph import VerifiedGraph
from party_matching.matching import FeatureScorer
from party_matching.reporting import _write_rules_changes
from party_matching.rules_enhancement import (
    _root_token_index, _unique_short_name, _unsafe_removed,
    enhance_rules_decisions, name_views, soft_token_coverage,
)


class EnhancedRulesTests(unittest.TestCase):
    def setUp(self):
        self.graph = VerifiedGraph("account", "unused-graph.json", load_existing=False)
        self.graph.add_parties([
            {"partyId": "carahsoft", "partyName": "Carahsoft Technology Corp."},
            {"partyId": "cooper", "partyName": "Cooper Holdings, Inc."},
        ])
        self.scorer = FeatureScorer("nonexistent-test-artifact.joblib")
        self.scorer.plain_threshold = 0.8
        self.scorer.system_threshold = 0.9
        self.config = {"matching": {"cross_encoder_mode": "off", "minimum_root_margin": 0.04},
                       "decision": {"rules_containment_enabled": False}}

    def proposal(self, adm_id, raw, root, name):
        return MatchProposal(
            adm_party_id=adm_id, mention_id=f"{adm_id}:m0", mention_text=raw,
            connector_before=None, connector_after=None,
            owner_party_id=root, owner_party_name=name,
            root_party_id=root, root_party_name=name, matched_name=name,
            candidate_type="official", candidate_source="official", candidate_confidence=None,
            feature_score=0.65, rules_score=0.65, char_tfidf_score=0.6,
            word_tfidf_score=0.5,
        )

    def rejected(self, adm_id, raw, root, name, score=0.65):
        return FinalDecision("account", adm_id, raw, "NO_MATCH", score,
                             "INSUFFICIENT_SUPPORT", verified_party_id=root,
                             verified_party_name=name, matched_member_id=root,
                             matched_member_name=name, matched_candidate_name=name,
                             matched_candidate_type="official", margin=0.2)

    def test_views_and_soft_coverage(self):
        self.assertIn(("first_segment", "Carahsoft Technology", "Partner"),
                      name_views("Carahsoft Technology - Partner"))
        self.assertIn(("compact_spacing", "docusign", ""), name_views("Docu Sign"))
        raw, candidate = soft_token_coverage("Carahsoft Technologies", "Carahsoft Technology Corp.", {})
        self.assertGreater(raw, 0.9)
        self.assertGreater(candidate, 0.9)

    def test_root_unique_short_name_requires_one_root(self):
        record = PartyRecord("account", "raw", "Carahsoft", 1)
        name = "Carahsoft Technology Corp."
        baseline = self.rejected("raw", record.raw_name, "carahsoft", name, 0.692)
        proposals = [self.proposal("raw", record.raw_name, "carahsoft", name)]
        self.assertIsNotNone(_unique_short_name(record, baseline, proposals,
                                                 _root_token_index(self.graph), 0.0))
        self.graph.add_parties([{"partyId": "other", "partyName": "Carahsoft Europe"}])
        self.assertIsNone(_unique_short_name(record, baseline, proposals,
                                              _root_token_index(self.graph), 0.0))

    def test_brandlike_suffix_is_unsafe(self):
        self.assertTrue(_unsafe_removed("SpringCM", "cooper", [], {}))
        self.assertFalse(_unsafe_removed("Partner", "carahsoft", [], {}))

    def test_only_rejected_plain_rows_are_enhanced(self):
        carahsoft = PartyRecord("account", "raw-c", "Carahsoft Technology - Partner", 1)
        cooper = PartyRecord("account", "raw-o", "Cooper Holdings, Inc. - SpringCM", 2)
        connector = PartyRecord("account", "raw-obo", "Carahsoft OBO Partner", 3)
        records = [carahsoft, cooper, connector]
        names = {"carahsoft": "Carahsoft Technology Corp.", "cooper": "Cooper Holdings, Inc."}
        proposals = {
            "raw-c": [self.proposal("raw-c", carahsoft.raw_name, "carahsoft", names["carahsoft"])],
            "raw-o": [self.proposal("raw-o", cooper.raw_name, "cooper", names["cooper"])],
            "raw-obo": [self.proposal("raw-obo", connector.raw_name, "carahsoft", names["carahsoft"])],
        }
        baseline = [self.rejected(r.adm_party_id, r.raw_name,
                                  "cooper" if r is cooper else "carahsoft",
                                  names["cooper" if r is cooper else "carahsoft"])
                    for r in records]
        result, stats = enhance_rules_decisions(
            records, proposals, baseline, self.graph,
            SimpleNamespace(char_vectorizer=None, word_vectorizer=None),
            self.scorer, self.config, {}, {"confidenceCutoff": 0.0},
        )
        self.assertEqual([item.decision for item in result], ["MATCH", "NO_MATCH", "NO_MATCH"])
        self.assertEqual(stats["safe_view_matches"], 1)
        self.assertEqual(baseline[0].decision, "NO_MATCH")

    def test_changed_rows_audit_marks_correctness_without_touching_excel(self):
        rows = [{"source": {"raw": "Carahsoft Technology - Partner"},
                 "prediction": {"ADM Party ID": "raw-c", "Dataset Split": "TEST_UNSEEN",
                                "Expected Canonical Name": "Carahsoft Technology Corp.",
                                "Expected Verified ID": "carahsoft", "Scorable": True}}]
        prior = {"raw-c": {"decision": "NO_MATCH"}}
        enhanced = {"raw-c": {"decision": "MATCH", "verified_party_id": "carahsoft",
                              "verified_party_name": "Carahsoft Technology Corp.",
                              "decision_tier": "RULES_SAFE_VIEW_FIRST_SEGMENT"}}
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "rules_changes.csv"
            _write_rules_changes(output, rows, {"raw_name_column": "raw"}, prior, enhanced)
            self.assertIn("CORRECT", output.read_text(encoding="utf-8-sig"))
            self.assertIn("Carahsoft Technology - Partner", output.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
