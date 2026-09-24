import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from party_matching.domain import FinalDecision, MatchProposal, PartyRecord
from party_matching.graph import VerifiedGraph
from party_matching.matching import FeatureScorer
from party_matching.reporting import _write_rules_changes
from party_matching.rules_enhancement import (
    _anchored_official_prefix, _minor_spelling_variant, _recover_second_pass, _root_prefix_index, _root_token_index,
    _unique_short_name, _unsafe_removed,
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
                                                 _root_token_index(self.graph), {}, 0.0))
        self.graph.add_parties([{"partyId": "other", "partyName": "Carahsoft Europe"}])
        self.assertIsNone(_unique_short_name(record, baseline, proposals,
                                              _root_token_index(self.graph), {}, 0.0))

    def test_multiword_short_name_requires_one_verified_root(self):
        self.graph.add_parties([{"partyId": "ingram-inc", "partyName": "Ingram Micro Inc."}])
        record = PartyRecord("account", "ingram-raw", "Ingram Micro", 1)
        name = "Ingram Micro Inc."
        baseline = self.rejected(record.adm_party_id, record.raw_name, "ingram-inc", name)
        proposals = [self.proposal(record.adm_party_id, record.raw_name, "ingram-inc", name)]
        prefixes = _root_prefix_index(self.graph, {record.raw_name})
        self.assertIsNotNone(_unique_short_name(record, baseline, proposals,
                                                 _root_token_index(self.graph), prefixes, 0.0))
        self.graph.add_parties([{"partyId": "ingram-other", "partyName": "Ingram Micro"}])
        prefixes = _root_prefix_index(self.graph, {record.raw_name})
        self.assertIsNone(_unique_short_name(record, baseline, proposals,
                                              _root_token_index(self.graph), prefixes, 0.0))

    def test_brandlike_suffix_is_unsafe(self):
        self.assertTrue(_unsafe_removed("SpringCM", "cooper", [], {}))
        self.assertFalse(_unsafe_removed("Partner", "carahsoft", [], {}))

    def test_complete_official_name_prefix_not_just_shared_prefix(self):
        self.assertEqual(
            _anchored_official_prefix("Ingram Micro Inc. LATAM Export Division", "Ingram Micro Inc."),
            ("Ingram Micro Inc", "LATAM Export Division"),
        )
        self.assertIsNone(_anchored_official_prefix("Ingram Micro Marketplace", "Ingram Micro Inc."))
        self.assertIsNone(_anchored_official_prefix("Carahsoft", "Carahsoft Technology Corp."))

    def test_core_anchor_supplements_existing_enhancement_and_vetoes_competitor(self):
        self.graph.add_parties([
            {"partyId": "ingram", "partyName": "Ingram Micro Inc."},
            {"partyId": "seal", "partyName": "Seal Software Limited"},
        ])
        old = PartyRecord("account", "old", "Carahsoft Technology - Partner", 1)
        core = PartyRecord("account", "core", "Ingram Micro Inc. LATAM Export Division", 2)
        conflict = PartyRecord("account", "conflict",
                               "Cooper Holdings, Inc. as successor to Seal Software Limited", 3)
        proposals = {
            "old": [self.proposal("old", old.raw_name, "carahsoft", "Carahsoft Technology Corp.")],
            "core": [self.proposal("core", core.raw_name, "ingram", "Ingram Micro Inc.")],
            "conflict": [self.proposal("conflict", conflict.raw_name, "cooper", "Cooper Holdings, Inc."),
                         self.proposal("conflict", conflict.raw_name, "seal", "Seal Software Limited")],
        }
        baseline = [self.rejected("old", old.raw_name, "carahsoft", "Carahsoft Technology Corp."),
                    self.rejected("core", core.raw_name, "ingram", "Ingram Micro Inc."),
                    self.rejected("conflict", conflict.raw_name, "cooper", "Cooper Holdings, Inc.")]
        result, stats = enhance_rules_decisions(
            [old, core, conflict], proposals, baseline, self.graph,
            SimpleNamespace(char_vectorizer=None, word_vectorizer=None),
            self.scorer, self.config, {}, {"confidenceCutoff": 0.0},
        )
        self.assertEqual([item.decision for item in result], ["MATCH", "MATCH", "NO_MATCH"])
        self.assertEqual(result[0].decision_tier, "RULES_SAFE_VIEW_FIRST_SEGMENT")
        self.assertEqual(result[1].decision_tier, "RULES_SAFE_VIEW_CORE_ANCHOR")
        self.assertEqual(stats["core_anchor_matches"], 1)
        self.assertEqual(stats["unsafe_core_anchors_skipped"], 1)

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

    def test_second_pass_recovers_other_retrieved_root_without_fabricating_score(self):
        raw = "Carahsoft Technology Corp. - Partner"
        record = PartyRecord("account", "second", raw, 1)
        wrong = self.proposal("second", raw, "cooper", "Cooper Holdings, Inc.")
        correct = self.proposal("second", raw, "carahsoft", "Carahsoft Technology Corp.")
        old = self.rejected("second", raw, "cooper", "Cooper Holdings, Inc.")
        enhanced = {"second": old}
        previous, stats = _recover_second_pass(
            [record], {"second": [wrong, correct]}, enhanced, self.graph,
            self.scorer, {}, {"confidenceCutoff": 0.0},
            _root_token_index(self.graph), _root_prefix_index(self.graph, {raw}),
        )
        self.assertIs(previous["second"], old)
        self.assertEqual(enhanced["second"].verified_party_id, "carahsoft")
        self.assertEqual(enhanced["second"].confidence, correct.rules_score)
        self.assertEqual(stats["official_name_with_context"], 1)

    def test_second_pass_abstains_on_competing_name_and_connector(self):
        raw = "Cooper Holdings, Inc. - Carahsoft Technology Corp."
        record = PartyRecord("account", "conflict", raw, 1)
        connector = PartyRecord("account", "connector", "Carahsoft OBO Cooper", 2)
        proposals = {
            "conflict": [self.proposal("conflict", raw, "cooper", "Cooper Holdings, Inc."),
                         self.proposal("conflict", raw, "carahsoft", "Carahsoft Technology Corp.")],
            "connector": [self.proposal("connector", connector.raw_name, "carahsoft",
                                        "Carahsoft Technology Corp.")],
        }
        enhanced = {"conflict": self.rejected("conflict", raw, "cooper", "Cooper Holdings, Inc."),
                    "connector": self.rejected("connector", connector.raw_name, "carahsoft",
                                               "Carahsoft Technology Corp.")}
        previous, _ = _recover_second_pass(
            [record, connector], proposals, enhanced, self.graph, self.scorer, {},
            {"confidenceCutoff": 0.0}, _root_token_index(self.graph),
            _root_prefix_index(self.graph, {raw, connector.raw_name}),
        )
        self.assertFalse(previous)
        self.assertEqual(enhanced["conflict"].decision, "NO_MATCH")
        self.assertEqual(enhanced["connector"].decision, "NO_MATCH")

    def test_second_pass_legal_suffix_does_not_collapse_separate_roots(self):
        self.graph.add_parties([{"partyId": "qbs-limited", "partyName": "QBS Software Limited"}])
        raw = "QBS Software Ltd"
        record = PartyRecord("account", "qbs-raw", raw, 1)
        proposal = self.proposal(record.adm_party_id, raw, "qbs-limited", "QBS Software Limited")
        old = self.rejected(record.adm_party_id, raw, "qbs-limited", "QBS Software Limited")
        enhanced = {record.adm_party_id: old}
        kwargs = ( [record], {record.adm_party_id: [proposal]}, enhanced, self.graph,
                   self.scorer, {}, {"confidenceCutoff": 0.0}, _root_token_index(self.graph),
                   _root_prefix_index(self.graph, {raw}) )
        _recover_second_pass(*kwargs)
        self.assertEqual(enhanced[record.adm_party_id].decision_tier,
                         "RULES_SECOND_PASS_LEGAL_SUFFIX_VARIANT")
        self.graph.add_parties([{"partyId": "qbs-ltd", "partyName": "QBS Software Ltd"}])
        enhanced[record.adm_party_id] = old
        _recover_second_pass(*kwargs)
        self.assertEqual(enhanced[record.adm_party_id].decision, "NO_MATCH")

    def test_second_pass_allows_only_a_small_complete_name_typo(self):
        raw = "Carasoft Technology"
        record = PartyRecord("account", "typo", raw, 1)
        proposal = self.proposal("typo", raw, "carahsoft", "Carahsoft Technology Corp.")
        proposal.rules_score = 0.50
        old = self.rejected("typo", raw, "carahsoft", "Carahsoft Technology Corp.")
        enhanced = {"typo": old}
        _recover_second_pass(
            [record], {"typo": [proposal]}, enhanced, self.graph, self.scorer, {},
            {"confidenceCutoff": 0.0}, _root_token_index(self.graph),
            _root_prefix_index(self.graph, {raw}),
        )
        self.assertEqual(enhanced["typo"].decision_tier,
                         "RULES_SECOND_PASS_MINOR_SPELLING_VARIANT")
        self.assertEqual(enhanced["typo"].confidence, 0.50)
        self.assertFalse(
            _minor_spelling_variant("Carasoft Technology Partner", "Carahsoft Technology Corp.", {})
        )

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
