import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from party_matching.domain import FinalDecision, MatchProposal, PartyRecord
from party_matching.graph import VerifiedGraph
from party_matching.matching import FeatureScorer
from party_matching.reporting import _write_rules_changes
from party_matching.rules_enhancement import (
    _anchored_official_prefix, _minor_spelling_variant, _recover_regional, _recover_second_pass,
    _regional_qualifier_evidence, _root_prefix_index, _root_token_index, _laboratory_base,
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
        self.config["decision"].update({"rules_second_pass_enabled": True,
                                         "rules_identity_tiebreak_enabled": True})
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

    def test_explicit_legal_form_breaks_only_a_distinct_form_tie(self):
        self.graph.add_parties([
            {"partyId": "tock-inc", "partyName": "Tock, Inc."},
            {"partyId": "tock-llc", "partyName": "Tock, LLC"},
        ])
        raw = "Tock Inc"
        record = PartyRecord("account", "tock-raw", raw, 1)
        inc = self.proposal(record.adm_party_id, raw, "tock-inc", "Tock, Inc.")
        llc = self.proposal(record.adm_party_id, raw, "tock-llc", "Tock, LLC")
        inc.rules_score, llc.rules_score = 0.95, 0.93
        old = self.rejected(record.adm_party_id, raw, "tock-llc", "Tock, LLC")
        old.reason, old.runner_up_party_id, old.margin = "AMBIGUOUS_FINAL_TARGETS", "tock-inc", 0.02
        proposals = {record.adm_party_id: [inc, llc]}
        retriever = SimpleNamespace(char_vectorizer=None, word_vectorizer=None)
        self.config["decision"].update({"rules_second_pass_enabled": True,
                                         "rules_identity_tiebreak_enabled": False})
        disabled, _ = enhance_rules_decisions(
            [record], proposals, [old], self.graph, retriever, self.scorer,
            self.config, {}, {"confidenceCutoff": 0.0},
        )
        self.assertEqual(disabled[0].decision, "NO_MATCH")
        self.config["decision"]["rules_identity_tiebreak_enabled"] = True
        enabled, stats = enhance_rules_decisions(
            [record], proposals, [old], self.graph, retriever, self.scorer,
            self.config, {}, {"confidenceCutoff": 0.0},
        )
        self.assertEqual(enabled[0].verified_party_id, "tock-inc")
        self.assertEqual(enabled[0].decision_tier, "RULES_SECOND_PASS_EXPLICIT_LEGAL_FORM")
        self.assertEqual(enabled[0].confidence, inc.rules_score)
        self.assertEqual(stats["second_pass"]["explicit_legal_form"], 1)
        self.assertEqual(old.decision, "NO_MATCH")

    def test_legal_form_tie_abstains_when_form_is_unstated_or_equivalent(self):
        self.graph.add_parties([
            {"partyId": "sodexo", "partyName": "Sodexo"},
            {"partyId": "sodexo-inc", "partyName": "Sodexo, Inc."},
            {"partyId": "qbs-ltd", "partyName": "QBS Software Ltd"},
            {"partyId": "qbs-limited", "partyName": "QBS Software Limited"},
        ])
        for adm_id, raw, roots, names in (
            ("sodexo-raw", "Sodexo", ("sodexo", "sodexo-inc"), ("Sodexo", "Sodexo, Inc.")),
            ("qbs-raw", "QBS Software Ltd", ("qbs-ltd", "qbs-limited"),
             ("QBS Software Ltd", "QBS Software Limited")),
        ):
            with self.subTest(raw=raw):
                record = PartyRecord("account", adm_id, raw, 1)
                proposals = [self.proposal(adm_id, raw, root, name)
                             for root, name in zip(roots, names)]
                old = self.rejected(adm_id, raw, roots[0], names[0])
                old.reason, old.runner_up_party_id, old.margin = "AMBIGUOUS_FINAL_TARGETS", roots[1], 0.02
                enhanced = {adm_id: old}
                previous, _ = _recover_second_pass(
                    [record], {adm_id: proposals}, enhanced, self.graph, self.scorer,
                    {}, {"confidenceCutoff": 0.0}, _root_token_index(self.graph),
                    _root_prefix_index(self.graph, {raw}), enable_identity_tiebreak=True,
                )
                self.assertFalse(previous)
                self.assertIs(enhanced[adm_id], old)

    def test_and_equivalence_requires_one_official_root(self):
        self.graph.add_parties([
            {"partyId": "loveday", "partyName": "Loveday & Partners Ltd"},
        ])
        raw = "Loveday and Partners Ltd"
        record = PartyRecord("account", "loveday-raw", raw, 1)
        proposal = self.proposal(record.adm_party_id, raw, "loveday", "Loveday & Partners Ltd")
        proposal.rules_score = 0.7996
        old = self.rejected(record.adm_party_id, raw, "loveday", "Loveday & Partners Ltd")
        enhanced = {record.adm_party_id: old}
        args = ([record], {record.adm_party_id: [proposal]}, enhanced, self.graph,
                self.scorer, {}, {"confidenceCutoff": 0.0}, _root_token_index(self.graph),
                _root_prefix_index(self.graph, {raw}))
        _recover_second_pass(*args)
        self.assertIs(enhanced[record.adm_party_id], old)
        previous, stats = _recover_second_pass(*args, enable_identity_tiebreak=True)
        self.assertIs(previous[record.adm_party_id], old)
        self.assertEqual(enhanced[record.adm_party_id].verified_party_id, "loveday")
        self.assertEqual(enhanced[record.adm_party_id].decision_tier,
                         "RULES_SECOND_PASS_AND_EQUIVALENCE")
        self.assertEqual(stats["and_equivalence"], 1)
        self.graph.add_parties([
            {"partyId": "loveday-other", "partyName": "Loveday and Partners Ltd"},
        ])
        other = self.proposal(record.adm_party_id, raw, "loveday-other", "Loveday and Partners Ltd")
        old.reason, old.runner_up_party_id = "AMBIGUOUS_FINAL_TARGETS", "loveday-other"
        enhanced[record.adm_party_id] = old
        previous, _ = _recover_second_pass(
            [record], {record.adm_party_id: [proposal, other]}, enhanced,
            self.graph, self.scorer, {}, {"confidenceCutoff": 0.0},
            _root_token_index(self.graph), _root_prefix_index(self.graph, {raw}),
            enable_identity_tiebreak=True,
        )
        self.assertFalse(previous)
        self.assertIs(enhanced[record.adm_party_id], old)

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

    def test_labs_laboratories_is_a_unique_name_variant_not_a_family_merge(self):
        self.graph.add_parties([{"partyId": "abbott", "partyName": "Abbott Laboratories"}])
        raw = "Abbott Labs"
        record = PartyRecord("account", "abbott-raw", raw, 1)
        proposal = self.proposal("abbott-raw", raw, "abbott", "Abbott Laboratories")
        proposal.rules_score = 0.65
        old = self.rejected("abbott-raw", raw, "abbott", "Abbott Laboratories")
        enhanced = {"abbott-raw": old}
        args = ([record], {"abbott-raw": [proposal]}, enhanced, self.graph,
                self.scorer, {}, {"confidenceCutoff": 0.0}, _root_token_index(self.graph),
                _root_prefix_index(self.graph, {raw}))
        _recover_second_pass(*args, enable_identity_tiebreak=True)
        self.assertEqual(enhanced["abbott-raw"].decision_tier,
                         "RULES_SECOND_PASS_LABORATORY_ABBREVIATION")

        # Molecular/Nutrition are different names, not mere abbreviations.
        self.assertNotEqual(
            _laboratory_base("Abbott Molecular"), _laboratory_base("Abbott Laboratories")
        )
        self.graph.add_parties([{"partyId": "abbott-labs", "partyName": "Abbott Labs"}])
        enhanced["abbott-raw"] = old
        previous, _ = _recover_second_pass(*args, enable_identity_tiebreak=True)
        self.assertFalse(previous)
        self.assertIs(enhanced["abbott-raw"], old)

    def test_leading_the_is_local_equivalence_only_when_root_is_unique(self):
        self.graph.add_parties([{"partyId": "foundation", "partyName": "The Mastercard Foundation"}])
        raw = "Mastercard Foundation"
        record = PartyRecord("account", "foundation-raw", raw, 1)
        proposal = self.proposal("foundation-raw", raw, "foundation", "The Mastercard Foundation")
        proposal.rules_score = 0.799
        old = self.rejected("foundation-raw", raw, "foundation", "The Mastercard Foundation")
        enhanced = {"foundation-raw": old}
        args = ([record], {"foundation-raw": [proposal]}, enhanced, self.graph,
                self.scorer, {}, {"confidenceCutoff": 0.0}, _root_token_index(self.graph),
                _root_prefix_index(self.graph, {raw}))
        _recover_second_pass(*args, enable_identity_tiebreak=True)
        self.assertEqual(enhanced["foundation-raw"].decision_tier,
                         "RULES_SECOND_PASS_LEADING_ARTICLE_EQUIVALENCE")
        self.graph.add_parties([{"partyId": "foundation-other", "partyName": raw}])
        enhanced["foundation-raw"] = old
        previous, _ = _recover_second_pass(*args, enable_identity_tiebreak=True)
        self.assertFalse(previous)
        self.assertIs(enhanced["foundation-raw"], old)

    def test_regional_route_adds_only_unambiguous_geographic_variant(self):
        self.graph.add_parties([{"partyId": "acme", "partyName": "Acme Corp."}])
        raw = "Acme Brazil Corp"
        record = PartyRecord("account", "regional", raw, 1)
        proposal = self.proposal("regional", raw, "acme", "Acme Corp.")
        old = self.rejected("regional", raw, "acme", "Acme Corp.")
        enhanced = {"regional": old}
        previous, stats = _recover_regional(
            [record], {"regional": [proposal]}, enhanced, self.graph,
            {}, {"confidenceCutoff": 0.0},
        )
        self.assertIs(previous["regional"], old)
        self.assertEqual(stats["matches"], 1)
        self.assertEqual(enhanced["regional"].verified_party_id, "acme")
        self.assertEqual(enhanced["regional"].confidence, proposal.rules_score)
        self.assertEqual(enhanced["regional"].match_method,
                         "rules:regional_heuristic_not_proven_ownership")
        self.assertEqual(enhanced["regional"].reason,
                         "REGIONAL_HEURISTIC_NOT_PROVEN_OWNERSHIP")

    def test_regional_route_rejects_unknown_context_and_weak_lexical_signal(self):
        self.graph.add_parties([{"partyId": "acme", "partyName": "Acme Corp."}])
        for raw in ("Acme Brazil Analytics Corp", "Acme Brazil Holdings Corp", "Acme Division Corp"):
            item = self.proposal("raw", raw, "acme", "Acme Corp.")
            self.assertFalse(_regional_qualifier_evidence(
                raw, item, {("acme",): {"acme"}},
            ))
        weak = self.proposal("raw", "Acme Brazil Corp", "acme", "Acme Corp.")
        weak.char_tfidf_score = weak.word_tfidf_score = 0.54
        self.assertFalse(_regional_qualifier_evidence(
            "Acme Brazil Corp", weak, {("acme",): {"acme"}},
        ))

    def test_regional_route_abstains_for_separate_verified_family_roots(self):
        self.graph.add_parties([
            {"partyId": "ingram-inc", "partyName": "Ingram Micro Inc."},
            {"partyId": "ingram-other", "partyName": "Ingram Micro"},
        ])
        raw = "Ingram Micro USA"
        record = PartyRecord("account", "ingram-regional", raw, 1)
        proposal = self.proposal(record.adm_party_id, raw, "ingram-inc", "Ingram Micro Inc.")
        old = self.rejected(record.adm_party_id, raw, "ingram-inc", "Ingram Micro Inc.")
        enhanced = {record.adm_party_id: old}
        previous, _ = _recover_regional(
            [record], {record.adm_party_id: [proposal]}, enhanced, self.graph,
            {}, {"confidenceCutoff": 0.0},
        )
        self.assertFalse(previous)
        self.assertIs(enhanced[record.adm_party_id], old)

    def test_regional_flag_is_additive_to_existing_enhancement(self):
        self.graph.add_parties([{"partyId": "acme", "partyName": "Acme Corp."}])
        raw = "Acme Brazil Corp"
        record = PartyRecord("account", "regional", raw, 1)
        proposal = self.proposal("regional", raw, "acme", "Acme Corp.")
        old = self.rejected("regional", raw, "acme", "Acme Corp.")
        retriever = SimpleNamespace(char_vectorizer=None, word_vectorizer=None)
        disabled, _ = enhance_rules_decisions(
            [record], {"regional": [proposal]}, [old], self.graph, retriever,
            self.scorer, self.config, {}, {"confidenceCutoff": 0.0},
        )
        self.config["decision"]["rules_regional_enabled"] = True
        enabled, stats = enhance_rules_decisions(
            [record], {"regional": [proposal]}, [old], self.graph, retriever,
            self.scorer, self.config, {}, {"confidenceCutoff": 0.0},
        )
        self.assertEqual(disabled[0].decision, "NO_MATCH")
        self.assertEqual(enabled[0].decision, "MATCH")
        self.assertEqual(stats["regional"]["matches"], 1)
        self.assertIs(stats["_regional_previous"]["regional"], old)

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
