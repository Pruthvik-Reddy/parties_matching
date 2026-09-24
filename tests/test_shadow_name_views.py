"""TEMPORARY SHADOW EXPERIMENT tests; remove with the experimental module."""

import unittest

from party_matching.shadow_name_views import name_views


class NameViewTests(unittest.TestCase):
    def test_generic_views_are_bounded_and_deduplicated(self):
        views = name_views("Hines - West Region")
        self.assertIn(("first_segment", "Hines"), views)
        self.assertIn(("last_segment", "West Region"), views)
        self.assertLessEqual(len(views), 3)
        self.assertIn(("first_segment", "Hines"), name_views("Hines--Europe"))

    def test_typo_and_spacing_views_do_not_require_entity_names(self):
        self.assertIn(("compact_spacing", "docusign"), name_views("Docu Sign"))
        self.assertIn(("plural_ies", "carahsoft technology"), name_views("Carahsoft Technologies"))

    def test_simple_name_has_no_partial_view(self):
        self.assertEqual(name_views("Carahsoft"), [])


if __name__ == "__main__":
    unittest.main()
